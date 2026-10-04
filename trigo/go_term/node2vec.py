from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BaseGOEncoder, GONamespaceEncoding, GOSubgraph, register_go_encoder

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable


def _edge_data_from_subgraph(
    subgraph: GOSubgraph,
    id_to_idx: Dict[str, int],
) -> tuple[Tensor, Tensor, tuple[str, ...]]:
    if not subgraph.edges:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            ("is_a",),
        )

    edge_pairs: List[tuple[int, int]] = []
    for s, t in subgraph.edges:
        if s not in id_to_idx or t not in id_to_idx:
            continue
        edge_pairs.append((id_to_idx[s], id_to_idx[t]))

    if not edge_pairs:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            ("is_a",),
        )

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)
    return edge_index, edge_type, ("is_a",)


def _build_undirected_adjacency(num_nodes: int, edge_index: Tensor) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        s = int(edge_index[0, i].item())
        t = int(edge_index[1, i].item())
        if t not in adj[s]:
            adj[s].append(t)
        if s not in adj[t]:
            adj[t].append(s)
    return adj


class _BiasedRandomWalker:
    def __init__(self, adj: List[List[int]], p: float, q: float):
        self.adj = adj
        self.p = p
        self.q = q
        self.neighbor_sets = [set(nei) for nei in adj]

    def _next_step(self, prev_node: int, curr_node: int) -> int:
        neighbors = self.adj[curr_node]
        if not neighbors:
            return curr_node
        if prev_node < 0:
            return random.choice(neighbors)

        weights = []
        prev_neighbors = self.neighbor_sets[prev_node]
        for nxt in neighbors:
            if nxt == prev_node:
                weights.append(1.0 / max(self.p, 1e-8))
            elif nxt in prev_neighbors:
                weights.append(1.0)
            else:
                weights.append(1.0 / max(self.q, 1e-8))
        return random.choices(neighbors, weights=weights, k=1)[0]

    def sample_walk(self, start_node: int, walk_length: int) -> List[int]:
        walk = [start_node]
        while len(walk) < walk_length:
            curr = walk[-1]
            if not self.adj[curr]:
                break
            prev = walk[-2] if len(walk) > 1 else -1
            nxt = self._next_step(prev, curr)
            walk.append(nxt)
        return walk

    def sample_walks(
        self,
        num_walks: int,
        walk_length: int,
        seed: int,
        show_progress: bool = False,
        desc: str = "random walks",
    ) -> List[List[int]]:
        random.seed(seed)
        nodes = list(range(len(self.adj)))
        walks: List[List[int]] = []
        walk_iter = range(num_walks)
        if show_progress:
            walk_iter = tqdm(walk_iter, desc=desc, unit="pass", leave=False)
        for _ in walk_iter:
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.sample_walk(node, walk_length))
        return walks


class _SkipGramNS(nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int):
        super().__init__()
        self.in_embed = nn.Embedding(num_nodes, embedding_dim)
        self.out_embed = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.in_embed.weight)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, src: Tensor, pos_ctx: Tensor, neg_ctx: Tensor) -> Tensor:
        src_v = self.in_embed(src)  # [B, D]
        pos_v = self.out_embed(pos_ctx)  # [B, D]
        neg_v = self.out_embed(neg_ctx)  # [B, K, D]

        pos_score = torch.sum(src_v * pos_v, dim=-1)  # [B]
        pos_loss = -F.logsigmoid(pos_score)

        neg_score = torch.bmm(neg_v, src_v.unsqueeze(-1)).squeeze(-1)  # [B, K]
        neg_loss = -torch.sum(F.logsigmoid(-neg_score), dim=-1)
        return (pos_loss + neg_loss).mean()

    def embeddings(self) -> Tensor:
        return self.in_embed.weight.detach()


@dataclass
class Node2VecConfig:
    embedding_dim: int = 256
    walk_length: int = 30
    context_size: int = 5
    walks_per_node: int = 10
    p: float = 1.0
    q: float = 1.0
    num_negative_samples: int = 5
    epochs: int = 5
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    directed_walk: bool = False
    device: str = "auto"
    show_progress: bool = True


def _pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _walks_to_skipgram_pairs(walks: Sequence[Sequence[int]], context_size: int) -> Tuple[Tensor, Tensor]:
    src_list: List[int] = []
    ctx_list: List[int] = []
    for walk in walks:
        n = len(walk)
        for i, center in enumerate(walk):
            left = max(0, i - context_size)
            right = min(n, i + context_size + 1)
            for j in range(left, right):
                if j == i:
                    continue
                src_list.append(center)
                ctx_list.append(walk[j])
    if not src_list:
        return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    return torch.tensor(src_list, dtype=torch.long), torch.tensor(ctx_list, dtype=torch.long)


def _build_negative_sampling_distribution(adj: List[List[int]]) -> Tensor:
    degrees = torch.tensor([max(len(n), 1) for n in adj], dtype=torch.float32)
    prob = torch.pow(degrees, 0.75)
    prob = prob / prob.sum()
    return prob


@register_go_encoder("node2vec_torch")
class TorchNode2VecEncoder(BaseGOEncoder):
    def __init__(self, config: Node2VecConfig | None = None):
        self.config = config or Node2VecConfig()

    def _prepare_graph(self, subgraph: GOSubgraph) -> tuple[Dict[str, int], Tensor, Tensor, tuple[str, ...], List[List[int]]]:
        id_to_idx = {go_id: idx for idx, go_id in enumerate(subgraph.term_ids)}
        edge_index, edge_type, relation_vocab = _edge_data_from_subgraph(subgraph, id_to_idx)
        if edge_index.size(1) == 0:
            # keep isolated nodes so downstream can still index every term
            adj = [[] for _ in subgraph.term_ids]
            return id_to_idx, edge_index, edge_type, relation_vocab, adj
        adj = _build_undirected_adjacency(len(subgraph.term_ids), edge_index)
        return id_to_idx, edge_index, edge_type, relation_vocab, adj

    def fit_transform(self, subgraph: GOSubgraph) -> GONamespaceEncoding:
        cfg = self.config
        device = _pick_device(cfg.device)
        id_to_idx, edge_index, edge_type, relation_vocab, adj = self._prepare_graph(subgraph)
        num_nodes = len(subgraph.term_ids)

        if num_nodes == 0:
            return GONamespaceEncoding(
                namespace=subgraph.namespace,
                term_ids=[],
                embeddings=torch.empty((0, cfg.embedding_dim), dtype=torch.float32),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_type=torch.empty((0,), dtype=torch.long),
                relation_vocab=("is_a",),
            )

        walker = _BiasedRandomWalker(adj=adj, p=cfg.p, q=cfg.q)
        walks = walker.sample_walks(
            num_walks=cfg.walks_per_node,
            walk_length=cfg.walk_length,
            seed=cfg.seed,
            show_progress=cfg.show_progress,
            desc=f"[{subgraph.namespace}] random walks",
        )
        src, pos = _walks_to_skipgram_pairs(walks, context_size=cfg.context_size)

        if src.numel() == 0:
            # degenerate graph; fallback to random embeddings
            embeds = torch.randn(num_nodes, cfg.embedding_dim) * 0.01
            return GONamespaceEncoding(
                namespace=subgraph.namespace,
                term_ids=subgraph.term_ids,
                embeddings=embeds,
                edge_index=edge_index,
                edge_type=edge_type,
                relation_vocab=relation_vocab,
            )

        model = _SkipGramNS(num_nodes=num_nodes, embedding_dim=cfg.embedding_dim).to(device)
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        neg_prob = _build_negative_sampling_distribution(adj).to(device)

        permutation = torch.arange(src.size(0))
        epoch_iter = range(cfg.epochs)
        if cfg.show_progress:
            epoch_iter = tqdm(epoch_iter, desc=f"[{subgraph.namespace}] node2vec", unit="epoch")
        for epoch in epoch_iter:
            g = torch.Generator().manual_seed(cfg.seed + epoch)
            permutation = permutation[torch.randperm(permutation.numel(), generator=g)]

            batch_iter = range(0, permutation.numel(), cfg.batch_size)
            if cfg.show_progress:
                batch_iter = tqdm(
                    batch_iter,
                    desc=f"[{subgraph.namespace}] epoch {epoch + 1}/{cfg.epochs}",
                    unit="batch",
                    leave=False,
                )
            for start in batch_iter:
                idx = permutation[start : start + cfg.batch_size]
                batch_src = src[idx].to(device, non_blocking=True)
                batch_pos = pos[idx].to(device, non_blocking=True)
                batch_neg = torch.multinomial(
                    neg_prob,
                    num_samples=batch_src.size(0) * cfg.num_negative_samples,
                    replacement=True,
                ).view(batch_src.size(0), cfg.num_negative_samples)

                loss = model(batch_src, batch_pos, batch_neg)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        embeddings = model.embeddings().cpu()
        return GONamespaceEncoding(
            namespace=subgraph.namespace,
            term_ids=subgraph.term_ids,
            embeddings=embeddings,
            edge_index=edge_index,
            edge_type=edge_type,
            relation_vocab=relation_vocab,
        )

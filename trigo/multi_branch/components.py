from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

#mlp
class MLPExpert(nn.Module):
    """
    Generic vector expert for branch-specific/shared branches.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

#FFN
class NamespaceResidualBlock(nn.Module):
    """
    Residual FFN block for branch experts.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.ffn(self.norm(x))


class NamespacePrivateAdapter(nn.Module):
    """
    Lightweight namespace-private residual adapter.

    It gives each BP/MF/CC branch a small private correction path after the
    first branch fusion, while keeping the shared protein/GO pathways intact.
    """

    def __init__(self, dim: int, hidden_dim: int = 128, dropout: float = 0.1, residual_init: float = 0.1):
        super().__init__()
        hidden_dim = max(1, int(hidden_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        init = float(min(max(residual_init, 1e-4), 1.0 - 1e-4))
        self.residual_logit = nn.Parameter(torch.logit(torch.tensor(init, dtype=torch.float32)))

    def residual_scale(self) -> Tensor:
        return torch.sigmoid(self.residual_logit)

    def forward(self, x: Tensor) -> Tensor:
        scale = self.residual_scale().to(device=x.device, dtype=x.dtype)
        return x + scale * self.net(x)


class NamespaceConditionedExpert(nn.Module):
    """
    Stronger namespace expert:
    - residual depth for better capacity
    - context-conditioned affine modulation (FiLM-like)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_context_residual_gate: bool = False,
        context_residual_init: float = 0.1,
        use_context_concat: bool = False,
    ):
        super().__init__()
        self.use_context_residual_gate = bool(use_context_residual_gate)
        self.use_context_concat = bool(use_context_concat)
        if self.use_context_concat:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.context_affine = None
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.context_affine = nn.Linear(input_dim, hidden_dim * 2)
        if self.use_context_residual_gate and not self.use_context_concat:
            init = float(min(max(context_residual_init, 1e-4), 1.0 - 1e-4))
            self.context_residual_logit = nn.Parameter(torch.logit(torch.tensor(init, dtype=torch.float32)))
        else:
            self.register_parameter("context_residual_logit", None)
        self.blocks = nn.ModuleList(
            [NamespaceResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim * 2, dropout=dropout) for _ in range(num_layers)]
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: Tensor, context: Optional[Tensor] = None, use_context_modulation: bool = True) -> Tensor:
        if context is None:
            context = x
        if self.use_context_concat:
            h = self.input_proj(torch.cat([x, context], dim=-1))
        else:
            h = self.input_proj(x)
        if use_context_modulation and not self.use_context_concat:
            if self.context_affine is None:
                raise RuntimeError("Context affine layer is unavailable for modulation.")
            gamma, beta = self.context_affine(context).chunk(2, dim=-1)
            if self.use_context_residual_gate:
                context_delta = h * torch.tanh(gamma) + beta
                context_scale = torch.sigmoid(self.context_residual_logit).to(dtype=h.dtype, device=h.device)
                h = h + context_scale * context_delta
            else:
                h = h * (1.0 + torch.tanh(gamma)) + beta
        for block in self.blocks:
            h = block(h)
        return self.out(h)

#产生softmax权重
class PLEVectorGate(nn.Module):
    """
    PLE-style gate for vector experts.
    """

    def __init__(
        self,
        input_dim: int,
        num_inputs: int,
        gate_input_dim: Optional[int] = None,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.gate_input_dim = gate_input_dim or input_dim
        self.gate_net = nn.Sequential(
            nn.Linear(self.gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_inputs),
        )
        self.global_bias = nn.Parameter(torch.zeros(num_inputs))

    def forward(self, expert_vectors: List[Tensor], gate_input: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        if len(expert_vectors) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} vectors, got {len(expert_vectors)}")

        if gate_input is None:
            gate_input = torch.mean(torch.stack(expert_vectors, dim=1), dim=1)

        logits = self.gate_net(gate_input) + self.global_bias.unsqueeze(0)
        weights = torch.softmax(logits, dim=-1)
        stacked = torch.stack(expert_vectors, dim=1)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        return fused, weights

#GO 和蛋白之间的 cross-attention
class NamespaceContextEncoder(nn.Module):
    """
    Encode one namespace subgraph embedding table into a protein-conditioned context vector.
    """

    def __init__(self, protein_dim: int, hidden_dim: int = 256, dropout: float = 0.1, top_k: int = 0):
        super().__init__()
        self.query_proj = nn.Linear(protein_dim, hidden_dim)
        self.key_proj = nn.LazyLinear(hidden_dim)
        self.value_proj = nn.LazyLinear(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, protein_dim)
        self.dropout = nn.Dropout(dropout)
        self.top_k = int(top_k)

    def forward(self, protein_shared_repr: Tensor, go_embeddings: Tensor, top_k: Optional[int] = None) -> Tensor:
        # protein_shared_repr: [B, Dp], go_embeddings: [N, Dg]
        q = self.query_proj(protein_shared_repr)  # [B, H]
        k = self.key_proj(go_embeddings)  # [N, H]
        v = self.value_proj(go_embeddings)  # [N, H]
        attn_scores = torch.matmul(q, k.t()) / math.sqrt(k.size(-1))  # [B, N]

        # Top-k attention reduces noise from very large GO candidate spaces.
        k_limit = self.top_k if top_k is None else int(top_k)
        if k_limit > 0 and attn_scores.size(-1) > k_limit:
            top_scores, top_idx = torch.topk(attn_scores, k=k_limit, dim=-1)
            top_weights = torch.softmax(top_scores, dim=-1)
            top_weights = self.dropout(top_weights)
            top_values = v[top_idx]  # [B, K, H]
            context = (top_weights.unsqueeze(-1) * top_values).sum(dim=1)  # [B, H]
        else:
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            context = torch.matmul(attn_weights, v)  # [B, H]
        return self.out_proj(context)  # [B, Dp]


class CrossNamespaceSemanticRefinementBlock(nn.Module):
    """
    Namespace-specific second-stage refinement over BP/MF/CC branch representations.

    Inputs are the first-stage namespace protein representations. The block:
    1. keeps the target namespace representation as the skip expert,
    2. projects the other namespace representations into the target branch,
    3. computes a target-namespace GO context and conditioned expert output,
    4. gates [target skip, cross-namespace projections, target GO expert].
    """

    def __init__(
        self,
        namespaces: Tuple[str, ...],
        expert_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        gate_hidden_dim: int = 128,
        context_hidden_dim: int = 256,
        top_k: int = 256,
        dropout: float = 0.1,
        use_context_residual_gate: bool = False,
        context_residual_init: float = 0.1,
        use_context_concat: bool = False,
        refinement_residual_init: float = 0.1,
        detach_cross_namespace_sources: bool = False,
    ):
        super().__init__()
        self.namespaces = tuple(namespaces)
        self.expert_dim = int(expert_dim)
        self.detach_cross_namespace_sources = bool(detach_cross_namespace_sources)

        self.cross_projectors = nn.ModuleDict()
        for target_ns in self.namespaces:
            for source_ns in self.namespaces:
                if source_ns == target_ns:
                    continue
                self.cross_projectors[f"{source_ns}_to_{target_ns}"] = MLPExpert(
                    input_dim=self.expert_dim,
                    output_dim=self.expert_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
        self.namespace_context_encoders = nn.ModuleDict(
            {
                ns: NamespaceContextEncoder(
                    protein_dim=self.expert_dim,
                    hidden_dim=context_hidden_dim,
                    dropout=dropout,
                    top_k=top_k,
                )
                for ns in self.namespaces
            }
        )
        self.namespace_experts = nn.ModuleDict(
            {
                ns: NamespaceConditionedExpert(
                    input_dim=self.expert_dim,
                    output_dim=self.expert_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    use_context_residual_gate=use_context_residual_gate,
                    context_residual_init=context_residual_init,
                    use_context_concat=use_context_concat,
                )
                for ns in self.namespaces
            }
        )
        num_gate_inputs = len(self.namespaces) + 1
        self.final_gates = nn.ModuleDict(
            {
                ns: PLEVectorGate(
                    input_dim=self.expert_dim,
                    num_inputs=num_gate_inputs,
                    gate_input_dim=self.expert_dim * num_gate_inputs,
                    hidden_dim=gate_hidden_dim,
                    dropout=dropout,
                )
                for ns in self.namespaces
            }
        )
        init = float(min(max(refinement_residual_init, 1e-4), 1.0 - 1e-4))
        self.refinement_residual_logits = nn.ParameterDict(
            {
                ns: nn.Parameter(torch.logit(torch.tensor(init, dtype=torch.float32)))
                for ns in self.namespaces
            }
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        legacy_key = f"{prefix}refinement_residual_logit"
        if legacy_key in state_dict:
            legacy_logit = state_dict.pop(legacy_key)
            for ns in self.namespaces:
                state_dict.setdefault(f"{prefix}refinement_residual_logits.{ns}", legacy_logit.clone())
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        branch_outputs: Dict[str, Tensor],
        go_embeddings_by_namespace: Dict[str, Tensor],
        ablate_go_context: bool = False,
        ablate_go_modulation: bool = False,
    ) -> tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor], Tensor]:
        missing = [ns for ns in self.namespaces if ns not in branch_outputs]
        if missing:
            raise RuntimeError(f"Missing branch outputs for cross-namespace refinement: {missing}")

        refined: Dict[str, Tensor] = {}
        namespace_contexts: Dict[str, Tensor] = {}
        gate_weights: Dict[str, Tensor] = {}
        first_branch = next(iter(branch_outputs.values()))
        namespace_expert_outputs: List[Tensor] = []
        for ns in self.namespaces:
            skip = branch_outputs[ns]
            residual_scale = torch.sigmoid(self.refinement_residual_logits[ns]).to(
                device=first_branch.device,
                dtype=first_branch.dtype,
            )
            if ablate_go_context or ns not in go_embeddings_by_namespace:
                go_context = torch.zeros_like(skip)
            else:
                go_context = self.namespace_context_encoders[ns](skip, go_embeddings_by_namespace[ns])
            namespace_contexts[ns] = go_context

            ns_go_out = self.namespace_experts[ns](
                skip,
                context=go_context,
                use_context_modulation=not ablate_go_modulation,
            )
            namespace_expert_outputs.append(ns_go_out)

            expert_vectors = [skip]
            for source_ns in self.namespaces:
                if source_ns == ns:
                    continue
                source_repr = branch_outputs[source_ns]
                if self.detach_cross_namespace_sources:
                    source_repr = source_repr.detach()
                expert_vectors.append(self.cross_projectors[f"{source_ns}_to_{ns}"](source_repr))
            expert_vectors.append(ns_go_out)
            gate_input = torch.cat(expert_vectors, dim=-1)
            fused, weights = self.final_gates[ns](
                expert_vectors,
                gate_input=gate_input,
            )
            refined[ns] = skip + residual_scale * (fused - skip)
            gate_weights[ns] = weights

        summary_out = torch.stack(namespace_expert_outputs, dim=0).mean(dim=0)
        return refined, namespace_contexts, gate_weights, summary_out

def _build_parent_first_topo_order(num_nodes: int, edge_index: Tensor) -> List[int]:
    """
    Build topological order where ancestors (parents) appear before descendants.
    Input edge direction is child -> parent.
    """
    indegree = [0] * num_nodes
    children_by_parent: List[List[int]] = [[] for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        child = int(edge_index[0, i].item())
        parent = int(edge_index[1, i].item())
        if not (0 <= child < num_nodes and 0 <= parent < num_nodes):
            continue
        children_by_parent[parent].append(child)
        indegree[child] += 1

    queue = [i for i in range(num_nodes) if indegree[i] == 0]
    topo: List[int] = []
    while queue:
        node = queue.pop(0)
        topo.append(node)
        for child in children_by_parent[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    if len(topo) != num_nodes:
        # Fallback for unexpected cycles/noise.
        return list(range(num_nodes))
    return topo

#预测头
class GOSimilarityPredictor(nn.Module):
    """
    Branch prediction by similarity to namespace GO node embeddings.
    """

    def __init__(
        self,
        query_dim: int,
        dropout: float = 0.1,
        use_term_bias: bool = True,
    ):
        super().__init__()
        self.go_proj = nn.LazyLinear(query_dim)
        self.query_norm = nn.LayerNorm(query_dim)
        self.go_norm = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_term_bias = bool(use_term_bias)

        # Log-space temperature for stable, strictly-positive scaling:
        # scale = exp(clamp(logit_scale, min, max)).
        self.logit_scale = nn.Parameter(torch.tensor(0.0))
        self.logit_scale_min = -5.0
        self.logit_scale_max = 5.0
        # Checkpoint marker:
        # - present => logit_scale is already log-space (new format)
        # - absent  => treat checkpoint logit_scale as legacy direct-scale and convert by log().
        self.register_buffer("logit_scale_is_log", torch.tensor(1, dtype=torch.uint8))
        self.term_bias = nn.Parameter(torch.zeros(0), requires_grad=self.use_term_bias)

    def set_term_bias_from_prior(self, prior_logit: Tensor, trainable: bool = True) -> None:
        prior = prior_logit.detach().float().reshape(-1)
        self.term_bias = nn.Parameter(prior, requires_grad=bool(trainable))

    def _ensure_term_bias_shape(self, num_terms: int, device: torch.device) -> None:
        if not self.use_term_bias:
            return
        if self.term_bias.numel() == num_terms:
            if self.term_bias.device != device:
                self.term_bias.data = self.term_bias.data.to(device)
            return
        self.term_bias = nn.Parameter(torch.zeros(num_terms, device=device), requires_grad=True)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        scale_key = prefix + "logit_scale"
        marker_key = prefix + "logit_scale_is_log"
        term_bias_key = prefix + "term_bias"
        if term_bias_key in state_dict and torch.is_tensor(state_dict[term_bias_key]):
            loaded_bias = state_dict[term_bias_key]
            if self.term_bias.numel() != loaded_bias.numel():
                self.term_bias = nn.Parameter(torch.zeros_like(loaded_bias), requires_grad=loaded_bias.requires_grad)
        if scale_key in state_dict and marker_key not in state_dict:
            legacy_scale = state_dict[scale_key]
            if torch.is_tensor(legacy_scale):
                state_dict[scale_key] = legacy_scale.clamp(min=1e-6).log()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if term_bias_key in missing_keys and self.term_bias.numel() == 0:
            missing_keys.remove(term_bias_key)

    def forward(self, branch_repr: Tensor, go_embeddings: Tensor) -> Tensor:
        q = self.query_norm(self.dropout(branch_repr))  # [B, D]
        go = self.go_norm(self.go_proj(go_embeddings))  # [N, D]
        scale = torch.exp(self.logit_scale.clamp(min=self.logit_scale_min, max=self.logit_scale_max))
        logits = torch.matmul(q, go.t()) * scale
        if self.use_term_bias:
            self._ensure_term_bias_shape(go_embeddings.size(0), device=logits.device)
            logits = logits + self.term_bias.unsqueeze(0)
        return logits


class DirectMLPPredictor(nn.Module):
    """
    Branch prediction head without GO semantic inputs.

    This head is used for strong GO-semantics ablations: it maps the branch
    representation directly to namespace logits and ignores GO embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        num_terms: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_term_bias: bool = True,
    ):
        super().__init__()
        self.use_term_bias = bool(use_term_bias)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_terms),
        )
        self.term_bias = nn.Parameter(torch.zeros(num_terms), requires_grad=self.use_term_bias)

    def set_term_bias_from_prior(self, prior_logit: Tensor, trainable: bool = True) -> None:
        prior = prior_logit.detach().float().reshape(-1)
        self.term_bias = nn.Parameter(prior, requires_grad=bool(trainable) and self.use_term_bias)

    def forward(self, branch_repr: Tensor, go_embeddings: Tensor | None = None) -> Tensor:
        logits = self.net(branch_repr)
        if self.use_term_bias:
            logits = logits + self.term_bias.to(device=logits.device, dtype=logits.dtype).unsqueeze(0)
        return logits

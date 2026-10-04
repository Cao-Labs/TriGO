from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from trigo.go_term import GOEncodingBundle

from .feature_cache import feature_file_path, load_cached_feature
from .go_utils import ProteinAnnotationRecord, annotation_to_label_indices, build_go_term_index


GO_ROOT_BY_NAMESPACE = {
    "bp": "GO:0008150",
    "mf": "GO:0003674",
    "cc": "GO:0005575",
}


@dataclass
class ProteinGODatasetConfig:
    contact_distance_threshold: float = 8.0
    skip_without_label: bool = True
    max_residues: int = 1022
    propagate_labels_to_ancestors: bool = False
    drop_namespace_roots_from_labels: bool = True


def random_split_records(
    records: Sequence[ProteinAnnotationRecord],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[List[ProteinAnnotationRecord], List[ProteinAnnotationRecord], List[ProteinAnnotationRecord]]:
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    records = list(records)
    rng = random.Random(seed)
    rng.shuffle(records)

    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_records = records[:n_train]
    val_records = records[n_train : n_train + n_val]
    test_records = records[n_train + n_val :]
    return train_records, val_records, test_records


class ProteinGODataset(Dataset):
    def __init__(
        self,
        records: Sequence[ProteinAnnotationRecord],
        cache_dir: str,
        go_bundle: GOEncodingBundle,
        cfg: ProteinGODatasetConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg or ProteinGODatasetConfig()
        self.cache_dir = Path(cache_dir)
        self._slim_esm = None
        self._slim_coordinates = None
        self.slim_metadata: Mapping[str, object] | None = None
        slim_metadata_path = self.cache_dir / "metadata.pt"
        if (
            slim_metadata_path.exists()
            and (self.cache_dir / "esm.f16.bin").exists()
            and (self.cache_dir / "coordinates.f32.bin").exists()
        ):
            self.slim_metadata = torch.load(slim_metadata_path, map_location="cpu", weights_only=True)
        slim_entries = self.slim_metadata.get("entries", {}) if self.slim_metadata is not None else {}
        self.go_bundle = go_bundle
        self.namespace_term_ids = {
            ns: go_bundle.encodings[ns].term_ids for ns in ("bp", "mf", "cc") if ns in go_bundle.encodings
        }
        self.go_index = build_go_term_index(self.namespace_term_ids)
        self.root_index_by_ns = {}
        self.parents_by_ns = {}
        self.ancestor_cache_by_ns = {}
        if self.cfg.propagate_labels_to_ancestors:
            for ns in ("bp", "mf", "cc"):
                if ns not in go_bundle.encodings:
                    continue
                ns_enc = go_bundle.encodings[ns]
                root_go = GO_ROOT_BY_NAMESPACE.get(ns)
                root_idx = None
                if root_go is not None:
                    try:
                        root_idx = ns_enc.term_ids.index(root_go)
                    except ValueError:
                        root_idx = None
                self.root_index_by_ns[ns] = root_idx
                self.parents_by_ns[ns] = self._get_or_build_parent_adjacency(ns)
                self.ancestor_cache_by_ns[ns] = {}

        self.samples: List[Dict[str, object]] = []
        for record in records:
            feat_path = feature_file_path(self.cache_dir, record.entry)
            has_slim_feature = record.entry in slim_entries
            if not feat_path.exists() and not has_slim_feature:
                continue
            label_indices = annotation_to_label_indices(record, self.go_index)
            if self.cfg.propagate_labels_to_ancestors:
                label_indices = self._expand_label_indices_to_ancestors(label_indices)
            has_label = any(len(label_indices[ns]) > 0 for ns in ("bp", "mf", "cc"))
            if self.cfg.skip_without_label and not has_label:
                continue
            self.samples.append(
                {
                    "entry": record.entry,
                    "feature_path": str(feat_path),
                    "has_slim_feature": has_slim_feature,
                    "label_indices": label_indices,
                }
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_slim_esm"] = None
        state["_slim_coordinates"] = None
        return state

    def _load_slim_feature(self, entry: str) -> Dict[str, Tensor]:
        if self.slim_metadata is None:
            raise RuntimeError("Slim feature metadata is not available.")
        entries = self.slim_metadata["entries"]
        offset, stored_length = entries[entry]
        sequence_dim = int(self.slim_metadata["sequence_dim"])
        length = int(stored_length)
        if self.cfg.max_residues > 0:
            length = min(length, self.cfg.max_residues)
        if self._slim_esm is None:
            self._slim_esm = np.memmap(
                self.cache_dir / "esm.f16.bin", mode="r", dtype=np.float16
            ).reshape(-1, sequence_dim)
            self._slim_coordinates = np.memmap(
                self.cache_dir / "coordinates.f32.bin", mode="r", dtype=np.float32
            ).reshape(-1, 3)
        offset = int(offset)
        esm = np.array(self._slim_esm[offset : offset + length], copy=True)
        coordinates = np.array(self._slim_coordinates[offset : offset + length], copy=True)
        return {
            "esm_embedding": torch.from_numpy(esm),
            "ca_coords": torch.from_numpy(coordinates),
        }

    def _get_or_build_parent_adjacency(self, namespace: str) -> List[List[int]]:
        ns_enc = self.go_bundle.encodings[namespace]
        cached = getattr(ns_enc, "_parents_adjacency_cache", None)
        if cached is not None and len(cached) == len(ns_enc.term_ids):
            return cached

        num_terms = len(ns_enc.term_ids)
        parents: List[List[int]] = [[] for _ in range(num_terms)]
        edge_index = ns_enc.edge_index.detach().cpu()
        if edge_index.numel() > 0:
            for child, parent in edge_index.t().tolist():
                child = int(child)
                parent = int(parent)
                if 0 <= child < num_terms and 0 <= parent < num_terms:
                    parents[child].append(parent)
        setattr(ns_enc, "_parents_adjacency_cache", parents)
        return parents

    def _expand_single_index_to_ancestors(self, namespace: str, index: int) -> List[int]:
        cache = self.ancestor_cache_by_ns[namespace]
        if index in cache:
            return cache[index]

        parents = self.parents_by_ns[namespace]
        ancestors = {index}
        queue = deque([index])
        while queue:
            node = queue.popleft()
            for parent in parents[node]:
                if parent in ancestors:
                    continue
                ancestors.add(parent)
                queue.append(parent)

        expanded = sorted(ancestors)
        cache[index] = expanded
        return expanded

    def _expand_label_indices_to_ancestors(self, label_indices: Dict[str, List[int]]) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for ns in ("bp", "mf", "cc"):
            idxs = label_indices.get(ns, [])
            if not idxs or ns not in self.parents_by_ns:
                out[ns] = sorted(set(idxs))
                continue

            expanded = set()
            for idx in idxs:
                expanded.update(self._expand_single_index_to_ancestors(ns, int(idx)))

            if self.cfg.drop_namespace_roots_from_labels:
                root_idx = self.root_index_by_ns.get(ns)
                if root_idx is not None:
                    expanded.discard(root_idx)
            out[ns] = sorted(expanded)
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        item = self.samples[index]
        if item["has_slim_feature"]:
            feat = self._load_slim_feature(str(item["entry"]))
        else:
            feat = load_cached_feature(item["feature_path"])

        sequence_embeddings: Tensor = torch.nan_to_num(feat["esm_embedding"].float())
        ca_coords: Tensor = torch.nan_to_num(feat["ca_coords"].float())

        lengths = [sequence_embeddings.size(0), ca_coords.size(0)]
        min_len = min(lengths)
        if sequence_embeddings.size(0) != min_len or ca_coords.size(0) != min_len:
            sequence_embeddings = sequence_embeddings[:min_len]
            ca_coords = ca_coords[:min_len]

        if self.cfg.max_residues > 0 and sequence_embeddings.size(0) > self.cfg.max_residues:
            sequence_embeddings = sequence_embeddings[: self.cfg.max_residues]
            ca_coords = ca_coords[: self.cfg.max_residues]

        dist = torch.cdist(ca_coords, ca_coords, p=2.0).float()
        edge_mask = dist <= float(self.cfg.contact_distance_threshold)
        contact_map = torch.where(edge_mask, dist, torch.full_like(dist, float("inf")))
        contact_map.fill_diagonal_(float("inf"))
        residue_mask = torch.ones(sequence_embeddings.size(0), dtype=torch.bool)

        return {
            "entry": item["entry"],
            "sequence_embeddings": sequence_embeddings,
            "contact_map": contact_map,
            "residue_mask": residue_mask,
            "label_indices": item["label_indices"],
        }


class ProteinGOBatchCollator:
    def __init__(self, go_bundle: GOEncodingBundle):
        self.num_terms = {ns: len(go_bundle.encodings[ns].term_ids) for ns in ("bp", "mf", "cc")}

    def __call__(self, batch: List[Dict[str, object]]) -> Dict[str, object]:
        batch_size = len(batch)
        max_len = max(item["sequence_embeddings"].size(0) for item in batch)
        emb_dim = batch[0]["sequence_embeddings"].size(1)

        seq_batch = torch.zeros((batch_size, max_len, emb_dim), dtype=torch.float32)
        contact_batch = torch.zeros((batch_size, max_len, max_len), dtype=torch.float32)
        mask_batch = torch.zeros((batch_size, max_len), dtype=torch.bool)

        labels = {
            ns: torch.zeros((batch_size, self.num_terms[ns]), dtype=torch.float32)
            for ns in ("bp", "mf", "cc")
        }
        entries = []

        for i, item in enumerate(batch):
            seq = item["sequence_embeddings"]
            contact = item["contact_map"]
            residue_mask = item["residue_mask"]
            L = seq.size(0)
            seq_batch[i, :L] = seq
            contact_batch[i, :L, :L] = contact
            mask_batch[i, :L] = residue_mask

            label_indices = item["label_indices"]
            for ns in ("bp", "mf", "cc"):
                idxs = label_indices.get(ns, [])
                if idxs:
                    labels[ns][i, idxs] = 1.0
            entries.append(item["entry"])

        return {
            "entries": entries,
            "sequence_embeddings": seq_batch,
            "contact_map": contact_batch,
            "residue_mask": mask_batch,
            "labels": labels,
        }

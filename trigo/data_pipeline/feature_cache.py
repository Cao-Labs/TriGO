from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch import Tensor

from .esm_utils import ESMEncoderConfig, ESMSequenceEncoder
from .go_utils import ProteinAnnotationRecord
from .pdb_utils import parse_pdb_ca, truncate_residue_level_features

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable


@dataclass
class FeatureBuildConfig:
    max_residues: int = 1022
    chain_id: str | None = "A"
    contact_distance_threshold: float = 8.0
    save_fp16_embedding: bool = True
    force_recompute: bool = False
    show_progress: bool = True


def feature_file_path(cache_dir: str | Path, entry: str) -> Path:
    return Path(cache_dir) / f"{entry}.pt"


def load_cached_feature(path: str | Path) -> Dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


class ProteinFeatureCacheBuilder:
    """
    Build and cache per-protein features:
    - aligned sequence from PDB C-alpha residues
    - ESM residue embeddings
    - C-alpha coordinates
    - thresholded contact map (store distance for edges, +inf otherwise)
    """

    def __init__(self, esm_cfg: ESMEncoderConfig, build_cfg: FeatureBuildConfig):
        self.esm_encoder = ESMSequenceEncoder(esm_cfg)
        self.cfg = build_cfg

    def _build_one(self, record: ProteinAnnotationRecord) -> Dict[str, object]:
        pdb_seq, ca_coords, residue_keys = parse_pdb_ca(record.pdb_path, chain_id=self.cfg.chain_id)
        pdb_seq, ca_coords = truncate_residue_level_features(
            sequence=pdb_seq,
            ca_coords=ca_coords,
            max_residues=self.cfg.max_residues,
        )

        esm_embedding = self.esm_encoder.encode(pdb_seq)
        if esm_embedding.size(0) != ca_coords.size(0):
            min_len = min(esm_embedding.size(0), ca_coords.size(0))
            esm_embedding = esm_embedding[:min_len]
            ca_coords = ca_coords[:min_len]
            pdb_seq = pdb_seq[:min_len]

        if self.cfg.save_fp16_embedding:
            esm_embedding = esm_embedding.half()

        dist = torch.cdist(ca_coords, ca_coords, p=2.0).float()
        edge_mask = dist <= float(self.cfg.contact_distance_threshold)
        contact_map = torch.where(edge_mask, dist, torch.full_like(dist, float("inf")))
        contact_map.fill_diagonal_(float("inf"))

        return {
            "entry": record.entry,
            "sequence": pdb_seq,
            "esm_embedding": esm_embedding.cpu(),
            "ca_coords": ca_coords.cpu(),
            "contact_map": contact_map.cpu(),
            "residue_keys": residue_keys[: len(pdb_seq)],
            "source_pdb_path": record.pdb_path,
        }

    def build_and_cache(
        self,
        records: List[ProteinAnnotationRecord],
        cache_dir: str | Path,
    ) -> Dict[str, str]:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        status = {}

        record_iter = records
        if self.cfg.show_progress:
            record_iter = tqdm(records, desc="protein features", unit="protein")

        for record in record_iter:
            out_path = feature_file_path(cache_dir, record.entry)
            if out_path.exists() and not self.cfg.force_recompute:
                status[record.entry] = "cached"
                continue

            try:
                item = self._build_one(record)
                torch.save(item, out_path)
                status[record.entry] = "built"
            except Exception as exc:  # noqa: BLE001
                status[record.entry] = f"failed: {exc}"
        return status

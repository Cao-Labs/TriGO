from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trigo.data_pipeline import (
    ESMEncoderConfig,
    FeatureBuildConfig,
    ProteinFeatureCacheBuilder,
    load_annotation_records,
    load_go_namespace_and_name,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Prepare protein feature cache from PDB files and ESM embeddings")
    p.add_argument("--xlsx-path", type=str, required=True)
    p.add_argument("--pdb-dir", type=str, required=True)
    p.add_argument("--obo-path", type=str, required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument(
        "--esm-model-path",
        type=str,
        default=str(PROJECT_ROOT / "pretrained" / "esm2_t33_650M_UR50D.pt"),
        help="fair-esm .pt checkpoint or Hugging Face ESMC model directory.",
    )
    p.add_argument("--esm-layer", type=int, default=33)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-residues", type=int, default=1022)
    p.add_argument("--contact-threshold", type=float, default=8.0)
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--chain-id", type=str, default="", help="PDB chain ID. Empty uses all chains.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    go_to_namespace, _ = load_go_namespace_and_name(args.obo_path)
    records = load_annotation_records(args.xlsx_path, args.pdb_dir, go_to_namespace)
    if args.max_records > 0:
        records = records[: args.max_records]
    print(f"records with pdb: {len(records)}")

    esm_cfg = ESMEncoderConfig(
        model_path=args.esm_model_path,
        repr_layer=args.esm_layer,
        device=args.device,
        chunk_size=args.max_residues,
    )
    build_cfg = FeatureBuildConfig(
        max_residues=args.max_residues,
        chain_id=args.chain_id or None,
        contact_distance_threshold=args.contact_threshold,
        force_recompute=args.force,
        show_progress=not args.no_progress,
    )
    builder = ProteinFeatureCacheBuilder(esm_cfg, build_cfg)
    status = builder.build_and_cache(records, cache_dir=args.cache_dir)

    counts = Counter()
    for value in status.values():
        if value.startswith("failed"):
            counts["failed"] += 1
        else:
            counts[value] += 1
    print("summary:", dict(counts))


if __name__ == "__main__":
    main()

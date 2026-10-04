from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trigo.data_pipeline import (  # noqa: E402
    ESMEncoderConfig,
    FeatureBuildConfig,
    ProteinAnnotationRecord,
    ProteinFeatureCacheBuilder,
    load_annotation_records,
    load_go_namespace_and_name,
)
from trigo.data_pipeline.feature_cache import load_cached_feature  # noqa: E402
from trigo.go_term import GOEncodingBundle  # noqa: E402
from trigo.multi_branch import MultiBranchProteinGOConfig, MultiBranchProteinGOModel  # noqa: E402
from trigo.protein_ple import ProteinPLEConfig  # noqa: E402
from train_model import (  # noqa: E402
    collect_go_terms_from_records,
    move_go_bundle_to_device,
    prune_go_bundle_to_terms,
)

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001

    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable


NAMESPACES = ("bp", "mf", "cc")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Predict GO functions for new proteins from a trained TriGO checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt.")
    p.add_argument(
        "--run-config",
        type=str,
        default="",
        help="Training run_config.json. Defaults to checkpoint directory/run_config.json when present.",
    )
    p.add_argument("--output-dir", type=str, default="outputs/predictions")

    # Files needed to reconstruct the trained GO prediction space.
    p.add_argument("--go-embedding-path", type=str, default="", help="Defaults to run_config.json value.")
    p.add_argument("--obo-path", type=str, default="", help="Defaults to run_config.json value.")
    p.add_argument(
        "--xlsx-path",
        type=str,
        default="",
        help="Training annotation workbook. Required for filtered GO space when run_config.json is unavailable.",
    )
    p.add_argument(
        "--pdb-dir",
        type=str,
        default="",
        help="Training PDB directory used to reproduce filtered GO term selection.",
    )
    p.add_argument("--go-space", type=str, default="auto", choices=("auto", "filtered", "full"))

    # Prediction inputs. Cached features are fastest; PDB inputs can be converted on demand.
    p.add_argument(
        "--feature-cache-dir",
        type=str,
        default="",
        help="Directory of existing {Entry}.pt protein feature files. Defaults to run_config.json value.",
    )
    p.add_argument("--feature-path", action="append", default=[], help="Specific cached .pt feature file.")
    p.add_argument(
        "--feature-list",
        type=str,
        default="",
        help="Text file with one cached .pt path or protein entry per line.",
    )
    p.add_argument("--pdb-path", action="append", default=[], help="New protein PDB file. Can be repeated.")
    p.add_argument(
        "--entry",
        action="append",
        default=[],
        help="Protein ID for each --pdb-path. Defaults to the PDB filename stem.",
    )
    p.add_argument(
        "--prediction-cache-dir",
        type=str,
        default="",
        help="Where features built from --pdb-path are cached. Defaults to output_dir/protein_feature_cache.",
    )
    p.add_argument(
        "--esm-model-path",
        type=str,
        default=str(PROJECT_ROOT / "pretrained" / "esm2_t33_650M_UR50D.pt"),
        help="fair-esm .pt checkpoint or Hugging Face ESMC model directory. Required to build features from PDB.",
    )
    p.add_argument("--esm-layer", type=int, default=33)
    p.add_argument("--chain-id", type=str, default="", help="PDB chain ID for new PDB inputs. Empty uses all chains.")
    p.add_argument("--force-feature-recompute", action="store_true")

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-residues", type=int, default=0, help="0 uses checkpoint/run_config default.")
    p.add_argument("--contact-threshold", type=float, default=0.0, help="0 uses run_config default.")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-prob", type=float, default=0.0)
    p.add_argument("--namespaces", type=str, default="bp,mf,cc")
    p.add_argument("--save-full-matrix", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def resolve_path(raw: str | None, fallback: str | None = None) -> Path | None:
    value = raw or fallback
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    repo_path = PROJECT_ROOT / path
    if repo_path.exists():
        return repo_path
    return path


def load_run_config(checkpoint_path: Path, run_config_arg: str) -> dict:
    config_path = resolve_path(run_config_arg)
    if config_path is None:
        candidate = checkpoint_path.parent / "run_config.json"
        config_path = candidate if candidate.exists() else None
    if config_path is None or not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["_path"] = str(config_path)
    return payload


def config_arg(run_config: dict, name: str, default=None):
    return run_config.get("args", {}).get(name, default)


def model_term_counts(checkpoint: dict) -> dict[str, int]:
    cfg = checkpoint.get("model_cfg", {})
    return {
        "bp": int(cfg.get("bp_num_terms", 0)),
        "mf": int(cfg.get("mf_num_terms", 0)),
        "cc": int(cfg.get("cc_num_terms", 0)),
    }


def bundle_term_counts(bundle: GOEncodingBundle) -> dict[str, int]:
    return {ns: len(bundle.encodings[ns].term_ids) for ns in NAMESPACES}


def counts_match(bundle: GOEncodingBundle, expected: dict[str, int]) -> bool:
    actual = bundle_term_counts(bundle)
    return all(actual.get(ns) == expected.get(ns) for ns in NAMESPACES)


def load_prediction_go_bundle(
    *,
    checkpoint: dict,
    run_config: dict,
    args: argparse.Namespace,
    go_embedding_path: Path,
    go_to_namespace: dict[str, str],
) -> tuple[GOEncodingBundle, str]:
    full_bundle = GOEncodingBundle.load(str(go_embedding_path), map_location="cpu")
    expected_counts = model_term_counts(checkpoint)
    requested = args.go_space
    if requested == "auto":
        requested = config_arg(run_config, "go_space", "auto") or "auto"

    if requested == "full" and counts_match(full_bundle, expected_counts):
        return full_bundle, "full"

    xlsx_path = resolve_path(args.xlsx_path, config_arg(run_config, "xlsx_path", None))
    pdb_dir = resolve_path(args.pdb_dir, config_arg(run_config, "pdb_dir", None))
    max_records = int(config_arg(run_config, "max_records", 0) or 0)

    def build_filtered() -> GOEncodingBundle:
        if xlsx_path is None or not xlsx_path.exists():
            raise RuntimeError(
                "Filtered GO space requires the original training --xlsx-path, "
                "or run_config.json next to the checkpoint."
            )
        if pdb_dir is None:
            raise RuntimeError(
                "Filtered GO space requires the original training --pdb-dir, "
                "or run_config.json next to the checkpoint."
            )
        records = load_annotation_records(str(xlsx_path), str(pdb_dir), go_to_namespace)
        if max_records > 0:
            records = records[:max_records]
        kept_terms = collect_go_terms_from_records(records)
        return prune_go_bundle_to_terms(full_bundle, kept_terms)

    if requested == "filtered":
        filtered_bundle = build_filtered()
        if not counts_match(filtered_bundle, expected_counts):
            raise RuntimeError(
                "Filtered GO term counts do not match checkpoint. "
                f"checkpoint={expected_counts}, filtered_bundle={bundle_term_counts(filtered_bundle)}"
            )
        return filtered_bundle, "filtered"

    if counts_match(full_bundle, expected_counts):
        return full_bundle, "full"

    filtered_bundle = build_filtered()
    if counts_match(filtered_bundle, expected_counts):
        return filtered_bundle, "filtered"

    raise RuntimeError(
        "Could not reconstruct the checkpoint GO space. "
        f"checkpoint={expected_counts}, full_bundle={bundle_term_counts(full_bundle)}, "
        f"filtered_bundle={bundle_term_counts(filtered_bundle)}"
    )


def read_feature_list(path: Path, feature_cache_dir: Path | None) -> list[Path]:
    paths: list[Path] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            item_path = Path(item).expanduser()
            if item_path.suffix == ".pt" or item_path.exists():
                paths.append(item_path if item_path.is_absolute() else (PROJECT_ROOT / item_path))
            else:
                if feature_cache_dir is None:
                    raise RuntimeError(f"Feature-list entry '{item}' is not a path and --feature-cache-dir is unavailable.")
                paths.append(feature_cache_dir / f"{item}.pt")
    return paths


def build_features_from_pdb(args: argparse.Namespace, output_dir: Path) -> list[Path]:
    if not args.pdb_path:
        return []
    if args.entry and len(args.entry) != len(args.pdb_path):
        raise ValueError("--entry must be passed once for each --pdb-path, or omitted.")

    cache_dir = resolve_path(args.prediction_cache_dir)
    if cache_dir is None:
        cache_dir = output_dir / "protein_feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    records: list[ProteinAnnotationRecord] = []
    feature_paths: list[Path] = []
    missing_records: list[ProteinAnnotationRecord] = []
    for i, raw_pdb in enumerate(args.pdb_path):
        pdb_path = resolve_path(raw_pdb)
        if pdb_path is None or not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {raw_pdb}")
        entry = args.entry[i].strip() if args.entry else pdb_path.stem
        record = ProteinAnnotationRecord(
            entry=entry,
            sequence="",
            go_ids=[],
            go_by_namespace={"bp": [], "mf": [], "cc": []},
            pdb_path=str(pdb_path),
            split=None,
        )
        records.append(record)
        feature_path = cache_dir / f"{entry}.pt"
        feature_paths.append(feature_path)
        if args.force_feature_recompute or not feature_path.exists():
            missing_records.append(record)

    if missing_records:
        esm_cfg = ESMEncoderConfig(
            model_path=args.esm_model_path,
            repr_layer=args.esm_layer,
            device=args.device,
            chunk_size=int(args.max_residues or 1022),
        )
        build_cfg = FeatureBuildConfig(
            max_residues=int(args.max_residues or 1022),
            chain_id=args.chain_id or None,
            contact_distance_threshold=float(args.contact_threshold or 8.0),
            force_recompute=args.force_feature_recompute,
            show_progress=not args.no_progress,
        )
        builder = ProteinFeatureCacheBuilder(esm_cfg, build_cfg)
        status = builder.build_and_cache(missing_records, cache_dir=cache_dir)
        failed = {entry: value for entry, value in status.items() if value.startswith("failed")}
        if failed:
            raise RuntimeError(f"Failed to build protein features: {failed}")

    return feature_paths


def collect_feature_paths(args: argparse.Namespace, run_config: dict, output_dir: Path) -> list[Path]:
    feature_cache_dir = resolve_path(args.feature_cache_dir, config_arg(run_config, "feature_cache_dir", None))
    paths: list[Path] = []

    for raw in args.feature_path:
        path = resolve_path(raw)
        if path is None:
            raise FileNotFoundError(f"Feature path not found: {raw}")
        paths.append(path)

    if args.feature_list:
        list_path = resolve_path(args.feature_list)
        if list_path is None or not list_path.exists():
            raise FileNotFoundError(f"Feature list not found: {args.feature_list}")
        paths.extend(read_feature_list(list_path, feature_cache_dir))

    paths.extend(build_features_from_pdb(args, output_dir))

    if not paths and feature_cache_dir is not None and feature_cache_dir.exists():
        paths.extend(sorted(feature_cache_dir.glob("*.pt")))

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not resolved.exists():
            raise FileNotFoundError(f"Feature file not found: {resolved}")
        unique_paths.append(resolved)
        seen.add(resolved)
    if not unique_paths:
        raise RuntimeError("No prediction inputs found. Pass --pdb-path, --feature-path, or --feature-list.")
    return unique_paths


class CachedProteinFeatureDataset(Dataset):
    def __init__(self, feature_paths: Sequence[Path], max_residues: int, contact_threshold: float):
        self.feature_paths = list(feature_paths)
        self.max_residues = int(max_residues)
        self.contact_threshold = float(contact_threshold)

    def __len__(self) -> int:
        return len(self.feature_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.feature_paths[index]
        feat = load_cached_feature(path)
        entry = str(feat.get("entry") or path.stem)
        seq_emb = feat.get("esm_embedding", feat.get("sequence_embeddings", None))
        if not torch.is_tensor(seq_emb):
            raise ValueError(f"{path} does not contain tensor key 'esm_embedding'.")
        sequence_embeddings = seq_emb.float()
        ca_coords = feat.get("ca_coords", None)
        contact_map = feat.get("contact_map", None)

        if torch.is_tensor(ca_coords):
            ca_coords = ca_coords.float()
            min_len = min(sequence_embeddings.size(0), ca_coords.size(0))
            sequence_embeddings = sequence_embeddings[:min_len]
            ca_coords = ca_coords[:min_len]
            if self.max_residues > 0 and sequence_embeddings.size(0) > self.max_residues:
                sequence_embeddings = sequence_embeddings[: self.max_residues]
                ca_coords = ca_coords[: self.max_residues]
            dist = torch.cdist(ca_coords, ca_coords, p=2.0).float()
            edge_mask = dist <= self.contact_threshold
            contact = torch.where(edge_mask, dist, torch.full_like(dist, float("inf")))
            contact.fill_diagonal_(float("inf"))
        elif torch.is_tensor(contact_map):
            contact = contact_map.float()
            min_len = min(sequence_embeddings.size(0), contact.size(0), contact.size(1))
            sequence_embeddings = sequence_embeddings[:min_len]
            contact = contact[:min_len, :min_len]
            if self.max_residues > 0 and sequence_embeddings.size(0) > self.max_residues:
                sequence_embeddings = sequence_embeddings[: self.max_residues]
                contact = contact[: self.max_residues, : self.max_residues]
        else:
            raise ValueError(f"{path} must contain either 'ca_coords' or 'contact_map'.")

        return {
            "entry": entry,
            "feature_path": str(path),
            "sequence": str(feat.get("sequence", "")),
            "sequence_embeddings": sequence_embeddings,
            "contact_map": contact,
            "residue_mask": torch.ones(sequence_embeddings.size(0), dtype=torch.bool),
        }


class CachedFeatureBatchCollator:
    def __call__(self, batch: list[dict[str, object]]) -> dict[str, object]:
        batch_size = len(batch)
        max_len = max(item["sequence_embeddings"].size(0) for item in batch)
        emb_dim = batch[0]["sequence_embeddings"].size(1)

        seq_batch = torch.zeros((batch_size, max_len, emb_dim), dtype=torch.float32)
        contact_batch = torch.zeros((batch_size, max_len, max_len), dtype=torch.float32)
        mask_batch = torch.zeros((batch_size, max_len), dtype=torch.bool)
        entries = []
        feature_paths = []
        lengths = []

        for i, item in enumerate(batch):
            seq = item["sequence_embeddings"]
            contact = item["contact_map"]
            mask = item["residue_mask"]
            length = seq.size(0)
            seq_batch[i, :length] = seq
            contact_batch[i, :length, :length] = contact
            mask_batch[i, :length] = mask
            entries.append(item["entry"])
            feature_paths.append(item["feature_path"])
            lengths.append(length)

        return {
            "entries": entries,
            "feature_paths": feature_paths,
            "lengths": lengths,
            "sequence_embeddings": seq_batch,
            "contact_map": contact_batch,
            "residue_mask": mask_batch,
        }


def parse_namespaces(raw: str) -> list[str]:
    out = []
    for item in raw.split(","):
        ns = item.strip().lower()
        if not ns:
            continue
        if ns not in NAMESPACES:
            raise ValueError(f"Unknown namespace '{ns}'. Expected one of {NAMESPACES}.")
        out.append(ns)
    return out or list(NAMESPACES)


def collect_predictions(
    probs: Tensor,
    term_ids: Sequence[str],
    go_to_name: dict[str, str],
    top_k: int,
    min_prob: float,
) -> list[dict[str, object]]:
    probs_cpu = probs.detach().float().cpu()
    if top_k > 0:
        k = min(int(top_k), probs_cpu.numel())
        values, indices = torch.topk(probs_cpu, k=k)
    else:
        indices = torch.where(probs_cpu >= float(min_prob))[0]
        values = probs_cpu.index_select(0, indices)
        order = torch.argsort(values, descending=True)
        values = values.index_select(0, order)
        indices = indices.index_select(0, order)

    out = []
    rank = 0
    for score, idx in zip(values.tolist(), indices.tolist()):
        if score < min_prob:
            continue
        rank += 1
        go_id = term_ids[int(idx)]
        out.append(
            {
                "rank": rank,
                "go_id": go_id,
                "go_name": go_to_name.get(go_id, ""),
                "probability": float(score),
            }
        )
    return out


def open_matrix_writers(output_dir: Path, go_bundle: GOEncodingBundle, namespaces: Iterable[str]):
    handles = {}
    writers = {}
    for ns in namespaces:
        path = output_dir / f"predictions_matrix_{ns}.csv"
        handle = open(path, "w", encoding="utf-8", newline="")
        writer = csv.writer(handle)
        writer.writerow(["protein_id", *go_bundle.encodings[ns].term_ids])
        handles[ns] = handle
        writers[ns] = writer
    return handles, writers


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.top_k < 0:
        raise ValueError("--top-k must be >= 0.")
    if not (0.0 <= args.min_prob <= 1.0):
        raise ValueError("--min-prob must be in [0, 1].")

    checkpoint_path = resolve_path(args.checkpoint)
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    run_config = load_run_config(checkpoint_path, args.run_config)

    go_embedding_path = resolve_path(args.go_embedding_path, config_arg(run_config, "go_embedding_path", None))
    obo_path = resolve_path(args.obo_path, config_arg(run_config, "obo_path", None))
    if go_embedding_path is None or not go_embedding_path.exists():
        raise FileNotFoundError("GO embedding path is required. Pass --go-embedding-path or keep run_config.json.")
    if obo_path is None or not obo_path.exists():
        raise FileNotFoundError("OBO path is required. Pass --obo-path or keep run_config.json.")

    output_dir = resolve_path(args.output_dir) or Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    go_to_namespace, go_to_name = load_go_namespace_and_name(str(obo_path))
    go_bundle, used_go_space = load_prediction_go_bundle(
        checkpoint=checkpoint,
        run_config=run_config,
        args=args,
        go_embedding_path=go_embedding_path,
        go_to_namespace=go_to_namespace,
    )
    go_bundle = move_go_bundle_to_device(go_bundle, device)

    protein_cfg = ProteinPLEConfig(**checkpoint["protein_cfg"])
    model_cfg = MultiBranchProteinGOConfig(**checkpoint["model_cfg"])
    model = MultiBranchProteinGOModel(model_cfg, protein_encoder_cfg=protein_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    max_residues = int(args.max_residues or checkpoint["protein_cfg"].get("max_residues", 1022))
    contact_threshold = float(args.contact_threshold or config_arg(run_config, "contact_threshold", 8.0) or 8.0)
    feature_paths = collect_feature_paths(args, run_config, output_dir)

    dataset = CachedProteinFeatureDataset(
        feature_paths=feature_paths,
        max_residues=max_residues,
        contact_threshold=contact_threshold,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=CachedFeatureBatchCollator(),
        pin_memory=device.type == "cuda",
    )

    namespaces = [ns for ns in parse_namespaces(args.namespaces) if ns in go_bundle.encodings]
    jsonl_path = output_dir / "predictions.jsonl"
    tsv_path = output_dir / "predictions_topk.tsv"
    matrix_handles = {}
    matrix_writers = {}
    if args.save_full_matrix:
        matrix_handles, matrix_writers = open_matrix_writers(output_dir, go_bundle, namespaces)

    total_rows = 0
    iterator = dataloader
    if not args.no_progress:
        iterator = tqdm(dataloader, desc="predict", unit="batch")

    with open(jsonl_path, "w", encoding="utf-8") as jsonl_f, open(tsv_path, "w", encoding="utf-8", newline="") as tsv_f:
        tsv_writer = csv.writer(tsv_f, delimiter="\t")
        tsv_writer.writerow(["protein_id", "namespace", "rank", "go_id", "go_name", "probability", "feature_path"])

        for batch in iterator:
            outputs = model(
                sequence_embeddings=batch["sequence_embeddings"].to(device, non_blocking=True),
                contact_map=batch["contact_map"].to(device, non_blocking=True),
                go_bundle=go_bundle,
                residue_mask=batch["residue_mask"].to(device, non_blocking=True),
            )
            probs_by_ns = {
                ns: torch.sigmoid(logits).detach().float().cpu()
                for ns, logits in outputs["logits"].items()
            }

            for row_i, protein_id in enumerate(batch["entries"]):
                item = {
                    "protein_id": protein_id,
                    "feature_path": batch["feature_paths"][row_i],
                    "sequence_length": int(batch["lengths"][row_i]),
                    "go_space": used_go_space,
                    "predictions": {},
                }
                for ns in namespaces:
                    if ns not in probs_by_ns:
                        item["predictions"][ns] = []
                        continue
                    probs = probs_by_ns[ns][row_i]
                    term_ids = go_bundle.encodings[ns].term_ids
                    predictions = collect_predictions(
                        probs=probs,
                        term_ids=term_ids,
                        go_to_name=go_to_name,
                        top_k=args.top_k,
                        min_prob=args.min_prob,
                    )
                    item["predictions"][ns] = predictions
                    for pred in predictions:
                        tsv_writer.writerow(
                            [
                                protein_id,
                                ns,
                                pred["rank"],
                                pred["go_id"],
                                pred["go_name"],
                                f"{pred['probability']:.8g}",
                                batch["feature_paths"][row_i],
                            ]
                        )
                        total_rows += 1
                    if args.save_full_matrix:
                        matrix_writers[ns].writerow([protein_id, *[f"{x:.8g}" for x in probs.tolist()]])
                jsonl_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    for handle in matrix_handles.values():
        handle.close()

    summary = {
        "checkpoint": str(checkpoint_path),
        "run_config": run_config.get("_path"),
        "go_embedding_path": str(go_embedding_path),
        "obo_path": str(obo_path),
        "go_space": used_go_space,
        "go_term_counts": bundle_term_counts(go_bundle),
        "feature_count": len(feature_paths),
        "max_residues": max_residues,
        "contact_threshold": contact_threshold,
        "top_k": int(args.top_k),
        "min_prob": float(args.min_prob),
        "jsonl": str(jsonl_path),
        "tsv": str(tsv_path),
        "topk_rows": int(total_rows),
    }
    with open(output_dir / "prediction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"predicted proteins: {len(feature_paths)}")
    print(f"GO space: {used_go_space} {bundle_term_counts(go_bundle)}")
    print(f"saved jsonl: {jsonl_path}")
    print(f"saved tsv: {tsv_path}")
    print(f"saved summary: {output_dir / 'prediction_summary.json'}")


if __name__ == "__main__":
    main()

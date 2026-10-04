from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trigo.data_pipeline import (
    ProteinGOBatchCollator,
    ProteinGODataset,
    ProteinGODatasetConfig,
    load_annotation_records,
    load_go_namespace_and_name,
    random_split_records,
)
from trigo.go_term import GOEncodingBundle
from trigo.multi_branch import MultiBranchProteinGOConfig, MultiBranchProteinGOModel
from trigo.protein_ple import ProteinPLEConfig
from trigo.training import EngineConfig, MultiBranchBCELoss, evaluate, save_eval_pickle
from train_model import (
    collect_go_terms_from_records,
    prune_go_bundle_to_terms,
    replace_go_embeddings_with_one_hot,
    validate_required_annotation_split,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Test multi-branch protein function predictor")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--run-config",
        type=str,
        default="",
        help="Optional training run_config.json. Defaults to checkpoint directory/run_config.json when present.",
    )
    p.add_argument("--go-embedding-path", type=str, default="data/go_node2vec_embeddings.pt")
    p.add_argument("--xlsx-path", type=str, default="data/9606_annotations.xlsx")
    p.add_argument("--pdb-dir", type=str, default="data/9606_pdb")
    p.add_argument("--obo-path", type=str, default="data/go-basic.obo")
    p.add_argument("--feature-cache-dir", type=str, default="data/protein_feature_cache")
    p.add_argument("--require-split", action="store_true")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--max-residues", type=int, default=1022)
    p.add_argument("--contact-threshold", type=float, default=8.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="outputs/test")
    p.add_argument("--metrics-backend", type=str, default="fast", choices=("fast",))
    p.add_argument("--go-space", type=str, default="filtered", choices=("full", "filtered"))
    p.add_argument("--use-one-hot-go-embeddings", action="store_true")
    p.add_argument("--no-eval-propagation", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def move_go_bundle_to_device(bundle: GOEncodingBundle, device: torch.device) -> GOEncodingBundle:
    for _, enc in bundle.encodings.items():
        enc.embeddings = enc.embeddings.to(device)
        enc.edge_index = enc.edge_index.to(device)
        enc.edge_type = enc.edge_type.to(device)
    return bundle


def split_records_for_test(records):
    test_records = [r for r in records if (getattr(r, "split", None) or "").strip().lower() == "test"]
    if len(test_records) > 0:
        return test_records, True
    return [], False


def apply_run_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    run_config_path = Path(args.run_config) if args.run_config else Path(args.checkpoint).resolve().parent / "run_config.json"
    if not run_config_path.exists():
        return args

    with open(run_config_path, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    cfg_args = raw_cfg.get("args", {})
    explicitly_provided = {
        token[2:].split("=", 1)[0].replace("-", "_")
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    for key in (
        "go_embedding_path",
        "xlsx_path",
        "pdb_dir",
        "obo_path",
        "feature_cache_dir",
        "require_split",
        "seed",
        "train_ratio",
        "val_ratio",
        "max_records",
        "max_residues",
        "contact_threshold",
        "go_space",
        "use_one_hot_go_embeddings",
        "metrics_backend",
    ):
        if key not in explicitly_provided and key in cfg_args and cfg_args[key] is not None:
            setattr(args, key, cfg_args[key])
    print(f"loaded test defaults from {run_config_path}")
    return args


def main() -> None:
    args = parse_args()
    args = apply_run_config_defaults(args)
    device = pick_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    go_to_namespace, go_to_name = load_go_namespace_and_name(args.obo_path)

    records = load_annotation_records(args.xlsx_path, args.pdb_dir, go_to_namespace)
    if args.max_records > 0:
        records = records[: args.max_records]
    if args.require_split:
        validate_required_annotation_split(records)

    go_bundle = GOEncodingBundle.load(args.go_embedding_path, map_location="cpu")
    full_sizes = {ns: len(go_bundle.encodings[ns].term_ids) for ns in ("bp", "mf", "cc")}
    if args.go_space == "filtered":
        kept_terms = collect_go_terms_from_records(records)
        go_bundle = prune_go_bundle_to_terms(go_bundle, kept_terms)
        filtered_sizes = {ns: len(go_bundle.encodings[ns].term_ids) for ns in ("bp", "mf", "cc")}
        print(
            "GO space mode=filtered:"
            f" bp {full_sizes['bp']}->{filtered_sizes['bp']},"
            f" mf {full_sizes['mf']}->{filtered_sizes['mf']},"
            f" cc {full_sizes['cc']}->{filtered_sizes['cc']}"
        )
    else:
        print(
            "GO space mode=full:"
            f" bp={full_sizes['bp']}, mf={full_sizes['mf']}, cc={full_sizes['cc']}"
        )
    if args.use_one_hot_go_embeddings:
        go_bundle = replace_go_embeddings_with_one_hot(go_bundle)
        one_hot_sizes = {ns: go_bundle.encodings[ns].embeddings.size(1) for ns in ("bp", "mf", "cc")}
        print(
            "GO embedding mode=one-hot:"
            f" bp_dim={one_hot_sizes['bp']}, mf_dim={one_hot_sizes['mf']}, cc_dim={one_hot_sizes['cc']}"
        )
    go_bundle = move_go_bundle_to_device(go_bundle, device)
    test_records, used_annotation_split = split_records_for_test(records)
    if used_annotation_split:
        print("test split mode: annotation column 'Split'")
    else:
        _, _, test_records = random_split_records(
            records,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        print(
            "test split mode: random fallback "
            f"(train_ratio={args.train_ratio}, val_ratio={args.val_ratio})"
        )

    ds_cfg = ProteinGODatasetConfig(
        contact_distance_threshold=args.contact_threshold,
        max_residues=args.max_residues,
        propagate_labels_to_ancestors=False,
    )
    test_ds = ProteinGODataset(test_records, args.feature_cache_dir, go_bundle, cfg=ds_cfg)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ProteinGOBatchCollator(go_bundle),
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    protein_cfg = ProteinPLEConfig(**checkpoint["protein_cfg"])
    model_cfg = MultiBranchProteinGOConfig(**checkpoint["model_cfg"])
    model = MultiBranchProteinGOModel(model_cfg, protein_encoder_cfg=protein_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = MultiBranchBCELoss()
    engine_cfg = EngineConfig(
        show_progress=not args.no_progress,
        metrics_backend=args.metrics_backend,
        metrics_propagate=not args.no_eval_propagation,
    )
    out = evaluate(
        model=model,
        dataloader=test_loader,
        go_bundle=go_bundle,
        criterion=criterion,
        device=device,
        cfg=engine_cfg,
    )
    print("test metrics:", out["metrics"])

    for ns in ("bp", "mf", "cc"):
        goterms = go_bundle.encodings[ns].term_ids
        gonames = [go_to_name.get(go, go) for go in goterms]
        pred_path = output_dir / f"pred_{ns}.pkl"
        save_eval_pickle(
            save_path=pred_path,
            ontology=ns,
            proteins=out["proteins"],
            goterms=goterms,
            gonames=gonames,
            y_true=out["y_true"][ns],
            y_pred=out["y_pred"][ns],
        )
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out["metrics"], f, ensure_ascii=False, indent=2)
    print(f"saved -> {output_dir}")


if __name__ == "__main__":
    main()

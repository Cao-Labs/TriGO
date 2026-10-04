from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trigo.go_term import GOEncoderPipeline, GOEncoderPipelineConfig, Node2VecConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Prepare GO namespace Node2Vec embeddings")
    p.add_argument("--obo-path", type=str, required=True)
    p.add_argument("--output-path", type=str, required=True)
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--walk-length", type=int, default=30)
    p.add_argument("--walks-per-node", type=int, default=5)
    p.add_argument("--context-size", type=int, default=5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--num-negative-samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda", help="auto|cuda|cuda:0|cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device.startswith("cuda") and not torch.cuda.is_available():
        resolved_device = "cpu"
    else:
        resolved_device = args.device

    n2v_cfg = Node2VecConfig(
        embedding_dim=args.embedding_dim,
        walk_length=args.walk_length,
        walks_per_node=args.walks_per_node,
        context_size=args.context_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_negative_samples=args.num_negative_samples,
        seed=args.seed,
        device=resolved_device,
    )
    print(f"node2vec device: {resolved_device}")
    cfg = GOEncoderPipelineConfig(
        bp_node2vec=n2v_cfg,
        mf_node2vec=n2v_cfg,
        cc_node2vec=n2v_cfg,
    )
    pipeline = GOEncoderPipeline(cfg)

    summary = pipeline.summarize(args.obo_path)
    print("subgraphs:", summary)

    bundle = pipeline.fit_transform(args.obo_path)
    for ns in bundle.namespaces():
        enc = bundle.get(ns)
        print(f"{ns}: nodes={len(enc.term_ids)} emb_shape={tuple(enc.embeddings.shape)}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle.save(str(output_path))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()

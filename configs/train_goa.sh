#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
DEVICE="${DEVICE:-cuda:0}"

python -u "${REPO_ROOT}/train_model.py" \
  --xlsx-path "${DATA_ROOT}/goa/annotations.xlsx" \
  --pdb-dir "${DATA_ROOT}/goa/pdb" \
  --obo-path "${DATA_ROOT}/goa/go-basic.obo" \
  --feature-cache-dir "${DATA_ROOT}/goa/protein_feature_cache" \
  --go-embedding-path "${DATA_ROOT}/goa/go_node2vec_embeddings.pt" \
  --output-dir "${OUTPUT_ROOT}/goa" \
  --require-split \
  --device "${DEVICE}" \
  --epochs 15 --eval-interval 1 --batch-size 16 --num-workers 8 --seed 42 \
  --lr 1.8e-5 --weight-decay 1e-4 --lr-patience 3 --lr-reduce-factor 0.7 --min-lr 5e-7 \
  --max-residues 1022 --contact-threshold 8.0 \
  --branch-loss-weights '{"bp":1.0,"mf":1.18,"cc":1.03}' \
  --val-score-weights '{"bp":1.0,"mf":1.25,"cc":1.05}' \
  --bp-lr-multiplier 1.5 --mf-lr-multiplier 1.0 --cc-lr-multiplier 1.0 \
  --protein-dropout 0.15 \
  --protein-expert-hidden-dim 320 --protein-expert-output-dim 288 \
  --namespace-attention-topk 512 \
  --bp-namespace-attention-topk 512 \
  --mf-namespace-attention-topk 320 \
  --cc-namespace-attention-topk 256 \
  --use-go-context-residual-gate --go-context-residual-init 0.08 \
  --cross-namespace-refinement-repeats 2 \
  --cross-namespace-refinement-residual-init 0.4 \
  --asym-focal-gamma-pos 0.0 --asym-focal-gamma-neg 1.6 --asym-focal-clip 0.05 \
  --gate-entropy-weight 0.0 \
  --freq-loss-beta 0.999 --freq-loss-weight-min 0.5 --freq-loss-weight-max 2.5 --freq-loss-groups 2 \
  --disable-dynamic-branch-weight --ema-decay 0.999 \
  --amp-dtype bfloat16 --metrics-backend fast --go-space filtered \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
DEVICE="${DEVICE:-cuda:0}"

python -u "${REPO_ROOT}/train_model.py" \
  --xlsx-path "${DATA_ROOT}/cafa3/annotations.xlsx" \
  --pdb-dir "${DATA_ROOT}/cafa3/pdb" \
  --obo-path "${DATA_ROOT}/cafa3/go-basic.obo" \
  --feature-cache-dir "${DATA_ROOT}/cafa3/protein_feature_cache" \
  --go-embedding-path "${DATA_ROOT}/cafa3/go_node2vec_embeddings.pt" \
  --output-dir "${OUTPUT_ROOT}/cafa3" \
  --require-split \
  --device "${DEVICE}" \
  --epochs 25 --eval-interval 1 --batch-size 16 --num-workers 8 --seed 42 \
  --lr 2.3e-5 --weight-decay 7e-5 --lr-patience 4 --lr-reduce-factor 0.7 --min-lr 5e-7 \
  --max-residues 1022 --contact-threshold 8.0 \
  --branch-loss-weights '{"bp":1,"mf":1.02,"cc":1.08}' \
  --val-score-weights '{"bp":1,"mf":1.05,"cc":1.2}' \
  --bp-lr-multiplier 1.6 --mf-lr-multiplier 1.0 --cc-lr-multiplier 1.0 \
  --dropout 0.1 --protein-dropout 0.1 \
  --protein-expert-hidden-dim 256 --protein-expert-output-dim 256 \
  --namespace-attention-topk 512 \
  --bp-namespace-attention-topk 1280 \
  --mf-namespace-attention-topk 448 \
  --cc-namespace-attention-topk 448 \
  --use-go-context-residual-gate --go-context-residual-init 0.12 \
  --cross-namespace-refinement-repeats 8 \
  --cross-namespace-refinement-residual-init 0.5 \
  --asym-focal-gamma-pos 0.0 --asym-focal-gamma-neg 1.8 --asym-focal-clip 0.05 \
  --gate-entropy-weight 0.0 \
  --freq-loss-beta 0.999 --freq-loss-weight-min 0.5 --freq-loss-weight-max 3.0 --freq-loss-groups 2 \
  --disable-dynamic-branch-weight --ema-decay 0.999 \
  --amp-dtype bfloat16 --metrics-backend fast --go-space filtered \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
DEVICE="${DEVICE:-cuda:0}"

python -u "${REPO_ROOT}/train_model.py" \
  --xlsx-path "${DATA_ROOT}/cafa5/annotations.xlsx" \
  --pdb-dir "${DATA_ROOT}/cafa5/pdb" \
  --obo-path "${DATA_ROOT}/cafa5/go-basic.obo" \
  --feature-cache-dir "${DATA_ROOT}/cafa5/protein_feature_cache" \
  --go-embedding-path "${DATA_ROOT}/cafa5/go_node2vec_embeddings.pt" \
  --output-dir "${OUTPUT_ROOT}/cafa5" \
  --require-split \
  --device "${DEVICE}" \
  --epochs 100 --eval-interval 1 --batch-size 16 --num-workers 8 --seed 2026 \
  --lr 2.3e-5 --weight-decay 7e-5 --lr-patience 6 --lr-reduce-factor 0.7 --min-lr 5e-7 \
  --max-residues 1022 --contact-threshold 8.0 \
  --branch-loss-weights '{"bp":1.15,"mf":1.02,"cc":1.08}' \
  --val-score-weights '{"bp":1.0,"mf":1.05,"cc":1.20}' \
  --bp-lr-multiplier 1.75 --mf-lr-multiplier 1.0 --cc-lr-multiplier 1.0 \
  --dropout 0.1 --protein-dropout 0.1 \
  --protein-expert-hidden-dim 256 --protein-expert-output-dim 256 \
  --model-expert-dim 256 --shared-expert-hidden-dim 512 \
  --namespace-expert-hidden-dim 512 --namespace-context-hidden-dim 256 --gate-hidden-dim 128 \
  --namespace-attention-topk 512 \
  --bp-namespace-attention-topk 1280 \
  --mf-namespace-attention-topk 448 \
  --cc-namespace-attention-topk 448 \
  --use-go-context-residual-gate --go-context-residual-init 0.12 \
  --cross-namespace-refinement-repeats 6 \
  --cross-namespace-refinement-residual-init 0.2 \
  --asym-focal-gamma-pos 0.0 --asym-focal-gamma-neg 1.7 --asym-focal-clip 0.05 \
  --gate-entropy-weight 0.0 \
  --freq-loss-beta 0.999 --freq-loss-weight-min 0.5 --freq-loss-weight-max 3.0 \
  --bp-freq-loss-weight-max 3.2 --freq-loss-groups 2 \
  --disable-dynamic-branch-weight --ema-decay 0.999 \
  --amp-dtype bfloat16 --metrics-backend fast --go-space filtered \
  --save-epoch-start 10 --save-epoch-end 100 --save-epoch-interval 10 \
  "$@"

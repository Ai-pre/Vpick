#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-results/gold_pairwise_judge_v4}"

python3 src/evaluate_pairwise_judge.py \
  --scores "$OUT_DIR/scores/pairwise_judge_scores.csv" \
  --sources "$OUT_DIR/input/pairwise_sources_private.csv" \
  --build-summary "$OUT_DIR/input/build_summary.json" \
  --config config/gold_pairwise_judge_v4.json \
  --human-labels "$OUT_DIR/input/human_pairwise_labels.csv" \
  --out-dir "$OUT_DIR/validation"

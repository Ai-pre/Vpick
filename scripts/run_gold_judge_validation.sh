#!/usr/bin/env bash
set -euo pipefail

ALL_GOLD_DATASET="${1:-data/processed/gold_dataset_pairs_all.csv}"
OUT_DIR="${2:-data/processed/gold_judge_v1}"
REPEAT_COUNT="${REPEAT_COUNT:-2}"

# This validation stage intentionally contains published Gold intervals only.
# Performance labels and engagement metrics remain in the private manifest.
python3 src/build_judge_candidates.py \
  --dataset "$ALL_GOLD_DATASET" \
  --scenes-dir data/raw/vpick \
  --out-dir "$OUT_DIR/input"

python3 src/run_llm_judge.py \
  --candidates "$OUT_DIR/input/candidates_blind.csv" \
  --sources "$OUT_DIR/input/candidate_sources_private.csv" \
  --config config/gold_judge_v1.json \
  --out-dir "$OUT_DIR/scores" \
  --mode candidate \
  --repeat-count "$REPEAT_COUNT" \
  --batch-size 1 \
  --workers 6

python3 src/evaluate_llm_judge.py \
  --candidate-scores "$OUT_DIR/scores/candidate_judge_scores.csv" \
  --sources "$OUT_DIR/input/candidate_sources_private.csv" \
  --out-dir "$OUT_DIR/validation"

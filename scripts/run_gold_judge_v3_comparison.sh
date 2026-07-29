#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:-data/processed/gold_judge_v1/input}"
OUT_DIR="${2:-data/processed/gold_judge_v3_comparison}"
REPEAT_COUNT="${REPEAT_COUNT:-2}"

python3 src/run_llm_judge.py \
  --candidates "$INPUT_DIR/candidates_blind.csv" \
  --sources "$INPUT_DIR/candidate_sources_private.csv" \
  --config config/gold_judge_v3_multimodel.json \
  --out-dir "$OUT_DIR/scores" \
  --mode candidate \
  --repeat-count "$REPEAT_COUNT" \
  --batch-size 1 \
  --workers 6

python3 src/evaluate_llm_judge.py \
  --candidate-scores "$OUT_DIR/scores/candidate_judge_scores.csv" \
  --sources "$INPUT_DIR/candidate_sources_private.csv" \
  --out-dir "$OUT_DIR/validation"

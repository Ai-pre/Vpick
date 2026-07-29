#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-data/processed/gold_dataset_pairs_all.csv}"
BASE_INPUT="${2:-data/processed/gold_judge_v1/input}"
OUT_DIR="${3:-results/gold_pairwise_judge_v4}"
REPEAT_COUNT="${REPEAT_COUNT:-2}"

python3 src/build_gold_pairwise_eval.py \
  --dataset "$DATASET" \
  --candidates "$BASE_INPUT/candidates_blind.csv" \
  --sources "$BASE_INPUT/candidate_sources_private.csv" \
  --out-dir "$OUT_DIR/input" \
  --matches-per-neg 2 \
  --annotators 3

RUN_ARGS=(
  --run-id gpt_56_terra_pairwise_v4
  --run-id claude_sonnet5_pairwise_v4
)
if [[ -n "${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}" ]]; then
  RUN_ARGS+=(--run-id gemini_35_flash_pairwise_v4)
fi

python3 src/run_pairwise_judge.py \
  --pairs "$OUT_DIR/input/pairwise_candidates_blind.jsonl" \
  --config config/gold_pairwise_judge_v4.json \
  --out-dir "$OUT_DIR/scores" \
  --repeat-count "$REPEAT_COUNT" \
  --workers 4 \
  "${RUN_ARGS[@]}"

python3 src/evaluate_pairwise_judge.py \
  --scores "$OUT_DIR/scores/pairwise_judge_scores.csv" \
  --sources "$OUT_DIR/input/pairwise_sources_private.csv" \
  --build-summary "$OUT_DIR/input/build_summary.json" \
  --config config/gold_pairwise_judge_v4.json \
  --human-labels "$OUT_DIR/input/human_pairwise_labels.csv" \
  --out-dir "$OUT_DIR/validation"

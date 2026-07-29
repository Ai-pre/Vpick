#!/usr/bin/env bash
set -euo pipefail

PAIR_DIR="${1:-results/gold_pairwise_judge_v4}"
CANDIDATES="${2:-data/processed/gold_judge_v1/input/candidates_blind.csv}"
OUT_DIR="${3:-results/gold_pairwise_judge_v5_multimodal}"
TEXT_SCORES="${4:-results/gold_pairwise_judge_v4/scores/pairwise_judge_scores.csv}"
REPEAT_COUNT="${REPEAT_COUNT:-2}"
WORKERS="${WORKERS:-2}"
MAX_COMPARISONS="${MAX_COMPARISONS:-}"

if [[ -z "${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}" ]]; then
  echo "GEMINI_API_KEY or GOOGLE_API_KEY is required." >&2
  exit 2
fi

RUN_ARGS=()
if [[ -n "$MAX_COMPARISONS" ]]; then
  RUN_ARGS+=(--max-comparisons "$MAX_COMPARISONS")
fi

python3 src/run_gemini_multimodal_pairwise_judge.py \
  --pairs "$PAIR_DIR/input/pairwise_candidates_blind.jsonl" \
  --candidates "$CANDIDATES" \
  --config config/gold_pairwise_judge_v5_multimodal.json \
  --out-dir "$OUT_DIR/scores" \
  --repeat-count "$REPEAT_COUNT" \
  --workers "$WORKERS" \
  "${RUN_ARGS[@]}"

SCORE_ARGS=(--scores "$OUT_DIR/scores/pairwise_judge_scores.csv")
if [[ -f "$TEXT_SCORES" ]]; then
  SCORE_ARGS+=(--scores "$TEXT_SCORES")
fi

python3 src/evaluate_pairwise_judge.py \
  "${SCORE_ARGS[@]}" \
  --sources "$PAIR_DIR/input/pairwise_sources_private.csv" \
  --build-summary "$PAIR_DIR/input/build_summary.json" \
  --config config/gold_pairwise_judge_v5_multimodal.json \
  --human-labels "$PAIR_DIR/input/human_pairwise_labels.csv" \
  --out-dir "$OUT_DIR/validation"

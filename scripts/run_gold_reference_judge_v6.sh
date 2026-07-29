#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-data/processed/gold_dataset_pairs_all_updated.csv}"
OUT_DIR="${2:-results/gold_reference_judge_v6}"
SCORE_DIR="$OUT_DIR/scores_single"
VALIDATION_DIR="$OUT_DIR/validation_single"
REPEAT_COUNT="${REPEAT_COUNT:-2}"

python3 src/build_judge_candidates.py \
  --dataset "$DATASET" \
  --scenes-dir data/raw/vpick \
  --out-dir "$OUT_DIR/input"

python3 src/build_human_reference_sheet.py \
  --candidates "$OUT_DIR/input/candidates_blind.csv" \
  --sources "$OUT_DIR/input/candidate_sources_private.csv" \
  --sample-size "${HUMAN_SAMPLE_SIZE:-15}" \
  --output "$OUT_DIR/input/human_reference_scores.csv"

if [[ "${RUN_JUDGE:-1}" == "1" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is required." >&2
    exit 2
  fi
  python3 src/run_reference_judge.py \
    --candidates "$OUT_DIR/input/candidates_blind.csv" \
    --config config/gold_reference_judge_v6.json \
    --out-dir "$SCORE_DIR" \
    --repeat-count "$REPEAT_COUNT" \
    --batch-size "${JUDGE_BATCH_SIZE:-1}" \
    --workers "${JUDGE_WORKERS:-3}" \
    --retries "${JUDGE_RETRIES:-1}"
fi

python3 src/evaluate_reference_judge.py \
  --scores "$SCORE_DIR/reference_judge_scores.csv" \
  --sources "$OUT_DIR/input/candidate_sources_private.csv" \
  --config config/gold_reference_judge_v6.json \
  --human-scores "$OUT_DIR/input/human_reference_scores.csv" \
  --out-dir "$VALIDATION_DIR"

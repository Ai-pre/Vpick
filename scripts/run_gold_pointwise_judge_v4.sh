#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-data/processed/gold_dataset_pairs_all.csv}"
OUT_DIR="${2:-results/gold_pointwise_judge_v4}"
REPEAT_COUNT="${REPEAT_COUNT:-2}"
RUN_TERRA="${RUN_TERRA:-1}"
RUN_GEMINI="${RUN_GEMINI:-1}"

python3 src/build_judge_candidates.py \
  --dataset "$DATASET" \
  --scenes-dir data/raw/vpick \
  --out-dir "$OUT_DIR/input"

python3 src/build_human_pointwise_sheet.py \
  --candidates "$OUT_DIR/input/candidates_blind.csv" \
  --sources "$OUT_DIR/input/candidate_sources_private.csv" \
  --sample-size "${HUMAN_SAMPLE_SIZE:-15}" \
  --output "$OUT_DIR/input/human_pointwise_scores.csv"

SCORES=()

if [[ "$RUN_TERRA" == "1" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is required for Terra." >&2
    exit 2
  fi
  python3 src/run_pointwise_judge.py \
    --candidates "$OUT_DIR/input/candidates_blind.csv" \
    --config config/gold_pointwise_judge_v4_terra.json \
    --out-dir "$OUT_DIR/terra/scores" \
    --repeat-count "$REPEAT_COUNT" \
    --batch-size 1 \
    --workers "${TERRA_WORKERS:-6}" \
    --retries "${TERRA_RETRIES:-1}"
  SCORES+=(--scores "$OUT_DIR/terra/scores/pointwise_judge_scores.csv")
elif [[ -f "$OUT_DIR/terra/scores/pointwise_judge_scores.csv" ]]; then
  SCORES+=(--scores "$OUT_DIR/terra/scores/pointwise_judge_scores.csv")
fi

if [[ "$RUN_GEMINI" == "1" ]]; then
  if [[ -z "${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}" ]]; then
    echo "GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini." >&2
    exit 2
  fi
  python3 src/run_pointwise_judge.py \
    --candidates "$OUT_DIR/input/candidates_blind.csv" \
    --config config/gold_pointwise_judge_v4_gemini_multimodal.json \
    --out-dir "$OUT_DIR/gemini/scores" \
    --repeat-count "$REPEAT_COUNT" \
    --batch-size 5 \
    --workers 1 \
    --retries "${GEMINI_RETRIES:-0}" \
    --request-interval-sec "${GEMINI_INTERVAL_SEC:-12}"
  SCORES+=(--scores "$OUT_DIR/gemini/scores/pointwise_judge_scores.csv")
elif [[ -f "$OUT_DIR/gemini/scores/pointwise_judge_scores.csv" ]]; then
  SCORES+=(--scores "$OUT_DIR/gemini/scores/pointwise_judge_scores.csv")
fi

if [[ "${#SCORES[@]}" -eq 0 ]]; then
  echo "No pointwise score files are available." >&2
  exit 2
fi

python3 src/evaluate_pointwise_judge.py \
  "${SCORES[@]}" \
  --sources "$OUT_DIR/input/candidate_sources_private.csv" \
  --config config/gold_pointwise_judge_v4_terra.json \
  --human-scores "$OUT_DIR/input/human_pointwise_scores.csv" \
  --out-dir "$OUT_DIR/validation"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-full}"
BASE_DIR="results/gold_reference_judge_balanced_30_30_gpt"
INPUT_DIR="${INPUT_DIR:-$BASE_DIR/input_longform_ytdlp}"
RUN_DIR="${RUN_DIR:-$BASE_DIR/longform_ytdlp}"
HUMAN_SCORES="${HUMAN_SCORES:-$BASE_DIR/input_verified/human_reference_scores.csv}"
CONFIG="config/gold_reference_judge_balanced_30_30_gpt.json"

if [[ ! -f "$INPUT_DIR/candidates_blind.csv" ]]; then
  scripts/build_gold_reference_balanced_input.sh \
    data/processed/gold_reference_relabelled_2026-07-23.csv \
    "$INPUT_DIR"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set." >&2
  exit 2
fi

if [[ "$MODE" == "smoke" ]]; then
  python3 src/run_reference_judge.py \
    --candidates "$INPUT_DIR/candidates_blind.csv" \
    --config "$CONFIG" \
    --out-dir "$RUN_DIR/smoke" \
    --repeat-count 1 \
    --batch-size 5 \
    --workers 1 \
    --max-candidates 5 \
    --retries 1 \
    --no-cache
  exit 0
fi

if [[ "$MODE" != "full" ]]; then
  echo "Usage: $0 [smoke|full]" >&2
  exit 2
fi

python3 src/run_reference_judge.py \
  --candidates "$INPUT_DIR/candidates_blind.csv" \
  --config "$CONFIG" \
  --out-dir "$RUN_DIR/scores" \
  --repeat-count "${REPEAT_COUNT:-2}" \
  --batch-size "${JUDGE_BATCH_SIZE:-5}" \
  --workers "${JUDGE_WORKERS:-3}" \
  --retries "${JUDGE_RETRIES:-2}"

python3 src/evaluate_reference_judge.py \
  --scores "$RUN_DIR/scores/reference_judge_scores.csv" \
  --sources "$INPUT_DIR/candidate_sources_private.csv" \
  --config "$CONFIG" \
  --human-scores "$HUMAN_SCORES" \
  --out-dir "$RUN_DIR/validation"

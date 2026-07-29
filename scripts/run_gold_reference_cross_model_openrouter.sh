#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANDIDATES="$ROOT_DIR/results/gold_reference_judge_v8_ko/input/candidates_blind.csv"
RESULT_ROOT="$ROOT_DIR/results/gold_reference_judge_v8_ko/cross_model"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set." >&2
  exit 2
fi

run_model() {
  local name="$1"
  local config="$2"
  python3 "$ROOT_DIR/src/run_reference_judge.py" \
    --candidates "$CANDIDATES" \
    --config "$ROOT_DIR/config/$config" \
    --out-dir "$RESULT_ROOT/$name" \
    --repeat-count 1 \
    --batch-size 1 \
    --workers 3 \
    --max-tokens 1400 \
    --retries 3 \
    --request-interval-sec 0.2
}

run_model qwen gold_reference_judge_v8_qwen_openrouter.json
run_model mistral gold_reference_judge_v8_mistral_openrouter.json
run_model llama gold_reference_judge_v8_llama_openrouter.json

python3 "$ROOT_DIR/src/aggregate_cross_model_reference_judge.py" \
  --labels "$ROOT_DIR/results/gold_reference_judge_v8_ko/input/candidate_sources_private.csv" \
  --source "codex=$ROOT_DIR/deliverables/2026-07-23/vpick_llm_judge_scores_60.csv" \
  --source "qwen=$RESULT_ROOT/qwen/reference_judge_scores.csv" \
  --source "mistral=$RESULT_ROOT/mistral/reference_judge_scores.csv" \
  --source "llama=$RESULT_ROOT/llama/reference_judge_scores.csv" \
  --out-dir "$ROOT_DIR/deliverables/2026-07-23"

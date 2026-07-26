#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${BEST_JUDGE_INPUT:-${ROOT}/results/judge_evaluation_v2_2026-07-27/short_candidate_descriptions_codex/candidates_blind_short_description_60.jsonl}"
CONFIG="${BEST_JUDGE_CONFIG:-${ROOT}/config/shortform_judge_v10_opus.json}"
OUTPUT="${BEST_JUDGE_OUTPUT:-${ROOT}/results/best_judge_pipeline/run}"
REPEAT_COUNT="${BEST_JUDGE_REPEAT_COUNT:-2}"

python "${ROOT}/src/audit_best_judge_pipeline.py"

python "${ROOT}/src/run_shortform_judge_v9.py" \
  --input "${INPUT}" \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT}" \
  --repeat-count "${REPEAT_COUNT}" \
  "$@"

echo "Scores: ${OUTPUT}/shortform_judge_v9_scores.csv"
echo "Run summary: ${OUTPUT}/run_summary.json"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
DATASET="${1:?usage: run_judge_guided_improvement.sh DATASET SLATE SCENES_DIR OUT_DIR}"
SLATE="${2:?usage: run_judge_guided_improvement.sh DATASET SLATE SCENES_DIR OUT_DIR}"
SCENES_DIR="${3:?usage: run_judge_guided_improvement.sh DATASET SLATE SCENES_DIR OUT_DIR}"
OUT_DIR="${4:?usage: run_judge_guided_improvement.sh DATASET SLATE SCENES_DIR OUT_DIR}"
POINTWISE_SCORES="${POINTWISE_SCORES:-}"
V14_ARTIFACT="${V14_ARTIFACT:-data/private/judge_validation_94/performance_calibrator_v14_dev/shortform_success_judge_v14_dev.joblib}"
TOP_K="${TOP_K:-5}"

"$PYTHON_BIN" src/build_judge_candidates.py \
  --dataset "$DATASET" \
  --source "frozen_candidate_pool=$SLATE" \
  --scenes-dir "$SCENES_DIR" \
  --evidence-mode scenes \
  --require-evidence-source vpick_scenes \
  --require-uniform-provider \
  --exclude-gold \
  --top-k 50 \
  --out-dir "$OUT_DIR/input"

if [[ -z "$POINTWISE_SCORES" ]]; then
  "$PYTHON_BIN" src/run_shortform_judge_v9.py \
    --input "$OUT_DIR/input/candidates_blind.jsonl" \
    --config config/shortform_judge_v10_opus.json \
    --output-dir "$OUT_DIR/pointwise" \
    --repeat-count 1
  POINTWISE_SCORES="$OUT_DIR/pointwise/shortform_judge_v9_scores.csv"
fi

"$PYTHON_BIN" src/predict_shortform_success_v14_dev.py \
  --input "$OUT_DIR/input/candidates_blind.jsonl" \
  --artifact "$V14_ARTIFACT" \
  --output "$OUT_DIR/v14_predictions.json" \
  --allow-development-candidate

"$PYTHON_BIN" src/rerank_candidates_with_judges.py \
  --candidates "$OUT_DIR/input/candidates_blind.jsonl" \
  --pointwise-scores "$POINTWISE_SCORES" \
  --v14-predictions "$OUT_DIR/v14_predictions.json" \
  --output-dir "$OUT_DIR/reranked"

mkdir -p "$OUT_DIR/evaluation"

"$PYTHON_BIN" src/judge_scores_to_predictions.py \
  --dataset "$DATASET" \
  --candidates "$OUT_DIR/input/candidates_blind.jsonl" \
  --scores "$OUT_DIR/input/candidate_sources_private.csv" \
  --score-field source_rank \
  --lower-is-better \
  --output "$OUT_DIR/deterministic_baseline_predictions.csv" \
  --top-k "$TOP_K" \
  --run-id "deterministic_baseline" \
  --selector-type "frozen_deterministic_baseline" \
  --prompt-id "none" \
  --model-name "none"

"$PYTHON_BIN" src/evaluate_predictions.py \
  --dataset "$DATASET" \
  --predictions "$OUT_DIR/deterministic_baseline_predictions.csv" \
  --out-dir "$OUT_DIR/evaluation/deterministic_baseline"

for variant in pointwise_only v14_only hybrid_50_50; do
  "$PYTHON_BIN" src/judge_scores_to_predictions.py \
    --dataset "$DATASET" \
    --candidates "$OUT_DIR/input/candidates_blind.jsonl" \
    --scores "$OUT_DIR/reranked/${variant}_scores.csv" \
    --output "$OUT_DIR/${variant}_predictions.csv" \
    --score-field score \
    --top-k "$TOP_K" \
    --run-id "$variant" \
    --selector-type "$variant" \
    --prompt-id "shortform_judge_v10_ko" \
    --model-name "pointwise_v10_plus_v14"

  "$PYTHON_BIN" src/evaluate_predictions.py \
    --dataset "$DATASET" \
    --predictions "$OUT_DIR/${variant}_predictions.csv" \
    --out-dir "$OUT_DIR/evaluation/$variant"
done

"$PYTHON_BIN" src/summarize_judge_guided_improvement.py \
  --evaluation-dir "$OUT_DIR/evaluation" \
  --output-dir "$OUT_DIR/comparison"

echo "Judge-guided improvement results: $OUT_DIR/evaluation"

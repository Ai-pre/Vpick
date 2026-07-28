#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
DATASET="${1:-data/processed/pilot_dataset_pairs.csv}"
OUT_ROOT="${2:-data/processed/best_no_api_variable_duration}"
STAGE1_DIR="$OUT_ROOT/stage1"
TRIM_DIR="$OUT_ROOT/trim"
RERANK_DIR="$OUT_ROOT/final_rerank"
SLATE_DIR="$OUT_ROOT/longform_slate_top5"
LLM_DIR="$OUT_ROOT/llm_rerank_top5"
LLM_RERANK_DRY_RUN="${LLM_RERANK_DRY_RUN:-1}"
RAW_DIR="${RAW_DIR:-data/raw/vpick}"

"$PYTHON_BIN" src/select_diverse_candidates.py \
  --dataset "$DATASET" \
  --raw-dir "$RAW_DIR" \
  --output "$STAGE1_DIR/predictions.csv" \
  --strategy timeline_bins_bridge \
  --run-id timeline_bins_bridge_k12_b12_max130 \
  --top-k 12 \
  --bins 12 \
  --max-duration-sec 130 \
  --bridge-max-duration-sec 210

"$PYTHON_BIN" src/evaluate_predictions.py \
  --dataset "$DATASET" \
  --predictions "$STAGE1_DIR/predictions.csv" \
  --out-dir "$STAGE1_DIR/evaluation"

"$PYTHON_BIN" src/expand_trim_windows.py \
  --stage1-predictions "$STAGE1_DIR/predictions.csv" \
  --raw-dir "$RAW_DIR" \
  --output "$TRIM_DIR/predictions.csv" \
  --source-rank 20 \
  --source-run-id timeline_bins_bridge_k12_b12_max130 \
  --prepend-duration-sec 35 \
  --duration-sec 30 \
  --duration-sec 45 \
  --duration-sec 60 \
  --duration-sec 75 \
  --windows-per-source 5 \
  --include-scene-boundary-windows \
  --scene-boundary-min-duration-sec 15 \
  --scene-boundary-max-duration-sec 90 \
  --include-speech-boundary-windows \
  --speech-boundary-min-duration-sec 20 \
  --speech-boundary-max-duration-sec 90 \
  --speech-boundary-max-gap-sec 4 \
  --speech-boundary-max-speeches 18 \
  --speech-boundary-lead-pad-sec 1 \
  --speech-boundary-tail-pad-sec 1.5

"$PYTHON_BIN" src/evaluate_predictions.py \
  --dataset "$DATASET" \
  --predictions "$TRIM_DIR/predictions.csv" \
  --out-dir "$TRIM_DIR/evaluation"

"$PYTHON_BIN" src/rerank_trim_candidates.py \
  --dataset "$DATASET" \
  --raw-dir "$RAW_DIR" \
  --predictions "$TRIM_DIR/predictions.csv" \
  --output "$RERANK_DIR/predictions.csv" \
  --top-k 80 \
  --max-overlap 0.45 \
  --source-band-count 8

"$PYTHON_BIN" src/evaluate_predictions.py \
  --dataset "$DATASET" \
  --predictions "$RERANK_DIR/predictions.csv" \
  --out-dir "$RERANK_DIR/evaluation"

"$PYTHON_BIN" src/build_longform_slate.py \
  --predictions "$RERANK_DIR/predictions.csv" \
  --raw-dir "$RAW_DIR" \
  --output "$SLATE_DIR/slate.csv" \
  --top-k 5 \
  --selection-strategy adaptive_coverage \
  --context-pad-sec 20

LLM_DRY_RUN_FLAG=()
if [ "$LLM_RERANK_DRY_RUN" != "0" ]; then
  LLM_DRY_RUN_FLAG=(--dry-run)
fi

"$PYTHON_BIN" src/llm_rerank_top5.py \
  --dataset "$DATASET" \
  --slate "$SLATE_DIR/slate.csv" \
  --runs config/llm_rerank_top5_genre_lang.json \
  --output "$LLM_DIR/predictions.csv" \
  --artifact-dir "$LLM_DIR" \
  --input-top-k 5 \
  --output-top-k 5 \
  "${LLM_DRY_RUN_FLAG[@]}"

"$PYTHON_BIN" src/evaluate_predictions.py \
  --dataset "$DATASET" \
  --predictions "$LLM_DIR/predictions.csv" \
  --out-dir "$LLM_DIR/evaluation"

echo "Stage 1 summary: $STAGE1_DIR/evaluation/summary.json"
echo "Trim summary: $TRIM_DIR/evaluation/summary.json"
echo "Final rerank summary: $RERANK_DIR/evaluation/summary.json"
echo "Longform slate: $SLATE_DIR/slate.csv"
echo "LLM top5 rerank summary: $LLM_DIR/evaluation/summary.json"

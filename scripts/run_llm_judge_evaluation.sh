#!/usr/bin/env bash
set -euo pipefail

ALL_GOLD_DATASET="${1:-data/processed/gold_dataset_pairs_all.csv}"
OUT_DIR="${2:-data/processed/llm_judge_v2}"
OURS_PREDICTIONS="${3:-data/processed/best_pipeline_gold_v1_2026_07_17/llm_rerank_top5_gpt_only/predictions.csv}"
VPICK_PREDICTIONS="${4:-data/processed/best_pipeline_gold_v1_2026_07_17/vpick_baseline_generated_partial/predictions.csv}"
REPEAT_COUNT="${REPEAT_COUNT:-2}"

python3 src/build_judge_candidates.py \
  --dataset "$ALL_GOLD_DATASET" \
  --source "ours=$OURS_PREDICTIONS" \
  --source "vpick=$VPICK_PREDICTIONS" \
  --scenes-dir data/raw/vpick \
  --out-dir "$OUT_DIR/input" \
  --top-k 5

python3 src/run_llm_judge.py \
  --candidates "$OUT_DIR/input/candidates_blind.csv" \
  --sources "$OUT_DIR/input/candidate_sources_private.csv" \
  --config config/llm_judge_v2.json \
  --out-dir "$OUT_DIR/scores" \
  --mode both \
  --set-source ours \
  --set-source vpick \
  --repeat-count "$REPEAT_COUNT" \
  --batch-size 1 \
  --workers 6

python3 src/build_human_pairwise_sheet.py \
  --candidates "$OUT_DIR/input/candidates_blind.csv" \
  --sources "$OUT_DIR/input/candidate_sources_private.csv" \
  --out-dir "$OUT_DIR/human" \
  --system ours \
  --system vpick \
  --pairs-per-long 3 \
  --annotators 3

python3 src/evaluate_llm_judge.py \
  --candidate-scores "$OUT_DIR/scores/candidate_judge_scores.csv" \
  --set-scores "$OUT_DIR/scores/set_judge_scores.csv" \
  --sources "$OUT_DIR/input/candidate_sources_private.csv" \
  --human-labels "$OUT_DIR/human/human_pairwise_labels.csv" \
  --out-dir "$OUT_DIR/validation"

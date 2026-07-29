#!/usr/bin/env bash
set -euo pipefail

: "${YOUTUBE_API_KEY:?Set YOUTUBE_API_KEY before running this script.}"

AS_OF="${1:-$(date +%F)}"

python3 src/label_pilot_performance.py \
  --pairs data/processed/pilot_dataset_pairs.csv \
  --output data/processed/pilot_dataset_pairs.csv \
  --snapshot "data/processed/pilot_channel_short_stats_${AS_OF}.csv" \
  --as-of "$AS_OF" \
  --require-official-api

python3 src/merge_gold_datasets.py \
  --input main=data/processed/gold_dataset_pairs_main.csv \
  --input pilot=data/processed/pilot_dataset_pairs.csv \
  --input control=data/processed/gold_dataset_pairs_control.csv \
  --output data/processed/gold_dataset_pairs_all.csv \
  --summary data/processed/gold_dataset_pairs_all_summary.json

python3 src/build_judge_candidates.py \
  --dataset data/processed/gold_dataset_pairs_all.csv \
  --scenes-dir data/raw/vpick \
  --out-dir data/processed/gold_judge_v1/input

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU_ID VERSION [VERSION ...]" >&2
  exit 2
fi

GPU_ID="$1"
shift

ROOT="${VPICK_ABLATION_ROOT:-$HOME/vpick/experiments/prompt_ablation_posneg_2026-07-27}"
PYTHON="${VPICK_MR3_PYTHON:-$HOME/miniconda3/envs/promptscope/bin/python}"
CANDIDATES="$ROOT/input/candidates_short_description_final_60.csv"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for version in "$@"; do
  case "$version" in
    v1) prompt="v1_baseline_ko.txt" ;;
    v2) prompt="v2_resolution_ko.txt" ;;
    v3) prompt="v3_evidence_first_ko.txt" ;;
    v4) prompt="v4_direct_percentile_ko.txt" ;;
    v5) prompt="v5_funnel_ko.txt" ;;
    *)
      echo "unsupported version: $version" >&2
      exit 2
      ;;
  esac

  "$PYTHON" "$ROOT/src/run_prompt_ablation_mr3.py" \
    --version "$version" \
    --prompt "$ROOT/prompts/$prompt" \
    --candidates "$CANDIDATES" \
    --out-dir "$ROOT/results/$version" \
    --device cuda:0 \
    --dtype bfloat16 \
    --max-new-tokens 1024 \
    --resume
done

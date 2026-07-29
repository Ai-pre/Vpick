#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET="${1:-data/processed/gold_reference_relabelled_2026-07-23.csv}"
OUT_DIR="${2:-results/gold_reference_judge_balanced_30_30_gpt/input_longform_ytdlp}"

SUBTITLE_ARGS=()
for dir in \
  outputs/subtitle_alignment_audit_v2/subtitles \
  outputs/neg_candidate_alignment_2026-07-22_v2/subtitles \
  outputs/neg_candidate_ootb_batch2_2026-07-22/subtitles \
  outputs/neg_candidate_psick_batch2_2026-07-22/subtitles \
  outputs/neg_candidate_psick_batch3_2026-07-22/subtitles \
  outputs/neg_candidate_woni_batch2_2026-07-22/subtitles
do
  if [[ -d "$dir" ]]; then
    SUBTITLE_ARGS+=(--subtitle-cache-dir "$dir")
  fi
done

python3 src/build_judge_candidates.py \
  --dataset "$DATASET" \
  --scenes-dir data/raw/vpick \
  "${SUBTITLE_ARGS[@]}" \
  --evidence-mode long_subtitles \
  --require-evidence-source long_subtitle_interval \
  --require-uniform-provider \
  --out-dir "$OUT_DIR"

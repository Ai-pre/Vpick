#!/usr/bin/env bash
set -euo pipefail

key_line="$(grep '^export ANTHROPIC_API_KEY=' "$HOME/.bashrc" | tail -1)"
if [[ -z "$key_line" ]]; then
  echo "ANTHROPIC_API_KEY export not found in ~/.bashrc" >&2
  exit 1
fi
eval "$key_line"

cd "$HOME/vpick"
python3 src/run_anthropic_listwise_v2.py \
  --batches results/improvement_pipeline_v2_2026-07-29/listwise_judge/batches.jsonl \
  --prompt prompts/hierarchical_multislate_listwise_v2_ko.md \
  --output results/improvement_pipeline_v2_2026-07-29/listwise_judge/anthropic_sonnet4_v2_common18.jsonl \
  --longform-ids results/improvement_pipeline_v2_2026-07-29/listwise_judge/common18_longform_ids.txt \
  --model claude-sonnet-4-20250514 \
  "$@"

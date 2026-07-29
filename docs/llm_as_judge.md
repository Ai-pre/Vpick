# LLM-as-a-Judge Evaluation System

## Scope

This stage validates whether an LLM can serve as a short-form quality evaluator. It evaluates only fixed intervals from published short-form Gold pairs. It does not select a new interval, run the improved pipeline, compare Ours with Vpick, change boundaries, or evaluate final editing effects that are absent from Vpick scene data.

The validation is separated into three layers:

1. GPT Candidate Judge: scores one published Gold interval on six content-quality dimensions.
2. Performance validation: checks whether externally labeled Pos shorts score above Neg shorts and reports alignment with views and likes.
3. Human validation: checks whether the GPT score agrees with blind human judgments.

Selector metrics such as `IoU`, `Core Coverage`, and `Hit@K`, plus Ours/Vpick candidates, belong to a later experiment and are excluded here.

## Leakage Controls

`candidates_blind.csv` contains only the Gold interval evidence required for judging. The following fields are isolated in `candidate_sources_private.csv` and never sent to the LLM:

- Gold pair identity and performance label
- views and likes

Candidate intervals are immutable during judging. A deterministic hash is used as the blind candidate ID.

All 45 rows are published short-form Gold pairs. `performance_label` is independent of Gold status. After refreshing the Pilot channel statistics on 2026-07-21, the current distribution is:

- Pos: 31 high-performance Gold pairs
- Neg: 9 low-performance Gold pairs
- Unlabeled: 5 middle-performance Gold pairs retained for scoring but excluded from Pos/Neg discrimination metrics

Pilot labels use the midpoint percentile of views in a 50-Short channel cohort: percentile >= 75 is Pos, percentile <= 25 is Neg, and the middle 50% remains Unlabeled. The collection date, source, cohort size, and percentile are stored with the dataset so the labels can be reproduced or refreshed.

## Candidate Rubric

Each dimension is scored from 1 to 5 with anchored meanings. The final 0-100 score is computed by code, not accepted directly from the model.

| Dimension | Weight |
|---|---:|
| Opening strength (no fixed three-second rule) | 20% |
| Standalone comprehension | 20% |
| Setup-payoff or claim-conclusion completeness | 20% |
| Engagement or information value | 20% |
| Boundary naturalness | 10% |
| Titleability | 10% |

The same common dimensions are used for every genre. Genre only changes how `engagement_value` and `completeness` are interpreted.

## Run

```bash
REPEAT_COUNT=2 bash scripts/run_gold_judge_validation.sh
```

The required environment variable is `OPENAI_API_KEY`. The config is `config/gold_judge_v1.json` and the selected Judge is GPT-4o mini.

The v2 rubric deliberately removes the old fixed three-second heuristic. It scores whether the first meaningful unit is clear and compelling. Candidate, before-context, and after-context transcripts are supplied separately; outside context may reveal missing setup or payoff but must not be used to fill it in.

Candidate evaluation uses one Gold interval per API request. This prevents surrounding candidates and batch order from changing an otherwise independent score. The model runs at temperature 0, and two repeated runs are retained for rank-stability measurement.

The runner writes:

- `input/candidates_blind.csv`: LLM-safe candidate evidence
- `input/candidate_sources_private.csv`: private source/performance mapping
- `scores/candidate_judge_scores.csv`: per-repeat candidate scores
- `validation/judge_validation_summary.json`: machine-readable validation status
- `validation/judge_validation_report.md`: concise interpretation

## Validation Rule

The code reports `pending_human_labels` until the human preference sheet is completed. It reports `validated` only when both gates pass:

- human preference accuracy >= 0.70
- minimum repeat-run Spearman correlation >= 0.80

The Pos/Neg score gap, AUC, views, likes, and channel percentile are supporting diagnostics. They are not allowed to replace human validation.

## Human Labels

Human evaluation is the next step. Reviewers will receive Gold intervals without Pos/Neg labels or performance metrics and record blind quality judgments. The human results will be compared with GPT scores before the Judge is declared validated.

## Multi-model v3 Experiment

The v3 prompt adds an evidence-sufficiency decision before quality scoring. A judge may return `abstain` when transcript, scene description, or boundary evidence is insufficient, rather than converting missing evidence into a zero-quality score.

The comparison config includes GPT-4o-mini, GPT-5.6 Terra, Claude Sonnet 5, and Gemini 3.5 Flash:

```bash
REPEAT_COUNT=2 bash scripts/run_gold_judge_v3_comparison.sh
```

The runner accepts `--run-id` or `--provider` when only one configured judge should be executed. API keys are read from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY`, and `GEMINI_API_KEY` or `GOOGLE_API_KEY`. See `reports/gold_judge_v3_multimodel_2026-07-21.md` for the current comparison and its limitations.

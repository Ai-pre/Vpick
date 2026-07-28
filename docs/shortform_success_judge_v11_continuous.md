# Shortform Success Judge v11: Continuous Ranking

## Objective

The target system scores a previously unseen shortform candidate with a
`shortform_success_potential_0_100` value. The value is a content-based estimate
of the candidate's relative performance percentile under comparable publishing
conditions. It is not a raw view-count forecast.

The system is a hybrid Judge:

```text
Vpick description, transcript, and boundary context
-> Codex v10 feature extraction
-> continuous performance ranker
-> success-potential score
```

## Codex Features

Codex extracts seven 0-4 features without channel or performance information:

1. self-contained clarity
2. progression and payoff
3. boundary integrity
4. opening pull
5. change or surprise
6. emotional or information gain
7. memorable specificity

`source_salience` is not used because the 94-candidate input lacks a usable
longform overview and the extracted value is constant. Evidence sufficiency,
confidence, reasons, and failure flags are diagnostic only.

## Calibrator Inputs

The continuous ranker uses:

- the seven Codex scores;
- anonymous candidate description and transcript;
- before and after boundary context;
- duration and text-density features;
- line, speaker-marker, question, and exclamation counts;
- missing-description and missing-transcript indicators.

It rejects channel name, views, likes, performance label, performance
percentile, URL, and percentile bucket at prediction time. Transcript source is
not a deployment feature.

## Target and Validation

The sole target is the continuous within-channel Shorts view percentile.
Pos/Neg labels and AUC are not used for splitting, training, model selection,
or reporting.

```text
outer 5-fold GroupKFold by longform_id
+-- inner 4-fold GroupKFold
    +-- choose model family
    +-- choose hyperparameter
```

The primary metrics are:

- within-channel centered Spearman;
- channel-macro Spearman;
- same-channel pairwise ordering accuracy;
- local pairwise accuracy for percentile gaps from 10 to 40;
- longform bootstrap confidence intervals.

A source-presence score is retained only as a post-hoc shortcut control. It is
never passed to the deployment candidate models.

## Current Result

The best development-only model is `pairwise_char_tfidf_numeric`, but choosing
it after inspecting all OOF model results is optimistic. The primary estimate is
therefore the fully nested model-selection pipeline.

| metric | fully nested estimate |
|---|---:|
| channel-centered Spearman | 0.1048 |
| channel-macro Spearman | 0.1345 |
| source-residual Spearman | 0.0579 |
| same-channel pairwise accuracy | 0.5463 |
| local pairwise accuracy | 0.5592 |

The fixed source-presence control obtains channel-centered Spearman `0.1780`.
This is a post-hoc data diagnostic only and is excluded from every acceptance
gate. The fully nested pipeline fails the registered rank correlation,
pairwise accuracy, and bootstrap confidence gates.
The 2,000-repetition longform bootstrap 95% interval for channel-centered
Spearman is `[-0.0961, 0.3212]`.

The correct status is:

```text
experimental_rejected
```

The implementation is usable for research diagnostics and candidate
reranking, but it is not a validated production success predictor.

## Commands

Run the continuous-only validation:

```bash
python src/train_performance_calibrator_v11.py
```

Create the explicitly experimental full-data artifact:

```bash
python src/fit_shortform_success_judge.py --allow-rejected
```

Score one new candidate or a JSONL candidate batch:

```bash
python src/predict_shortform_success.py \
  --input new_candidates.jsonl \
  --allow-experimental
```

The private model artifact contains training vocabulary and is not published.
Public metadata is written to
`results/performance_calibrator_v11/deployment_artifact_METADATA.json`.

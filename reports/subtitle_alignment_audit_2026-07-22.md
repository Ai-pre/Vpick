# YouTube Subtitle Alignment Audit

Date: 2026-07-22

## Purpose

Test whether YouTube subtitles can automatically validate that a published Short is a continuous excerpt from its mapped long-form video. Video files were not downloaded. Only Korean subtitle JSON and public metadata were collected.

## Dataset

- Pairs: 45
- Unique long-form videos: 34
- Unique Shorts: 45
- Performance labels: pos 31, neg 9, unlabeled 5
- Existing manual gold start/end timestamps: 45

## Method

1. Collect manual Korean subtitles when available, otherwise Korean original automatic captions, with `yt-dlp`.
2. Parse caption text and timestamps from JSON3.
3. Split the Short transcript into approximately six-second chunks.
4. Match each chunk against candidate windows in the long-form transcript using RapidFuzz.
5. Select a mostly monotonic alignment path and measure coverage, source gaps, backward jumps, and source-span expansion.
6. Compare the predicted interval with the existing manual gold interval only after alignment. Gold timestamps are not used to find the match.

## Results

| Result | Pairs |
|---|---:|
| Subtitle available for both videos | 42 / 45 (93.3%) |
| Continuous alignment | 18 |
| Light edit suspected | 3 |
| Heavy edit suspected | 6 |
| Insufficient alignment | 15 |
| Missing Short subtitle | 3 |

When the five unlabeled rows are excluded, the operational pos/neg set contains 40 pairs: 15 continuous, 3 light-edit, 5 heavy-edit, 14 insufficient-alignment, and 3 missing-subtitle pairs. Both subtitle tracks were available for 37 / 40 pairs (92.5%).

For the 18 high-confidence continuous alignments:

- Mean gold interval IoU: 0.834
- Median gold interval IoU: 0.872
- Start boundary median absolute error: 0.96 seconds
- End boundary median absolute error: 1.98 seconds
- Start boundary within five seconds: 16 / 18 (88.9%)

Across every aligned pair, including edited candidates:

- Comparable pairs: 27 / 45
- Mean gold interval IoU: 0.686
- Start boundary within five seconds: 21 / 27 (77.8%)

## Review Queue

Light-edit candidates:

| Pair | Channel | Short | Existing gold |
|---|---|---|---:|
| G020 | 빠더너스 | 5HZFbXpM8wM | 166-234 |
| P005 | 안녕하세요원이입니다잘부탁드립니다 | QEpr_lLWw-0 | 1271-1329 |
| P001 | 안녕하세요원이입니다잘부탁드립니다 | c8vCVztYVg8 | 1339-1356 |

Heavy-edit candidates requiring manual confirmation:

| Pair | Channel | Short | Existing gold |
|---|---|---|---:|
| G017 | OOTB_Studio | iKDmq5Pb8VM | 641-700 |
| C003 | 숏박스 | wILE8SzU0j8 | 436-479 |
| G002 | 숏박스 | gQ-nHlRHDaE | 129-153 |
| P006 | 안녕하세요원이입니다잘부탁드립니다 | JFpNSb2pqLQ | 1202-1257 |
| C007 | 워크맨 | ofZ3m43OEhE | 293-310 |
| G027 | 워크맨 | e1D9hyrtyek | 204-271 |

Missing Short subtitle tracks:

- P010: Djc2pShwxpI
- G009: 8QmzdjiuBPo
- G028: vkjmsyrvRDI

## Interpretation

Subtitle alignment is suitable as a high-precision intake filter, not as a universal automatic labeler. A `continuous` result can be accepted automatically with the predicted start/end. `light_edit`, `heavy_edit`, missing subtitles, and insufficient alignments should be reviewed or excluded. Edit status is not a pos/neg performance label and must not be used as one.

The audit also exposed possible boundary-label inconsistencies. For example, P001 is labeled as 1339-1356, while the aligned Short transcript continues through approximately 1419 in the source. This suggests that some current gold intervals describe only the core moment rather than the full published Short.

## Reproduction

```bash
python -m pip install -r requirements.txt
python src/audit_short_long_alignment.py \
  --input results/gold_reference_judge_v6/input/candidate_sources_private.csv \
  --output-dir outputs/subtitle_alignment_audit
```

Generated artifacts:

- `outputs/subtitle_alignment_audit/alignment_audit.csv`
- `outputs/subtitle_alignment_audit/summary.json`
- `outputs/subtitle_alignment_audit/subtitles/*.json3`

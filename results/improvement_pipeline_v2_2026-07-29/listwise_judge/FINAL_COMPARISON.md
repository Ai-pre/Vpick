# Common-18 direct Judge comparison

| Method | Core@1 | Core@3 | Core@5 | Tight@5 | Best IoU@5 |
|---|---:|---:|---:|---:|---:|
| Vpick baseline | 0.000 | 0.056 | 0.056 | 0.056 | 0.060 |
| B2 adaptive coverage | 0.167 | 0.444 | 0.500 | 0.500 | 0.361 |
| Codex intrinsic only + MMR | 0.111 | 0.111 | 0.222 | 0.222 | 0.181 |
| Five-bin prior 75% + Judge 25% | 0.222 | 0.278 | 0.444 | 0.444 | 0.307 |
| **B2 Top4 + five-bin supplement** | **0.167** | **0.444** | **0.556** | **0.556** | **0.405** |

The selected pipeline keeps every B2 hit and adds `G016`. Its rank-5 interval
is `1329.869-1384.172`, while Gold is `1329-1398` (`IoU=0.787`).

Pure Judge replacement is not selected because it removes the timeline
coverage advantage. The Judge is useful here as a constrained supplement.

This is a non-blind development experiment, not a final holdout claim.

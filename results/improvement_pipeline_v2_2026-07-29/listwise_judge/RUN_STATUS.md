# Listwise v2 run status

- Prompt: `prompts/hierarchical_multislate_listwise_v2_ko.md`
- Fair comparison subset: 17 longforms, 18 Gold pairs
- Listwise candidates scored directly by Codex: 320
- Additional B2-only candidates scored: 8
- Total direct scores: 328
- External API used for the final run: no

## Selected development pipeline

1. Preserve B2 adaptive-coverage ranks 1 through 4.
2. Divide the scored pool into five timeline bins.
3. Select the supplement with 75% hierarchical prior and 25% intrinsic Judge.
4. Reject candidates overlapping a selected interval by more than 0.58.
5. Add the supplement as rank 5.

Common-18 result:

- Vpick Core@5: 0.0556
- B2 Core@5: 0.5000
- Selected pipeline Core@5: 0.5556
- Selected pipeline Tight@5: 0.5556
- Selected pipeline Best IoU@5: 0.4050

The selected pipeline adds a hit on `G016` without losing any B2 hit.

## Validation warning

This is a non-blind development result. The current Codex thread had previously
seen some Gold timestamps. Use a new-longform blind holdout before reporting it
as final generalization performance.

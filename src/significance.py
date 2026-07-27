"""Permutation significance for the judge-validation metrics.

The evaluation scripts report Spearman rho and AUC but no uncertainty, so a rho of
0.16 on 60 rows reads the same as a real effect. At these sample sizes that value
is well inside the null distribution, and ranking axes by it produces orderings
that do not survive a reshuffle. Every reported metric should carry a p-value and,
where a ranking is claimed, a multiple-comparison threshold.

Permutation rather than a parametric test: the score distributions are heavily
tied (judges reuse a handful of values) and the percentile target is bimodal, so
the usual normal approximations do not apply.
"""

from __future__ import annotations

import random
from typing import Callable, Iterable, Sequence


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        shared = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            ranks[order[position]] = shared
        index = stop + 1
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else None


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson(average_ranks(xs), average_ranks(ys))


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [s for s, l in zip(scores, labels) if l == 1]
    negatives = [s for s, l in zip(scores, labels) if l == 0]
    if not positives or not negatives:
        return None
    total = 0.0
    for p in positives:
        for q in negatives:
            total += 1.0 if p > q else (0.5 if p == q else 0.0)
    return total / (len(positives) * len(negatives))


def permutation_p(
    statistic: Callable[[Sequence[float]], float | None],
    values: Sequence[float],
    observed: float | None,
    *,
    iterations: int = 20000,
    seed: int = 20260727,
    null_value: float = 0.0,
) -> float | None:
    """Two-sided p for `observed`, shuffling `values` against a fixed partner.

    `statistic` receives a reshuffled copy of `values` and returns the metric.
    `null_value` is the statistic's value under the null (0 for a correlation,
    0.5 for an AUC).
    """
    if observed is None:
        return None
    rng = random.Random(seed)
    shuffled = list(values)
    hits = 0
    effect = abs(observed - null_value)
    for _ in range(iterations):
        rng.shuffle(shuffled)
        candidate = statistic(shuffled)
        if candidate is not None and abs(candidate - null_value) >= effect:
            hits += 1
    return (hits + 1) / (iterations + 1)


def spearman_with_p(
    scores: Sequence[float],
    targets: Sequence[float],
    *,
    iterations: int = 20000,
    seed: int = 20260727,
) -> tuple[float | None, float | None, int]:
    observed = spearman(scores, targets)
    p = permutation_p(
        lambda shuffled: spearman(shuffled, targets),
        scores,
        observed,
        iterations=iterations,
        seed=seed,
        null_value=0.0,
    )
    return observed, p, len(scores)


def auc_with_p(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    iterations: int = 20000,
    seed: int = 20260727,
) -> tuple[float | None, float | None, int]:
    observed = roc_auc(labels, scores)
    label_list = list(labels)

    def statistic(shuffled: Sequence[float]) -> float | None:
        return roc_auc(label_list, shuffled)

    p = permutation_p(
        statistic,
        scores,
        observed,
        iterations=iterations,
        seed=seed,
        null_value=0.5,
    )
    return observed, p, len(scores)


def holm_bonferroni(p_values: dict[str, float | None], alpha: float = 0.05) -> dict[str, dict]:
    """Step-down correction, so a table of metrics can be read as a ranking.

    Holm rather than plain Bonferroni: same guarantee, and it does not throw away
    the whole table when one metric is tested alongside a dozen exploratory ones.
    """
    usable = {k: v for k, v in p_values.items() if v is not None}
    ordered = sorted(usable.items(), key=lambda kv: kv[1])
    total = len(ordered)
    out: dict[str, dict] = {}
    still_rejecting = True
    for rank, (name, p) in enumerate(ordered):
        threshold = alpha / (total - rank) if total - rank else alpha
        if p > threshold:
            still_rejecting = False
        out[name] = {
            "p": round(p, 5),
            "holm_threshold": round(threshold, 5),
            "significant": bool(still_rejecting and p <= threshold),
            "rank": rank + 1,
        }
    for name, value in p_values.items():
        if value is None:
            out[name] = {"p": None, "holm_threshold": None, "significant": False, "rank": None}
    return out


def min_n_for_spearman(rho: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Rough sample size needed to detect `rho`, for reporting alongside results."""
    import math

    if not 0 < abs(rho) < 1:
        return 0
    z_alpha = 1.959963985 if alpha == 0.05 else 1.644853627
    z_beta = 0.841621234 if power == 0.8 else 1.281551566
    fisher = 0.5 * math.log((1 + abs(rho)) / (1 - abs(rho)))
    return int(math.ceil(((z_alpha + z_beta) / fisher) ** 2 + 3))


def describe(name: str, observed: float | None, p: float | None, n: int) -> str:
    if observed is None:
        return f"{name}: n/a (n={n})"
    verdict = "유의" if (p is not None and p < 0.05) else "무의미"
    return f"{name}: {observed:+.4f}  p={p:.4f}  n={n}  -> {verdict}" if p is not None else (
        f"{name}: {observed:+.4f}  p=n/a  n={n}"
    )


def stratified_spearman(
    groups: Iterable[tuple[str, Sequence[float], Sequence[float]]],
    *,
    iterations: int = 20000,
    seed: int = 20260727,
) -> dict[str, dict]:
    """Per-stratum rho with p, plus a sample-weighted macro average.

    Reported per stratum rather than pooled because a stratum missing a label
    class cannot be compared at all, and pooling hides that.
    """
    out: dict[str, dict] = {}
    weighted_sum = 0.0
    weight_total = 0
    for name, scores, targets in groups:
        rho, p, n = spearman_with_p(scores, targets, iterations=iterations, seed=seed)
        out[name] = {"rho": None if rho is None else round(rho, 4), "p": None if p is None else round(p, 5), "n": n}
        if rho is not None:
            weighted_sum += rho * n
            weight_total += n
    out["_macro_weighted"] = {
        "rho": round(weighted_sum / weight_total, 4) if weight_total else None,
        "p": None,
        "n": weight_total,
    }
    return out

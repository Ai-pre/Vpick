from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from train_performance_calibrator_v11 import (
    DEFAULT_PRIVATE_DIR,
    ROOT,
    json_safe,
    load_bundle,
)
from train_performance_calibrator_v12 import candidate_reliability
from train_performance_calibrator_v14_dev import (
    candidate_specs,
    development_metrics,
    remove_proxy_features,
    repeated_nested_oof,
)


DEFAULT_CONFIG = ROOT / "config" / "performance_calibrator_v14_dev.json"
DEFAULT_OUTPUT = (
    ROOT / "results" / "performance_calibrator_v14_ablation" / "summary.json"
)

VARIANTS: dict[str, dict[str, Any]] = {
    "clean_unweighted": {
        "local_pair_boost": 1.0,
        "mid_pair_boost": 1.0,
        "opposite_extreme_pair_weight": 1.0,
        "inner_selection_weights": {
            "mid_channel_centered_spearman": 0.4,
            "mid_pairwise_skill": 0.2,
            "same_channel_local_pairwise_skill": 0.4,
        },
    },
    "balanced_mid2_local3": {
        "local_pair_boost": 3.0,
        "mid_pair_boost": 2.0,
        "opposite_extreme_pair_weight": 0.5,
        "inner_selection_weights": {
            "mid_channel_centered_spearman": 0.35,
            "mid_pairwise_skill": 0.2,
            "same_channel_local_pairwise_skill": 0.45,
        },
    },
    "local_preserving_mid2_local4": {
        "local_pair_boost": 4.0,
        "mid_pair_boost": 2.0,
        "opposite_extreme_pair_weight": 0.5,
        "inner_selection_weights": {
            "mid_channel_centered_spearman": 0.3,
            "mid_pairwise_skill": 0.15,
            "same_channel_local_pairwise_skill": 0.55,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run predeclared three-seed v14 weighting ablations."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, default=3)
    return parser.parse_args()


def variant_config(
    base: dict[str, Any],
    override: dict[str, Any],
    seed_count: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    for key in (
        "local_pair_boost",
        "mid_pair_boost",
        "opposite_extreme_pair_weight",
    ):
        config["pair_weighting"][key] = override[key]
    config["inner_selection_weights"] = override[
        "inner_selection_weights"
    ]
    config["random_seeds"] = config["random_seeds"][:seed_count]
    return config


def main() -> None:
    args = parse_args()
    base = json.loads(args.config.read_text(encoding="utf-8"))
    raw_bundle = load_bundle(args.private_dir)
    bundle, kept_structure = remove_proxy_features(
        raw_bundle,
        [str(value) for value in base["excluded_proxy_features"]],
    )
    reliability = candidate_reliability(bundle)
    rows = []
    for name, override in VARIANTS.items():
        print(f"[v14-ablation] {name}", flush=True)
        config = variant_config(base, override, args.seeds)
        specs = candidate_specs(config)
        scores, tuning, repeat_metrics = repeated_nested_oof(
            bundle,
            specs,
            reliability,
            [int(value) for value in config["random_seeds"]],
            [float(value) for value in config["c_values"]],
            config,
        )
        metrics = development_metrics(
            bundle.y,
            scores,
            bundle.channels,
            float(config["pair_weighting"]["mid_percentile_min"]),
            float(config["pair_weighting"]["mid_percentile_max"]),
            float(config["pair_weighting"]["minimum_gap"]),
            config["inner_selection_weights"],
        )
        mid_skill = 2.0 * float(metrics["mid_only_pairwise_accuracy"]) - 1.0
        local_skill = (
            2.0
            * float(metrics["same_channel_local_pairwise_accuracy"])
            - 1.0
        )
        diagnostic_balance_score = (
            0.4 * float(metrics["mid_only_channel_centered_spearman"])
            + 0.3 * mid_skill
            + 0.3 * local_skill
        )
        rows.append(
            {
                "variant": name,
                "pair_weighting": config["pair_weighting"],
                "inner_selection_weights": config[
                    "inner_selection_weights"
                ],
                "seeds": config["random_seeds"],
                "metrics": metrics,
                "diagnostic_balance_score": diagnostic_balance_score,
                "selection_counts": dict(
                    sorted(
                        Counter(
                            str(item["selected_spec"]) for item in tuning
                        ).items()
                    )
                ),
                "repeat_metrics": repeat_metrics,
            }
        )
    output = {
        "status": "development_ablation_only",
        "warning": (
            "These variants were compared on the same 94 development items. "
            "The best row is a development choice, not a validation result."
        ),
        "seed_count": args.seeds,
        "kept_structure_features": kept_structure,
        "variants": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            json_safe(output),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluate_shortform_success_holdout_v14 import (  # noqa: E402
    evaluate,
    read_targets,
)
from train_performance_calibrator_v11 import json_safe  # noqa: E402


def main() -> None:
    config_path = REPO / "config" / "performance_judge_validation_v14.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = (
        REPO
        / "review_packages"
        / "fable5_v13_cross_validation_2026-07-28"
    )
    targets = read_targets(
        source
        / "data"
        / "reproduction"
        / "validation_targets_94_PRIVATE.csv",
        config,
    )
    oof = pd.read_csv(
        source / "data" / "diagnostics" / "oof_predictions_PRIVATE.csv",
        encoding="utf-8-sig",
    )
    predictions = oof[["candidate_id", "oof_frozen_ensemble"]].copy()
    predictions["shortform_success_potential_0_100"] = (
        predictions["oof_frozen_ensemble"].rank(method="average", pct=True)
        * 100.0
    )
    predictions = predictions[
        ["candidate_id", "shortform_success_potential_0_100"]
    ]

    result = evaluate(
        predictions,
        targets,
        config,
        "v13_repeated_grouped_oof_development_diagnostic",
        False,
    )
    destination = ROOT / "results" / "v13_under_v14" / "result.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            json_safe(result),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(destination)


if __name__ == "__main__":
    main()

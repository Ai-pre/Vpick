from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "README.md",
    "CLAUDE_REVIEW_PROMPT_KO.md",
    "independent_oof_audit.py",
    "verify_package.py",
    "data/oof_predictions_PRIVATE.csv",
    "data/nested_tuning_log_PRIVATE.json",
    "results/v14_summary_PUBLIC.json",
    "results/v14_model_comparison_PUBLIC.csv",
    "results/weighting_ablation_summary.json",
    "results/local_preserving_summary_PUBLIC.json",
    "results/deployment_artifact_METADATA.json",
    "results/artifact_training_scores.json",
    "results/independent_oof_audit.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).exists()]
    if missing:
        raise SystemExit(f"Missing package files: {missing}")

    with (ROOT / "data/oof_predictions_PRIVATE.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        oof = list(csv.DictReader(handle))
    ids = [row["candidate_id"] for row in oof]
    if len(oof) != 94 or len(set(ids)) != 94:
        raise SystemExit("OOF must contain 94 unique candidates.")
    if any(not math.isfinite(float(row["oof_v14_nested"])) for row in oof):
        raise SystemExit("OOF contains a non-finite score.")

    tuning = json.loads(
        (ROOT / "data/nested_tuning_log_PRIVATE.json").read_text(
            encoding="utf-8"
        )
    )
    rows = tuning["tuning_log"]
    if len(rows) != 50:
        raise SystemExit(f"Expected 50 outer-fold logs, found {len(rows)}.")
    if sorted({int(row["repeat_index"]) for row in rows}) != list(range(10)):
        raise SystemExit("Expected repeat indices 0 through 9.")
    if any(len(row["inner_candidates"]) != 15 for row in rows):
        raise SystemExit("Each outer fold must compare 3 specs x 5 C values.")

    summary = json.loads(
        (ROOT / "results/v14_summary_PUBLIC.json").read_text(encoding="utf-8")
    )
    expected = {
        "mid_only_channel_centered_spearman": 0.24954135733214142,
        "mid_only_pairwise_accuracy": 0.6,
        "same_channel_local_pairwise_accuracy": 0.5213270142180095,
        "extremes_pos_neg_auc": 0.7366666666666667,
    }
    for name, value in expected.items():
        observed = float(summary["metrics"][name])
        if not math.isclose(observed, value, rel_tol=0.0, abs_tol=1e-12):
            raise SystemExit(
                f"{name} mismatch: expected {value}, observed {observed}"
            )
    if summary["accepted_as_performance_judge"]:
        raise SystemExit("Development summary must not be marked accepted.")

    metadata = json.loads(
        (ROOT / "results/deployment_artifact_METADATA.json").read_text(
            encoding="utf-8"
        )
    )
    if metadata["accepted_as_performance_judge"]:
        raise SystemExit("Development artifact must not be marked accepted.")
    if metadata["spec"]["name"] != "field_aware_clean_text_only":
        raise SystemExit("Unexpected frozen deployment spec.")
    if float(metadata["c_value"]) != 3.0:
        raise SystemExit("Unexpected frozen C value.")

    training_scores = json.loads(
        (ROOT / "results/artifact_training_scores.json").read_text(
            encoding="utf-8"
        )
    )["results"]
    if len(training_scores) != 94:
        raise SystemExit("Artifact smoke test must contain 94 results.")

    hashes = {
        name: sha256(ROOT / name)
        for name in REQUIRED
    }
    destination = ROOT / "SHA256SUMS.json"
    destination.write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "candidate_count": len(oof),
                "outer_fold_logs": len(rows),
                "artifact_smoke_test_count": len(training_scores),
                "sha256_manifest": str(destination),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

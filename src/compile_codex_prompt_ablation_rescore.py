from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


CHECK_MAP = {
    "hook_within_3s": ("engagement", "opening_pull"),
    "surprise_or_twist": ("engagement", "change_or_surprise"),
    "emotional_peak": ("engagement", "emotional_or_information_gain"),
    "quotable_moment": ("engagement", "memorable_specificity"),
    "payoff_or_conclusion": ("editorial", "progression_payoff"),
}
ALLOWED_FLAGS = {
    "weak_hook",
    "no_surprise",
    "flat_emotion",
    "not_quotable",
    "weak_payoff",
    "awkward_start",
    "awkward_end",
    "asr_degraded",
    "insufficient_evidence",
}
FLAG_MAP = {
    "weak_opening": "weak_hook",
    "no_change": "no_surprise",
    "low_gain": "flat_emotion",
    "not_memorable": "not_quotable",
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def score(judgment: dict[str, Any], section: str, key: str) -> int:
    return int(judgment[section][key]["score"])


def split_lines(transcript: str) -> list[str]:
    lines: list[str] = []
    for raw in transcript.splitlines():
        text = re.sub(r"^\[[^\]]+\]\s*S[^:]*:\s*", "", raw).strip()
        if text and text not in {"[음악]", "[박수]"}:
            lines.append(text)
    return lines


def pick_quote(lines: list[str], keywords: tuple[str, ...]) -> str:
    for line in lines:
        if any(keyword in line for keyword in keywords):
            return line[:180]
    return ""


def mapped_flags(judgment: dict[str, Any]) -> list[str]:
    output: list[str] = []
    for flag in judgment.get("failure_flags", []):
        mapped = FLAG_MAP.get(flag, flag)
        if mapped in ALLOWED_FLAGS and mapped not in output:
            output.append(mapped)
    return output


def checks_0_4(judgment: dict[str, Any]) -> dict[str, int]:
    checks = {
        name: score(judgment, section, key)
        for name, (section, key) in CHECK_MAP.items()
    }
    boundary = score(judgment, "editorial", "boundary_integrity")
    flags = set(judgment.get("failure_flags", []))
    checks["natural_start"] = max(0, boundary - (1 if "awkward_start" in flags else 0))
    checks["natural_end"] = max(0, boundary - (1 if "awkward_end" in flags else 0))
    return checks


def saliency_v2(judgment: dict[str, Any]) -> int:
    opening = score(judgment, "engagement", "opening_pull")
    change = score(judgment, "engagement", "change_or_surprise")
    gain = score(judgment, "engagement", "emotional_or_information_gain")
    memorable = score(judgment, "engagement", "memorable_specificity")
    weighted = 0.29 * opening + 0.23 * change + 0.27 * gain + 0.21 * memorable
    return int(round(clamp(weighted / 4.0 * 100.0, 0.0, 100.0)))


def saliency_v3(judgment: dict[str, Any], base: int) -> int:
    intelligibility = int(judgment["evidence"]["transcript_intelligibility"])
    penalty = {5: 0, 4: 0, 3: 3, 2: 8, 1: 15}.get(intelligibility, 0)
    flags = set(judgment.get("failure_flags", []))
    if "asr_degraded" in flags:
        penalty += 3
    if "visual_dependent" in flags:
        penalty += 4
    return int(round(clamp(base - penalty, 0.0, 100.0)))


def v5_probabilities(
    judgment: dict[str, Any], duration_sec: float
) -> tuple[int, int, int]:
    opening = score(judgment, "engagement", "opening_pull")
    change = score(judgment, "engagement", "change_or_surprise")
    gain = score(judgment, "engagement", "emotional_or_information_gain")
    memorable = score(judgment, "engagement", "memorable_specificity")
    progression = score(judgment, "editorial", "progression_payoff")
    boundary = score(judgment, "editorial", "boundary_integrity")

    stop_anchors = (8, 18, 32, 48, 67)
    share_anchors = (2, 5, 10, 20, 36)
    p_stop = stop_anchors[opening] + (change - 2) * 2
    watch_axis = (progression + boundary + gain) / 3.0
    p_watch = 28 + 15 * watch_axis
    if duration_sec > 90:
        p_watch -= 8
    elif duration_sec > 60:
        p_watch -= 4
    share_axis = int(round((gain + memorable) / 2.0))
    p_share = share_anchors[share_axis] + max(0, change - 2) * 2
    return (
        int(round(clamp(p_stop, 1, 80))),
        int(round(clamp(p_watch, 20, 92))),
        int(round(clamp(p_share, 1, 50))),
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-score prior blind Codex evidence on the V1-V5 output scales."
    )
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    with Path(args.judgments).open("r", encoding="utf-8") as handle:
        judgments = {
            row["candidate_id"]: row
            for line in handle
            if line.strip()
            for row in [json.loads(line)]
        }
    with Path(args.candidates).open("r", encoding="utf-8") as handle:
        candidates = {
            row["candidate_id"]: row
            for line in handle
            if line.strip()
            for row in [json.loads(line)]
        }
    with Path(args.mapping).open("r", encoding="utf-8-sig", newline="") as handle:
        mapping = {
            row["candidate_id"]: row["source_candidate_id"]
            for row in csv.DictReader(handle)
        }
    if not (judgments.keys() == candidates.keys() == mapping.keys()):
        raise ValueError("judgment, candidate, and mapping IDs do not match")

    outputs: dict[str, list[dict[str, Any]]] = {
        version: [] for version in ("v1", "v2", "v3", "v4", "v5")
    }
    audit_rows: list[dict[str, Any]] = []
    for anonymous_id, judgment in judgments.items():
        candidate = candidates[anonymous_id]
        candidate_id = mapping[anonymous_id]
        reason = judgment["reason"]
        flags = mapped_flags(judgment)
        checks4 = checks_0_4(judgment)
        v2_market = saliency_v2(judgment)
        v3_market = saliency_v3(judgment, v2_market)
        v1_market = int(clamp(math.floor(v2_market / 20) + 1, 1, 5))
        checks2 = {
            key: int(clamp(math.floor((value + 1) / 2), 0, 2))
            for key, value in checks4.items()
        }

        lines = split_lines(candidate["transcript"])
        first_quote = lines[0][:180] if lines else ""
        ending_quote = lines[-1][:180] if lines else ""
        twist_quote = pick_quote(lines, ("근데", "아니", "진짜", "갑자기", "하지만"))
        peak_quote = pick_quote(lines, ("웃", "놀", "대박", "미쳤", "왜", "감동"))
        quotable_quote = max(lines, key=len)[:180] if lines else ""

        editorial = (
            score(judgment, "editorial", "source_salience")
            + score(judgment, "editorial", "self_contained_clarity")
            + score(judgment, "editorial", "progression_payoff")
            + score(judgment, "editorial", "boundary_integrity")
        ) / 4.0
        direct_percentile = int(
            round(clamp(0.75 * v2_market + 0.25 * editorial / 4.0 * 100.0, 0, 100))
        )
        duration_sec = float(candidate["duration_ms"]) / 1000.0
        p_stop, p_watch, p_share = v5_probabilities(judgment, duration_sec)
        confidence = int(judgment["confidence_1_5"])

        outputs["v1"].append(
            {
                "candidate_id": candidate_id,
                "verdict": judgment["verdict"],
                "evidence": {
                    "description_support": judgment["evidence"]["description_support"],
                    "transcript_intelligibility": judgment["evidence"][
                        "transcript_intelligibility"
                    ],
                    "boundary_observability": judgment["evidence"][
                        "boundary_observability"
                    ],
                },
                "saliency_market_1_5": v1_market,
                "checks": checks2,
                "overall_shortform_suitable": v1_market >= 3
                and checks2["payoff_or_conclusion"] >= 1,
                "confidence_1_5": confidence,
                "failure_flags": flags,
                "reason": reason,
            }
        )
        outputs["v2"].append(
            {
                "candidate_id": candidate_id,
                "verdict": judgment["verdict"],
                "evidence": {
                    "transcript_intelligibility": judgment["evidence"][
                        "transcript_intelligibility"
                    ],
                    "boundary_observability": judgment["evidence"][
                        "boundary_observability"
                    ],
                },
                "saliency_market_0_100": v2_market,
                "checks": checks4,
                "overall_shortform_suitable": v2_market >= 50
                and checks4["payoff_or_conclusion"] >= 2,
                "confidence_1_5": confidence,
                "failure_flags": flags,
                "reason": reason,
            }
        )
        outputs["v3"].append(
            {
                "candidate_id": candidate_id,
                "verdict": judgment["verdict"],
                "observation": reason,
                "quotes": {
                    "hook_quote": first_quote,
                    "twist_quote": twist_quote,
                    "peak_quote": peak_quote,
                    "quotable_quote": quotable_quote,
                    "ending_quote": ending_quote,
                },
                "checks": checks4,
                "saliency_market_0_100": v3_market,
                "overall_shortform_suitable": v3_market >= 50
                and checks4["payoff_or_conclusion"] >= 2,
                "confidence_1_5": confidence,
                "failure_flags": flags,
                "reason": reason,
            }
        )
        outputs["v4"].append(
            {
                "candidate_id": candidate_id,
                "verdict": judgment["verdict"],
                "reason": reason,
                "channel_percentile_0_100": direct_percentile,
                "confidence_1_5": confidence,
                "failure_flags": flags,
            }
        )
        outputs["v5"].append(
            {
                "candidate_id": candidate_id,
                "verdict": judgment["verdict"],
                "stop_reason": judgment["engagement"]["opening_pull"]["reason"],
                "p_stop": p_stop,
                "watch_reason": (
                    judgment["editorial"]["progression_payoff"]["reason"]
                    + " "
                    + judgment["editorial"]["boundary_integrity"]["reason"]
                ),
                "p_watch": p_watch,
                "share_reason": (
                    judgment["engagement"]["memorable_specificity"]["reason"]
                ),
                "p_share": p_share,
                "confidence_1_5": confidence,
                "failure_flags": flags,
            }
        )
        audit_rows.append(
            {
                "anonymous_candidate_id": anonymous_id,
                "candidate_id": candidate_id,
                "v1_saliency": v1_market,
                "v2_saliency": v2_market,
                "v3_saliency": v3_market,
                "v4_percentile": direct_percentile,
                "v5_p_stop": p_stop,
                "v5_p_watch": p_watch,
                "v5_p_share": p_share,
            }
        )

    out_dir = Path(args.out_dir)
    for version, rows in outputs.items():
        write_jsonl(out_dir / version / "judgments.jsonl", rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "rescore_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    summary = {
        "method": "codex_prompt_aligned_rescore_from_locked_v10_blind_evidence",
        "candidate_count": len(judgments),
        "versions": list(outputs),
        "note": (
            "This is a deterministic prompt-aligned re-score of locked Codex v10 "
            "blind evidence, not five isolated API calls."
        ),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")


def hangul_count(text: str) -> int:
    return len(HANGUL_RE.findall(text))


def repair_text(text: str) -> str:
    candidates = [text]
    for source_encoding in ("gbk", "latin1"):
        try:
            candidates.append(text.encode(source_encoding).decode("cp949"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return max(candidates, key=hangul_count)


def repair_value(value: Any) -> Any:
    if isinstance(value, str):
        return repair_text(value)
    if isinstance(value, list):
        return [repair_value(item) for item in value]
    if isinstance(value, dict):
        return {key: repair_value(item) for key, item in value.items()}
    return value


def all_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(all_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(all_text(item) for item in value.values())
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair GBK/CP949-mojibaked Korean YouTube JSON3 subtitles.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    payload = json.loads(source.read_text(encoding="utf-8"))
    before = hangul_count(all_text(payload))
    repaired = repair_value(payload)
    after = hangul_count(all_text(repaired))
    if after <= before:
        raise RuntimeError(f"Repair did not increase Hangul content: before={before}, after={after}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(repaired, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"input": str(source), "output": str(output), "hangul_before": before, "hangul_after": after}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

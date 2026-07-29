from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from source_evidence import (
    BLIND_FIELDS,
    read_longforms,
    source_outline,
)


PAIRWISE_RUBRIC = """Choose the candidate that is the stronger standalone
short-form highlight while remaining representative of its source video.
Compare these four dimensions equally:
1. Source importance: the segment is an important or representative moment
   within its own source video, rather than filler or transition.
2. Standalone completeness: setup, development, and payoff are understandable
   without watching the full source.
3. Boundary quality: the start provides the minimum needed context and the end
   follows the payoff without truncation or unnecessary trailing material.
4. Engagement: the segment has a concrete hook, reaction, turn, emotional peak,
   memorable information, or quotable moment that can hold a non-subscriber.
Do not reward ASR noise, title wording, channel identity, or information outside
the candidate. Use each candidate's source outline only to judge source
importance and boundaries."""


REFERENCE_ANSWER = """An ideal candidate is a defining moment of its source,
forms a complete micro-story on its own, starts and ends at deliberate edit
points, and contains a strong concrete hook or payoff."""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def candidate_evidence(
    row: dict[str, str],
    outline: str,
) -> str:
    return f"""[원본 장면 개요]
{outline}

[직전 문맥]
{row['before_context']}

[후보 구간]
길이: {row['duration_sec']}초
장면 설명: {row['description'] or '[설명 없음]'}
대사:
{row['transcript']}

[직후 문맥]
{row['after_context']}"""


def pairwise_messages(
    evidence_a: str,
    evidence_b: str,
) -> list[dict[str, str]]:
    prompt = f"""###Task Description:
An instruction, two responses to evaluate (Response A and Response B), a
reference answer, and an evaluation criterion are given.
1. Compare A and B directly using only the evaluation criterion.
2. State the decisive differences instead of reviewing each independently.
3. After the feedback, output the better response as A or B.
4. Output exactly: Feedback: ... [RESULT] A
5. Do not add any opening or closing statement.

###Instruction:
두 후보 중 원본의 좋은 구간을 더 잘 선택했고 독립적인 숏폼 하이라이트로
더 적합한 하나를 고르십시오. 채널 인지도나 성과 라벨은 고려하지 마십시오.

###Response A:
{evidence_a}

###Response B:
{evidence_b}

###Reference Answer:
{REFERENCE_ANSWER}

###Score Rubric:
{PAIRWISE_RUBRIC}

###Feedback:"""
    return [
        {
            "role": "system",
            "content": (
                "You are a fair judge assistant assigned to compare two "
                "candidates objectively under the supplied criterion."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def parse_winner(text: str) -> str:
    matches = re.findall(r"\[RESULT\]\s*([AB])\b", text, flags=re.I)
    if not matches:
        raise ValueError("No pairwise [RESULT] A/B marker found")
    return matches[-1].upper()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "pair_id",
        "channel",
        "order",
        "candidate_a_id",
        "candidate_b_id",
        "winner_position",
        "winner_candidate_id",
        "expected_positive_candidate_id_PRIVATE",
        "is_expected_positive_winner_PRIVATE",
        "feedback",
        "parse_status",
        "input_tokens",
        "generated_tokens",
        "elapsed_sec",
        "model",
        "seed",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="prometheus-eval/prometheus-7b-v2.0")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--longforms", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--outline-max-chars", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    candidate_rows = read_csv(Path(args.candidates))
    if tuple(candidate_rows[0]) != BLIND_FIELDS:
        raise ValueError(f"Unexpected candidate columns: {tuple(candidate_rows[0])}")
    candidates = {row["candidate_id"]: row for row in candidate_rows}
    if len(candidates) != len(candidate_rows):
        raise ValueError("Candidate input contains duplicate IDs")

    manifest_rows = read_csv(Path(args.manifest))
    manifest = {row["candidate_id"]: row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("Manifest contains duplicate IDs")
    if set(candidates) != set(manifest):
        raise ValueError("Candidate and manifest ID sets must match exactly")
    longforms = read_longforms(Path(args.longforms))

    by_channel: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for candidate_id, row in manifest.items():
        label = row["performance_label"]
        if label not in {"pos", "neg"}:
            raise ValueError(f"Unexpected performance label: {label}")
        by_channel[row["channel"]][label].append(candidate_id)

    pair_specs: list[dict[str, str]] = []
    for channel, labels in sorted(by_channel.items()):
        if len(labels["pos"]) != 1 or len(labels["neg"]) != 1:
            raise ValueError(
                f"Pairwise pilot requires one pos and one neg per channel: "
                f"{channel} has pos={len(labels['pos'])}, neg={len(labels['neg'])}"
            )
        positive_id = labels["pos"][0]
        negative_id = labels["neg"][0]
        pair_id = hashlib.sha256(
            f"{channel}|{positive_id}|{negative_id}".encode("utf-8")
        ).hexdigest()[:16]
        pair_specs.append(
            {
                "pair_id": pair_id,
                "channel": channel,
                "positive_id": positive_id,
                "negative_id": negative_id,
            }
        )

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa" if args.device.startswith("cuda") else "eager",
    )
    model.eval()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for pair_index, pair in enumerate(pair_specs):
        for order, candidate_a_id, candidate_b_id in (
            ("AB", pair["positive_id"], pair["negative_id"]),
            ("BA", pair["negative_id"], pair["positive_id"]),
        ):
            evidence: dict[str, str] = {}
            for candidate_id in (candidate_a_id, candidate_b_id):
                longform_id = manifest[candidate_id]["longform_id"]
                if longform_id not in longforms:
                    raise ValueError(f"Missing longform evidence: {longform_id}")
                evidence[candidate_id] = candidate_evidence(
                    candidates[candidate_id],
                    source_outline(
                        longforms[longform_id],
                        args.outline_max_chars,
                        candidate_start_sec=float(
                            manifest[candidate_id]["start_sec"]
                        ),
                        candidate_end_sec=float(
                            manifest[candidate_id]["end_sec"]
                        ),
                    ),
                )
            messages = pairwise_messages(
                evidence[candidate_a_id],
                evidence[candidate_b_id],
            )
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens,
            ).to(args.device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **model_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.perf_counter() - started
            output_ids = generated[0, model_inputs.input_ids.shape[1] :]
            raw_text = tokenizer.decode(output_ids, skip_special_tokens=False)
            try:
                winner_position = parse_winner(raw_text)
                winner_id = (
                    candidate_a_id
                    if winner_position == "A"
                    else candidate_b_id
                )
                parse_status = "winner"
                feedback = re.split(
                    r"\[RESULT\]",
                    raw_text,
                    flags=re.I,
                )[0].strip()
            except ValueError as error:
                winner_position = ""
                winner_id = ""
                parse_status = "parse_error"
                feedback = str(error)

            row = {
                "pair_id": pair["pair_id"],
                "channel": pair["channel"],
                "order": order,
                "candidate_a_id": candidate_a_id,
                "candidate_b_id": candidate_b_id,
                "winner_position": winner_position,
                "winner_candidate_id": winner_id,
                "expected_positive_candidate_id_PRIVATE": pair["positive_id"],
                "is_expected_positive_winner_PRIVATE": (
                    winner_id == pair["positive_id"] if winner_id else ""
                ),
                "feedback": feedback,
                "parse_status": parse_status,
                "input_tokens": int(model_inputs.attention_mask.sum()),
                "generated_tokens": int(output_ids.shape[-1]),
                "elapsed_sec": round(elapsed, 3),
                "model": args.model,
                "seed": args.seed + pair_index,
            }
            rows.append(row)
            digest = hashlib.sha256(
                f"{pair['pair_id']}|{order}|{prompt}".encode("utf-8")
            ).hexdigest()[:24]
            (raw_dir / f"{digest}.json").write_text(
                json.dumps(
                    {**row, "prompt": prompt, "raw_text": raw_text},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            write_csv(out_dir / "pairwise_scores_PRIVATE.csv", rows)
            print(
                json.dumps(
                    {
                        "event": "pair_complete",
                        "completed": len(rows),
                        "total": len(pair_specs) * 2,
                        "parse_status": parse_status,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    grouped_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_results[row["pair_id"]].append(row)
    order_consistent = 0
    complete_pairs = 0
    expected_wins = 0
    parsed_rows = [row for row in rows if row["parse_status"] == "winner"]
    for pair_rows in grouped_results.values():
        parsed_pair = [
            row for row in pair_rows if row["parse_status"] == "winner"
        ]
        if len(parsed_pair) != 2:
            continue
        complete_pairs += 1
        winners = {row["winner_candidate_id"] for row in parsed_pair}
        if len(winners) == 1:
            order_consistent += 1
        expected_wins += sum(
            bool(row["is_expected_positive_winner_PRIVATE"])
            for row in parsed_pair
        )
    summary = {
        "pair_count": len(pair_specs),
        "judgment_count": len(rows),
        "parsed_judgment_count": len(parsed_rows),
        "parse_rate": round(len(parsed_rows) / max(1, len(rows)), 4),
        "complete_pair_count": complete_pairs,
        "order_consistent_pair_count": order_consistent,
        "order_consistency_rate": round(
            order_consistent / max(1, complete_pairs),
            4,
        ),
        "expected_positive_win_rate_PRIVATE": round(
            expected_wins / max(1, len(parsed_rows)),
            4,
        ),
    }
    (out_dir / "pairwise_summary_PRIVATE.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from source_evidence import BLIND_FIELDS, read_longforms, source_outline

DIMENSIONS = {
    "source_importance": {
        "label": "원본 내 중요도",
        "definition": (
            "후보가 원본 영상의 핵심 사건이나 대표 장면에 해당하는지 평가합니다."
        ),
        "anchors": {
            1: "전환·준비·반복처럼 원본의 핵심과 거의 무관한 장면이다.",
            2: "관련은 있지만 빠져도 원본 이해나 재미가 거의 달라지지 않는다.",
            3: "의미 있는 장면이지만 원본의 핵심 장면이라고 보기는 어렵다.",
            4: "원본을 설명할 때 포함할 만한 중요하거나 대표적인 장면이다.",
            5: "원본을 대표하는 결정적 절정이며 가장 먼저 인용할 장면이다.",
        },
    },
    "standalone_completeness": {
        "label": "독립성 및 완결성",
        "definition": (
            "원본을 보지 않은 시청자가 후보만으로 상황과 전개 및 결말을 "
            "이해할 수 있는지 평가합니다."
        ),
        "anchors": {
            1: "맥락 없이는 이해할 수 없고 독립적인 전개가 없다.",
            2: "주제는 짐작되지만 상황이 발전하거나 회수되지 않는다.",
            3: "대체로 이해되지만 전개 또는 결말이 약하거나 일부 맥락이 필요하다.",
            4: "상황과 전개가 분명하며 사전 맥락 없이도 충분히 이해된다.",
            5: "설정·전개·회수가 모두 들어 있는 완결된 짧은 이야기다.",
        },
    },
    "boundary_quality": {
        "label": "시작 및 종료 경계 품질",
        "definition": (
            "후보가 필요한 맥락에서 자연스럽게 시작하고 핵심 회수 뒤에 "
            "자연스럽게 끝나는지 평가합니다."
        ),
        "anchors": {
            1: "문장 중간에서 시작하거나 핵심 회수 전에 끝나 양쪽 경계가 깨졌다.",
            2: "시작 또는 종료 중 하나가 명확히 어색하거나 잘렸다.",
            3: "볼 수는 있지만 시작 또는 종료를 눈에 띄게 다듬을 필요가 있다.",
            4: "필요한 맥락에서 시작하고 핵심 회수 직후 자연스럽게 끝난다.",
            5: "추가하거나 덜어낼 부분 없이 의도적으로 편집한 듯한 경계다.",
        },
    },
    "engagement": {
        "label": "콘텐츠 흡인력",
        "definition": (
            "비구독자의 시선을 붙잡을 훅·변화·반전·감정·정보 이득 또는 "
            "기억에 남는 구체성이 후보 안에 있는지 평가합니다."
        ),
        "anchors": {
            1: "주목할 사건·변화·감정·정보가 없어 바로 넘길 가능성이 높다.",
            2: "약한 재미나 정보는 있지만 시청을 유지할 뚜렷한 순간이 없다.",
            3: "기존 시청자에게는 무난한 하이라이트지만 확장성은 불확실하다.",
            4: "강한 훅·변화·감정·정보가 있어 비구독자도 끝까지 볼 가능성이 높다.",
            5: "제목이 바로 떠오를 만큼 강하고 공유하고 싶은 희소한 절정이다.",
        },
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if tuple(rows[0]) != BLIND_FIELDS:
        raise ValueError(f"Unexpected candidate columns: {tuple(rows[0])}")
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError("Candidate input contains duplicate candidate_id values")
    return rows


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["candidate_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest contains duplicate candidate_id values")
    return {row["candidate_id"]: row for row in rows}


def rubric_text(dimension: str) -> str:
    spec = DIMENSIONS[dimension]
    anchors = "\n".join(
        f"{score}점: {text}" for score, text in spec["anchors"].items()
    )
    return (
        f"평가 차원: {spec['label']}\n"
        f"정의: {spec['definition']}\n{anchors}"
    )


def mr3_messages(
    row: dict[str, str],
    outline: str,
    dimension: str,
) -> list[dict[str, str]]:
    system = f"""# 지시
귀하는 공정한 숏폼 하이라이트 평가자입니다. 아래 원본 영상 개요와 주변
문맥을 참고해 평가할 후보를 단 하나의 차원으로 채점하십시오. 후보 밖의
내용을 후보의 장점으로 인정하지 마십시오. ASR 오탈자 자체를 감점하지
말고 복원 가능한 의미를 평가하십시오. 설명과 대사 모두로 의미를 복원할
수 없을 때만 1점을 주고 explanation에 insufficient_evidence를 명시하십시오.

# 평가 기준
{rubric_text(dimension)}

# 응답 형식
사고 과정 뒤 최종 응답은 다음 JSON 객체 하나여야 합니다.
{{"explanation":"후보의 구체적인 근거를 짧게 설명","score":1}}
score는 1, 2, 3, 4, 5 중 하나의 정수입니다."""
    user = f"""# 원본 영상 장면 개요
{outline}

# 후보 직전 문맥
{row['before_context']}

# 평가할 후보
길이: {row['duration_sec']}초
장면 설명: {row['description'] or '[설명 없음]'}
대사:
{row['transcript']}

# 후보 직후 문맥
{row['after_context']}

# 평가 대상
{DIMENSIONS[dimension]['label']} 차원만 평가하십시오."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def prometheus_prompt(
    row: dict[str, str],
    outline: str,
    dimension: str,
) -> str:
    score_five_reference = DIMENSIONS[dimension]["anchors"][5]
    return f"""###Task Description:
An instruction, a response to evaluate, a reference answer that receives a
score of 5, and a score rubric are given.
1. Write concise feedback strictly based on the score rubric.
2. Write an integer score between 1 and 5.
3. Output exactly: Feedback: ... [RESULT] N
4. Do not add an opening or closing statement.

###The instruction to evaluate:
아래 후보가 한국어 롱폼 영상에서 선택한 독립적인 숏폼 하이라이트로
작동하는지 "{DIMENSIONS[dimension]['label']}" 차원만 평가하십시오.
원본 개요와 직전·직후 문맥은 참고용이며 후보 밖 내용을 후보의 장점으로
인정하지 마십시오. ASR 오탈자 자체는 감점하지 마십시오.

[원본 영상 장면 개요]
{outline}

[직전 문맥]
{row['before_context']}

[직후 문맥]
{row['after_context']}

###Response to evaluate:
[길이] {row['duration_sec']}초
[장면 설명] {row['description'] or '[설명 없음]'}
[대사]
{row['transcript']}

###Reference Answer (Score 5):
이 평가 차원에서 5점 후보는 다음 조건을 충족합니다:
{score_five_reference}

###Score Rubrics:
{rubric_text(dimension)}

###Feedback:"""


def prometheus_messages(
    row: dict[str, str],
    outline: str,
    dimension: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a fair judge assistant tasked with providing clear, "
                "objective feedback based on specific criteria. Apply the "
                "absolute 1-to-5 rubric exactly as written."
            ),
        },
        {
            "role": "user",
            "content": prometheus_prompt(row, outline, dimension),
        },
    ]


def extract_json(text: str) -> dict[str, Any]:
    candidate = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    decoder = json.JSONDecoder()
    for item in fenced + [candidate]:
        item = item.strip()
        try:
            parsed = json.loads(item)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        for match in re.finditer(r"\{", item):
            try:
                parsed, _ = decoder.raw_decode(item[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "score" in parsed:
                return parsed
    score_match = re.search(r'"score"\s*:\s*([1-5])', candidate)
    if score_match:
        return {"explanation": candidate.strip(), "score": int(score_match.group(1))}
    raise ValueError("No score JSON found")


def extract_prometheus(text: str) -> dict[str, Any]:
    matches = re.findall(r"\[RESULT\]\s*([1-5])", text, re.IGNORECASE)
    if not matches:
        matches = re.findall(r"(?:score|점수)\s*[:：]?\s*([1-5])", text, re.I)
    if not matches:
        raise ValueError("No Prometheus score found")
    score = int(matches[-1])
    feedback = re.split(r"\[RESULT\]", text, flags=re.IGNORECASE)[0]
    feedback = re.sub(r"^\s*Feedback\s*:\s*", "", feedback, flags=re.IGNORECASE)
    return {"explanation": feedback.strip(), "score": score}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "candidate_id",
        "dimension",
        "repeat_index",
        "score_1_5",
        "score_100",
        "explanation",
        "parse_status",
        "input_tokens",
        "generated_tokens",
        "elapsed_sec_share",
        "model",
        "family",
        "input_modality",
        "thinking",
        "seed",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("mr3", "prometheus"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--longforms", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input-modality", required=True)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--outline-max-chars", type=int, default=10000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    candidates = read_csv(Path(args.candidates))
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    manifest = read_manifest(Path(args.manifest))
    longforms = read_longforms(Path(args.longforms))
    for row in candidates:
        candidate_id = row["candidate_id"]
        if candidate_id not in manifest:
            raise ValueError(f"Candidate missing from manifest: {candidate_id}")
        longform_id = manifest[candidate_id]["longform_id"]
        if longform_id not in longforms:
            raise ValueError(f"Longform evidence missing: {longform_id}")

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa" if args.device.startswith("cuda") else "eager",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    load_sec = time.perf_counter() - load_started

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    score_path = out_dir / "dimension_scores.csv"
    existing: dict[tuple[str, str, int], dict[str, Any]] = {}
    if args.resume and score_path.exists():
        with score_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (
                    row["candidate_id"],
                    row["dimension"],
                    int(row["repeat_index"]),
                )
                existing[key] = dict(row)

    tasks: list[dict[str, Any]] = []
    for repeat_index in range(1, args.repeat_count + 1):
        for row in candidates:
            candidate_id = row["candidate_id"]
            longform_id = manifest[candidate_id]["longform_id"]
            outline = source_outline(
                longforms[longform_id],
                args.outline_max_chars,
                candidate_start_sec=float(manifest[candidate_id]["start_sec"]),
                candidate_end_sec=float(manifest[candidate_id]["end_sec"]),
            )
            for dimension in DIMENSIONS:
                key = (candidate_id, dimension, repeat_index)
                if key in existing:
                    continue
                if args.family == "mr3":
                    messages = mr3_messages(row, outline, dimension)
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=args.thinking,
                    )
                else:
                    messages = prometheus_messages(row, outline, dimension)
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                tasks.append(
                    {
                        "candidate_id": candidate_id,
                        "dimension": dimension,
                        "repeat_index": repeat_index,
                        "prompt": prompt,
                    }
                )

    score_rows = list(existing.values())
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    for offset in range(0, len(tasks), args.batch_size):
        batch = tasks[offset : offset + args.batch_size]
        prompts = [task["prompt"] for task in batch]
        repeat_indices = {int(task["repeat_index"]) for task in batch}
        if len(repeat_indices) != 1:
            raise ValueError("A generation batch cannot mix repeat indices")
        batch_repeat = next(iter(repeat_indices))
        batch_seed = args.seed + (batch_repeat - 1) * 100_000 + offset
        torch.manual_seed(batch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(batch_seed)
        model_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(args.device)
        input_width = int(model_inputs.input_ids.shape[1])
        generation_args: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.do_sample:
            generation_args.update(
                {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                }
            )
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**model_inputs, **generation_args)
        elapsed = time.perf_counter() - started
        per_item_elapsed = elapsed / max(1, len(batch))

        for batch_index, (task, sequence) in enumerate(zip(batch, generated)):
            output_ids = sequence[input_width:]
            if tokenizer.pad_token_id is not None:
                output_ids = output_ids[output_ids != tokenizer.pad_token_id]
            raw_text = tokenizer.decode(output_ids, skip_special_tokens=False)
            try:
                parsed = (
                    extract_json(raw_text)
                    if args.family == "mr3"
                    else extract_prometheus(raw_text)
                )
                score = int(parsed["score"])
                if score not in {1, 2, 3, 4, 5}:
                    raise ValueError(f"Score out of range: {score}")
                explanation = str(parsed.get("explanation", "")).strip()
                parse_status = "score"
            except (ValueError, TypeError, KeyError) as error:
                score = 0
                explanation = str(error)
                parse_status = "parse_error"

            task_seed = batch_seed
            input_tokens = int(model_inputs.attention_mask[batch_index].sum())
            digest = hashlib.sha256(
                json.dumps(
                    {
                        "model": args.model,
                        "candidate_id": task["candidate_id"],
                        "dimension": task["dimension"],
                        "repeat_index": task["repeat_index"],
                        "prompt": task["prompt"],
                        "seed": task_seed,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:24]
            raw_record = {
                **task,
                "model": args.model,
                "family": args.family,
                "thinking": args.thinking,
                "input_tokens": input_tokens,
                "generated_tokens": int(output_ids.shape[-1]),
                "elapsed_sec_share": per_item_elapsed,
                "raw_text": raw_text,
                "parsed": (
                    {
                        "score": score,
                        "explanation": explanation,
                    }
                    if parse_status == "score"
                    else None
                ),
                "parse_status": parse_status,
            }
            (raw_dir / f"{digest}.json").write_text(
                json.dumps(raw_record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            score_rows.append(
                {
                    "candidate_id": task["candidate_id"],
                    "dimension": task["dimension"],
                    "repeat_index": task["repeat_index"],
                    "score_1_5": score if parse_status == "score" else "",
                    "score_100": (
                        (score - 1) * 25 if parse_status == "score" else ""
                    ),
                    "explanation": explanation,
                    "parse_status": parse_status,
                    "input_tokens": input_tokens,
                    "generated_tokens": int(output_ids.shape[-1]),
                    "elapsed_sec_share": round(per_item_elapsed, 3),
                    "model": args.model,
                    "family": args.family,
                    "input_modality": args.input_modality,
                    "thinking": args.thinking,
                    "seed": task_seed,
                }
            )
        score_rows.sort(
            key=lambda row: (
                int(row["repeat_index"]),
                str(row["candidate_id"]),
                str(row["dimension"]),
            )
        )
        write_csv(score_path, score_rows)
        print(
            json.dumps(
                {
                    "event": "batch_complete",
                    "completed": min(offset + len(batch), len(tasks)),
                    "total": len(tasks),
                    "elapsed_sec": round(elapsed, 3),
                    "score_rows": len(score_rows),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    parsed_count = sum(row["parse_status"] == "score" for row in score_rows)
    summary = {
        "model": args.model,
        "family": args.family,
        "candidate_count": len(candidates),
        "dimension_count": len(DIMENSIONS),
        "repeat_count": args.repeat_count,
        "expected_score_count": len(candidates)
        * len(DIMENSIONS)
        * args.repeat_count,
        "score_row_count": len(score_rows),
        "parsed_score_count": parsed_count,
        "parse_success_rate": round(parsed_count / max(1, len(score_rows)), 4),
        "model_load_sec": round(load_sec, 3),
        "input_modality": args.input_modality,
        "generation": {
            "thinking": args.thinking,
            "do_sample": args.do_sample,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "temperature": args.temperature if args.do_sample else None,
            "top_p": args.top_p if args.do_sample else None,
            "top_k": args.top_k if args.do_sample else None,
            "seed": args.seed,
        },
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

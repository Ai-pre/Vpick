from __future__ import annotations

from typing import Iterable


def count_hangul(text: str) -> int:
    return sum(1 for char in text if "\uac00" <= char <= "\ud7a3")


def count_latin(text: str) -> int:
    return sum(1 for char in text if ("a" <= char.lower() <= "z"))


def detect_prompt_language(texts: Iterable[str], default: str = "en") -> str:
    sample = "\n".join(str(text or "") for text in texts)[:20000]
    hangul = count_hangul(sample)
    latin = count_latin(sample)
    letters = hangul + latin
    if hangul >= 20 and (letters == 0 or hangul / letters >= 0.25):
        return "ko"
    if latin >= 40:
        return "en"
    return default


def choose_prompt_id(default_prompt_id: str, prompt_id_by_language: dict[str, str] | None, language: str) -> str:
    if not prompt_id_by_language:
        return default_prompt_id
    return prompt_id_by_language.get(language) or prompt_id_by_language.get("default") or default_prompt_id


def keyword_score(text: str, weighted_terms: dict[str, float]) -> float:
    lowered = text.lower()
    return sum(lowered.count(term.lower()) * weight for term, weight in weighted_terms.items())


def detect_content_genre(texts: Iterable[str], default: str = "general") -> str:
    sample = "\n".join(str(text or "") for text in texts)[:50000]
    if not sample.strip():
        return default

    lecture_terms = {
        "강의": 3.0,
        "강연": 3.0,
        "수업": 2.5,
        "세미나": 3.0,
        "발표": 2.0,
        "교육": 2.0,
        "개념": 1.5,
        "원리": 1.5,
        "이론": 1.5,
        "프레임워크": 2.0,
        "방법론": 2.0,
        "질문": 1.0,
        "답변": 1.0,
        "슬라이드": 2.0,
        "자료": 1.0,
        "학습": 2.0,
        "lecture": 3.0,
        "talk": 1.5,
        "seminar": 3.0,
        "class": 2.0,
        "lesson": 2.0,
        "concept": 1.5,
        "framework": 2.0,
        "q&a": 2.0,
    }
    variety_vlog_terms = {
        "예능": 3.0,
        "브이로그": 3.0,
        "vlog": 3.0,
        "미션": 2.5,
        "도전": 2.0,
        "친구": 1.5,
        "주문": 2.0,
        "전화": 1.5,
        "반응": 2.0,
        "웃": 1.5,
        "장난": 2.0,
        "사투리": 2.0,
        "호칭": 1.5,
        "고향": 1.5,
        "여행": 1.5,
        "일상": 1.5,
        "카페": 1.0,
        "식당": 1.0,
        "challenge": 2.0,
        "reaction": 2.0,
        "friend": 1.5,
        "travel": 1.5,
    }

    lecture_score = keyword_score(sample, lecture_terms)
    variety_vlog_score = keyword_score(sample, variety_vlog_terms)

    if lecture_score >= 6.0 and lecture_score >= variety_vlog_score * 1.35:
        return "lecture"
    if variety_vlog_score >= 4.0 and variety_vlog_score >= lecture_score * 0.75:
        return "variety_vlog"
    return default


def choose_prompt_id_by_language_and_genre(
    default_prompt_id: str,
    prompt_id_by_language_and_genre: dict[str, dict[str, str]] | None,
    language: str,
    genre: str,
) -> str:
    if not prompt_id_by_language_and_genre:
        return default_prompt_id
    language_mapping = prompt_id_by_language_and_genre.get(language) or prompt_id_by_language_and_genre.get("default")
    if not language_mapping:
        return default_prompt_id
    return language_mapping.get(genre) or language_mapping.get("default") or default_prompt_id

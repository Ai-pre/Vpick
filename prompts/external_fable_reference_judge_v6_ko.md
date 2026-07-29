# Fable 독립 평가 요청문

당신은 롱폼 영상의 특정 구간이 독립적인 숏폼 하이라이트로 적합한지 평가하는 LLM-as-a-Judge입니다.

## 블라인드 규칙

1. 먼저 평가 대상 CSV에서 `candidate_id`, `duration_sec`, `description`, `transcript`, `before_context`, `after_context`만 사용해 독립 채점하십시오.
2. `gold_label`, `pos/neg`, 조회수, 좋아요 수, 채널명, 채널 내 조회수 백분위, 기존 GPT 평가 점수와 판정은 채점이 모두 끝날 때까지 절대 보거나 사용하지 마십시오.
3. 다른 후보와 비교하거나 순위를 매기지 말고, 각 후보를 동일한 절대 기준으로 평가하십시오.
4. `before_context`와 `after_context`는 시작·종료 경계를 판단할 때만 사용하고, 후보 밖 내용을 후보 자체의 완결성으로 계산하지 마십시오.
5. ASR 오류 자체를 콘텐츠 품질로 감점하지 마십시오. 설명과 대사 모두 부족하여 의미를 안정적으로 복원할 수 없을 때만 `abstain` 하십시오.
6. 예능, 브이로그, 인터뷰, 강연 등 장르별 전개 방식은 다르지만 동일한 평가 항목을 적용하십시오.

## 증거 충분성

각 항목을 1~5 정수로 평가하십시오.

- `description_support`: 장면 설명이 후보 내부 사건·행동·반응·주장을 구체적으로 보여주는 정도
- `transcript_intelligibility`: 핵심 상황·질문·반응·주장을 대사에서 복원할 수 있는 정도
- `boundary_observability`: 후보와 전후 문맥으로 시작·종료 경계를 판단할 수 있는 정도

## 하이라이트 중요도

- 1: 준비·이동·반복·배경 설명뿐이며 선택할 이유가 없음
- 2: 의미는 있으나 핵심 사건·주장이 약하고 요약 가치가 낮음
- 3: 사용 가능하지만 강한 하이라이트라고 보기는 어려움
- 4: 명확한 핵심과 충분한 중요도 또는 흥미가 있음
- 5: 원본에서 반드시 보존할 가치가 있는 강하고 독립적인 하이라이트

## 체크리스트

관찰 가능한 근거가 있을 때만 `true`로 판정하십시오.

- `central_focus_clear`: 중심 사건·질문·주장·반응이 명확한가?
- `highlight_worthy`: 단순 연결 장면이 아니라 시청할 가치가 있는가?
- `important_or_representative`: 원본의 중요하거나 대표적인 순간인가?
- `context_sufficient`: 원본 전체 없이도 핵심 내용을 이해할 수 있는가?
- `meaningful_progression`: 상황·행동·주장·감정이 실질적으로 전개되는가?
- `payoff_or_conclusion`: 반응·결과·결론·의미 있는 도착점이 있는가?
- `natural_start`: 핵심 발화나 사건을 훼손하지 않고 시작하는가?
- `natural_end`: 결론이나 반응을 자르지 않고 끝나는가?

## 종합 판정

- `overall_shortform_suitable`: 편집자가 이 구간을 독립 숏폼 후보로 실제 선택할 수 있을 때만 `true`
- `verdict=score`: 안정적으로 판단 가능
- `verdict=abstain`: description과 transcript가 모두 지나치게 불완전함

모든 후보를 다음 열을 가진 CSV로 출력하십시오.

```text
candidate_id,verdict,evidence_description_support_1_5,evidence_transcript_intelligibility_1_5,evidence_boundary_observability_1_5,highlight_saliency_1_5,check_central_focus_clear,check_highlight_worthy,check_important_or_representative,check_context_sufficient,check_meaningful_progression,check_payoff_or_conclusion,check_natural_start,check_natural_end,overall_shortform_suitable,confidence_1_5,failure_flags,reason
```

`failure_flags` 허용값:

```text
weak_evidence|weak_focus|not_highlight_worthy|not_representative|context_dependent|no_progression|weak_payoff|awkward_start|awkward_end|asr_degraded|insufficient_evidence
```

`reason`은 관찰 근거와 불확실성을 한국어 1~2문장으로 작성하십시오. 누락 없이 모든 `candidate_id`를 한 번씩만 평가하십시오.

모든 독립 채점이 끝난 뒤에만 gold label CSV와 기존 GPT 결과 CSV를 결합해 다음을 별도 요약하십시오.

- 채점 성공/abstain 수
- pos와 neg의 평균 점수 차이
- pos-neg AUC
- 기존 GPT 점수와의 Spearman 상관
- `overall_shortform_suitable` 일치율
- 모델 간 점수 차이가 큰 후보 10개와 차이 원인

gold label이나 기존 GPT 결과를 보고 앞서 산출한 개별 점수를 수정하지 마십시오.

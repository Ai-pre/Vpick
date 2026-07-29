# Claude v8 내재적 편집 품질 블라인드 평가

첨부한 `Vpick_Claude_v8_평가패키지.xlsx`와
`Vpick_Claude_v8_평가기준.md`를 사용해 후보 60개를 평가하십시오.

## 핵심 역할

이 작업은 실제 조회수나 채널 내 성과를 맞히는 분류가 아닙니다. 입력에서
관찰되는 구간 자체의 **내재적 편집 품질**만 평가하십시오.

성과가 낮은 영상도 품질이 높을 수 있고, 성과가 높은 영상도 품질이 낮을 수
있습니다. `pos/neg`, `relative_high/low`, 채널명과 실제 성과를 추측하지 마십시오.

## 실행 조건

1. 평가기준 Markdown의 v8 기준을 그대로 적용하십시오.
2. 엑셀의 `blind_candidates` 시트 60행만 평가 입력으로 사용하십시오.
3. 외부 검색, 유튜브 조회, 채널 추정, 조회수·좋아요 추정을 하지 마십시오.
4. 각 후보를 다른 후보와 비교하거나 순위 매기지 말고 독립 평가하십시오.
5. `before_context`와 `after_context`는 시작·종료 경계 판단에만 사용하십시오.
6. ASR 오류 자체를 품질로 감점하지 말고 의미 복원이 불가능할 때만 abstain
   하십시오.
7. 60개 candidate_id를 누락과 중복 없이 정확히 한 번씩 평가하십시오.

## 출력

한 후보당 JSON 객체 한 줄인 JSON Lines만 출력하십시오. 코드 블록, 머리말,
요약, 설명을 붙이지 마십시오.

`score` 예시:

{"candidate_id":"...","verdict":"score","evidence":{"description_support":1,"transcript_intelligibility":4,"boundary_observability":4},"content_mode":"entertainment","editorial_quality_1_5":4,"checks":{"self_contained_context":1,"central_focus_clear":1,"opening_pull":1,"meaningful_progression":1,"payoff_or_conclusion":1,"distinctive_value":1,"memorable_specificity":1,"natural_start":1,"natural_end":1},"overall_editorial_suitable":true,"confidence_1_5":4,"failure_flags":[],"reason":"한국어 1~2문장으로 후보 내부 근거와 불확실성을 설명."}

`abstain`이면 `content_mode`, `editorial_quality_1_5`, `checks`,
`overall_editorial_suitable`를 `null`로 두고 `failure_flags`에
`insufficient_evidence`를 포함하십시오.

허용되는 failure_flags:

`context_dependent`, `weak_focus`, `weak_opening`, `no_progression`,
`weak_payoff`, `low_distinctiveness`, `not_memorable`, `awkward_start`,
`awkward_end`, `visual_dependent`, `asr_degraded`,
`insufficient_evidence`

출력 전에 총 60줄, ID 일치·중복 없음, 점수 범위, JSON 문법을 자체 점검한 뒤
JSON Lines만 제출하십시오.

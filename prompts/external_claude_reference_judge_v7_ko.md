# Claude v7 블라인드 평가 실행 프롬프트

첨부한 `Vpick_Claude_v7_평가패키지.xlsx`와
`Vpick_Claude_v7_평가기준.md`를 사용해 숏폼 후보 60개를 평가하십시오.

## 반드시 지킬 조건

1. 평가 기준 문서의 v7 기준을 그대로 적용하십시오. 기준을 요약하거나 새 항목을
   추가하지 마십시오.
2. 엑셀의 `blind_candidates` 시트만 평가 입력으로 사용하십시오.
3. `candidate_id`, `duration_sec`, `description`, `transcript`,
   `before_context`, `after_context` 외의 정보는 추정하거나 검색하지 마십시오.
4. 외부 웹 검색, 유튜브 조회, 채널 인지도, 조회수·좋아요·성과 라벨을 사용하지
   마십시오.
5. 다른 후보와 비교하거나 순위를 매기지 말고, 같은 절대 기준으로 각 후보를
   독립 평가하십시오.
6. `before_context`와 `after_context`는 시작·종료 경계 판단에만 사용하십시오.
   후보 밖의 재미·반응·결론을 후보 점수에 포함하지 마십시오.
7. ASR 오류 자체는 콘텐츠 품질 감점 사유가 아닙니다. 후보 의미를 안정적으로
   복원할 수 없을 때만 `abstain`하십시오.
8. 60개 `candidate_id`를 누락과 중복 없이 정확히 한 번씩 평가하십시오.

## 출력

결과만 JSON Lines 형식으로 출력하십시오. 코드 블록, 머리말, 요약, 표, 설명을
붙이지 마십시오. 한 후보당 JSON 객체 한 줄입니다.

`verdict="score"`일 때:

{"candidate_id":"...","verdict":"score","evidence":{"description_support":1,"transcript_intelligibility":4,"boundary_observability":4},"saliency_market_1_5":3,"checks":{"hook_within_3s":1,"surprise_or_twist":0,"emotional_peak":2,"quotable_moment":1,"payoff_or_conclusion":2,"natural_start":2,"natural_end":1},"overall_shortform_suitable":true,"confidence_1_5":4,"failure_flags":[],"reason":"한국어 1~2문장. 관찰 근거와 불확실성."}

`verdict="abstain"`일 때는 `saliency_market_1_5`와 `checks`를 `null`로
두고, `failure_flags`에 `insufficient_evidence`를 포함하십시오.

허용되는 `failure_flags`:

`weak_hook`, `no_surprise`, `flat_emotion`, `not_quotable`,
`weak_payoff`, `awkward_start`, `awkward_end`, `visual_dependent`,
`asr_degraded`, `insufficient_evidence`

출력 전 다음을 자체 점검한 뒤 JSON Lines만 제출하십시오.

- 총 60줄인가
- `candidate_id`가 입력과 정확히 일치하고 중복이 없는가
- 모든 점수가 허용 범위 안인가
- `reason`이 한국어 1~2문장인가
- JSON 이외의 텍스트가 없는가

# Gemini 전달용 프롬프트

첨부한 `Vpick_Gemini_평가패키지.xlsx`를 읽고 `후보입력`, `평가기준`, `평가폼` 시트를 사용하여 45개 후보를 모두 평가하십시오.

## 핵심 규칙

1. 각 후보를 다른 후보와 비교하거나 순위를 매기지 말고 고정된 절대 기준으로 독립 평가하십시오.
2. 워크북의 `후보입력` 시트에 제공된 장면 설명, transcript, before_context, after_context만 근거로 사용하십시오.
3. YouTube 링크를 열거나 검색하지 마십시오. 조회수, 좋아요, 채널 규모, 실제 게시 성과를 추측하지 마십시오.
4. before_context와 after_context는 시작·종료 경계를 판단하는 용도로만 사용하고 후보 밖 내용을 후보의 완결성으로 계산하지 마십시오.
5. ASR 오류 자체를 콘텐츠 품질 감점으로 사용하지 말고, 의미를 안정적으로 복원할 수 없을 때만 `insufficient_evidence`로 판단하십시오.
6. `평가기준` 시트의 1~5 anchor와 Boolean 정의를 그대로 적용하십시오.
7. `평가폼` 시트에서 노란색 입력 열만 채우고 자동 계산 열과 수식은 변경하지 마십시오.

## 입력해야 하는 값

- `highlight_saliency_1_5`: 1~5 정수
- `check_*` 8개 항목: Yes는 1, No는 0
- `overall_shortform_suitable`: 현재 시작·종료 구간 그대로 독립 숏폼 후보로 채택 가능하면 1, 아니면 0
- `confidence_1_5`: 입력 증거를 기준으로 한 판단 확신도 1~5
- `failure_flags`: 해당하는 flag를 `|`로 연결, 없으면 빈칸
- `reason`: 관찰한 근거와 불확실성을 한국어 1~2문장으로 설명

## 중요한 판정 정의

- `payoff_or_conclusion=1`: 질문-답변, 행동-결과, 주장-결론 중 하나가 후보 내부에서 실제로 닫힐 때만 1입니다. 단순히 발화가 끝났다는 이유만으로 1을 주지 마십시오.
- `overall_shortform_suitable=1`: 재미의 유무만 보지 말고, 이 구간을 추가 맥락이나 경계 수정 없이 독립 숏폼 후보로 실제 채택할 수 있어야 합니다.
- `important_or_representative=1`: 전체 롱폼의 대표성을 추측하지 말고, 제공된 후보 안에 보존할 가치가 있는 실질적인 사건·주장·반응이 있을 때만 1입니다.

## 결과 반환

가장 우선적으로 `평가폼` 시트를 완성한 Excel 파일을 반환하십시오. 파일 수정이 불가능하면 아래 열 순서를 정확히 지킨 CSV 코드 블록을 반환하십시오. 후보 45개를 빠뜨리지 마십시오.

```text
candidate_id,evaluator_id,highlight_saliency_1_5,check_central_focus_clear,check_highlight_worthy,check_important_or_representative,check_context_sufficient,check_meaningful_progression,check_payoff_or_conclusion,check_natural_start,check_natural_end,overall_shortform_suitable,confidence_1_5,failure_flags,reason
```

`evaluator_id`는 모든 행에 `Gemini`를 입력하십시오. 평가를 시작하기 전에 기준을 임의로 추가하거나 변경하지 말고, 완료 후 누락된 candidate_id가 없는지 확인하십시오.

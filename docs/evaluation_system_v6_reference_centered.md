# 레퍼런스 중심 LLM-as-a-Judge 평가체계 v6

## 1. 평가 질문

고정된 하나의 롱폼 구간이 **하이라이트로서 중요하고, 독립적인 숏폼으로 편집 가능한가**를 LLM이 일관되게 평가할 수 있는지 검증한다.

평가체계의 본체는 pointwise LLM-as-a-Judge다. 사람 평가는 LLM을 대체하지 않으며, Judge의 점수가 사람의 판단과 일치하는지 확인하는 검증용 기준으로만 사용한다.

## 2. 데이터의 역할

- 총 45개는 모두 실제 게시 숏폼에서 원본 구간을 확인한 Gold 구간이다.
- Pos 31개, Neg 9개, Unlabeled 5개는 Gold 여부가 아니라 채널 내 게시 성과 라벨이다.
- LLM 입력에는 성과 라벨, 조회수, 좋아요, 채널명, 숏폼 주소를 넣지 않는다.
- 채널 내 조회수 백분위와 좋아요율은 LLM 채점이 끝난 뒤 외부 타당도 분석에만 결합한다.

## 3. 참고한 레퍼런스와 적용 범위

| 레퍼런스 | 가져온 요소 | 브이픽 적용 |
|---|---|---|
| NIST TREC/TRECVID | 공통 테스트셋, 고정 판단 기준, 블라인드 평가, baseline 비교 | 같은 45개 입력과 같은 프롬프트로 모델을 비교한다. NIST 인증이라고 표현하지 않는다. |
| QVHighlights | 1~5 saliency와 highlight detection 관점 | 쿼리 검색 지표가 아니라 구간의 하이라이트 중요도 1~5만 변형해 사용한다. |
| TVSum | 여러 평가자의 1~5 shot importance 평가 | 사람과 LLM이 같은 1~5 중요도 기준을 사용한다. |
| CheckEval | 주관적 Likert 평가를 세부 Boolean 질문으로 분해 | 8개의 예/아니오 체크리스트를 동일 가중 평균한다. |
| G-Eval | 명시적 평가 단계와 구조화된 form filling | 근거 확인 후 saliency, 체크리스트, 최종 판단을 JSON으로 출력한다. |
| NIST LLM relevance 경고 | LLM 판단을 그대로 Gold로 사용하지 않음 | 인간 평가와 실제 성과 신호로 Judge 자체를 검증한다. |

레퍼런스의 방법을 그대로 재현했다고 주장하지 않는다. 숏폼 구간 평가에 맞게 적용한 자체 프로토콜이며, 각 지표의 원형과 프로젝트 변형을 구분해 보고한다.

## 4. LLM Judge 입력

- 익명 candidate ID
- 구간 길이와 시작·종료 시각
- 언어와 장르
- Vpick 장면 설명
- 후보 내부 transcript
- 경계 확인용 before/after context

조회수, 좋아요, Pos/Neg, 채널명, 게시 성과, 다른 후보의 점수는 입력에서 제외한다.

## 5. LLM Judge 출력

### 5.1 하이라이트 중요도

QVHighlights와 TVSum을 참고한 `highlight_saliency_1_5`를 사용한다.

- 1: 하이라이트 가치가 없음
- 2: 일부 의미가 있으나 약함
- 3: 사용 가능하지만 강한 하이라이트는 아님
- 4: 선택할 만한 좋은 하이라이트
- 5: 반드시 보존할 가치가 있는 매우 강한 하이라이트

### 5.2 Boolean 체크리스트

CheckEval의 세분화 원칙을 참고한다. 모든 항목은 `true/false`이며 동일 가중치다.

1. 중심 사건·질문·주장이 명확하다.
2. 단순 연결 장면이 아니라 하이라이트 가치가 있다.
3. 원본의 중요하거나 대표적인 내용을 담는다.
4. 원본 전체 없이도 핵심 내용을 이해할 수 있다.
5. 상황·행동·주장·감정이 실질적으로 전개된다.
6. 반응·결과·결론 등 의미 있는 도착점이 있다.
7. 시작 경계가 자연스럽다.
8. 종료 경계가 자연스럽다.

### 5.3 점수 계산

- `saliency_score_100 = (saliency - 1) / 4 × 100`
- `checklist_score_100 = true 항목 수 / 8 × 100`
- `reference_score_100 = 0.5 × saliency_score + 0.5 × checklist_score`

최종 점수의 50:50 결합은 브이픽 프로젝트의 명시적 집계 규칙이다. 결과표에는 두 구성 점수를 항상 함께 공개해 임의 가중치의 영향을 확인할 수 있게 한다.

## 6. Judge 검증

### 6.1 반복 안정성

동일한 45개를 순서만 바꿔 2회 평가한다.

- 후보 간 상대 비교가 섞이지 않도록 한 API 요청에는 후보 1개만 넣는다.
- 채점 커버리지 0.80 이상
- `reference_score` 반복 Spearman 0.80 이상
- 최종 적합 여부 반복 일치율 0.80 이상

### 6.2 인간 평가 정합성

성과 라벨과 채널을 층화한 15개를 3명이 독립 평가한다. 한 사람은 15개만 보고 총 45행을 채운다.

- 사람도 LLM과 동일한 saliency와 8개 체크리스트를 사용한다.
- 후보 순서는 평가자마다 무작위로 바꾼다.
- 사람 평균 점수와 LLM 점수의 Spearman 0.70 이상을 목표로 한다.
- 최종 적합 여부의 LLM-사람 정확도 0.70 이상을 목표로 한다.
- 사람 간 saliency 상관과 체크리스트별 Fleiss κ를 함께 보고한다.

### 6.3 실제 성과와의 외부 타당도

- Pos 평균과 Neg 평균의 점수 차이
- Pos가 Neg보다 높은 점수를 받을 AUC
- 채널 내 조회수 백분위와 `reference_score`의 Spearman 상관
- 좋아요율과 `reference_score`의 Spearman 상관

성과 지표는 콘텐츠 외부 요인의 영향을 받으므로 Judge의 정답으로 사용하지 않는다. 인간 정합성이 Judge 채택의 주 검증이고, 성과 정합성은 보조 진단이다.

## 7. 휴먼 평가표 변경점

기존의 `editorial_quality_1_5`와 `performance_potential_1_5` 두 칸은 사용하지 않는다. 사람에게 조회수 잠재력을 추측시키지 않고, LLM과 동일한 다음 항목을 평가하게 한다.

- `highlight_saliency_1_5`
- 8개 `check_*` 예/아니오 항목
- `overall_shortform_suitable`
- `insufficient_evidence`
- `notes`

따라서 휴먼 평가는 별도의 새 평가체계가 아니라 **LLM Judge를 검증하기 위한 동일 루브릭의 인간 버전**이다.

## 8. 해석 규칙

- LLM 점수가 높고 사람 점수와도 일치하면 Judge가 해당 기준을 안정적으로 적용한다고 본다.
- LLM 점수가 성과 라벨과만 일치하고 사람과 불일치하면 조회수 패턴을 우연히 근사한 것이므로 채택하지 않는다.
- 사람 간 일치도가 낮으면 LLM 프롬프트보다 먼저 평가 가이드와 모호한 체크리스트를 수정한다.
- LLM Judge는 Gold 구간을 새로 만드는 도구가 아니라, 주어진 후보의 품질을 같은 기준으로 반복 채점하는 자동 평가자다.

## 9. 실행

```bash
cd ~/vpick
bash scripts/run_gold_reference_judge_v6.sh
```

휴먼 평가 전에는 결과 상태가 `pending_human_scores`다. `results/gold_reference_judge_v6/input/human_reference_scores.csv`를 작성한 뒤 평가 단계만 다시 실행하면 `results/gold_reference_judge_v6/validation_single/`에 최종 검증 상태가 계산된다.

## 10. 출처

- QVHighlights: https://proceedings.neurips.cc/paper/2021/hash/62e0973455fd26eb03e91d5741a4a3bb-Abstract.html
- TVSum: https://openaccess.thecvf.com/content_cvpr_2015/html/Song_TVSum_Summarizing_Web_2015_CVPR_paper.html
- CheckEval: https://aclanthology.org/2025.emnlp-main.796/
- G-Eval: https://aclanthology.org/2023.emnlp-main.153/
- NIST LLM relevance judgment warning: https://www.nist.gov/publications/dont-use-llms-make-relevance-judgments

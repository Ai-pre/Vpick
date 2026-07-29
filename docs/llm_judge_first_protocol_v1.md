# Judge-first 숏폼 평가체계

## 1. 현재 프로젝트의 우선순위

현재 메인 과제는 Ours가 Vpick보다 좋은 후보를 만드는지 확인하는 것이 아니라,
**후보 하나를 일관되고 타당하게 평가하는 LLM-as-a-Judge를 먼저 확립하는 것**이다.

따라서 전체 순서는 다음과 같이 고정한다.

```text
평가 질문 정의
-> 블라인드 pointwise Judge
-> 반복 신뢰도 검증
-> 인간 기준 정합성 검증
-> 실제 성과 신호 정합성 검증
-> 잠금 테스트셋 재검증
-> 통과한 Judge로만 Ours/Vpick 개선 실험
```

Ours와 Vpick 후보 비교는 마지막 단계의 개선 실험이다. Judge 검증 데이터와
프롬프트 튜닝에는 두 시스템의 이름이나 결과를 넣지 않는다.

## 2. Judge가 답하는 질문

입력은 롱폼의 고정된 후보 구간 하나다. Judge는 다음 두 질문에 답한다.

1. `editorial`: 원본에서 이 구간을 고른 선택과 시작·종료 경계가 좋은가
2. `engagement`: 동일 게시 조건을 가정할 때 내용 자체가 시청을 유도할 힘이 있는가

두 축을 분리하는 이유는 성과가 낮은 숏폼도 편집 품질은 높을 수 있기 때문이다.
특히 성과가 전반적으로 높은 채널의 하위 25% 숏폼은 `neg`이지만 내용상 좋은
숏폼일 수 있다. 이런 사례에서 editorial 점수가 높은 것은 Judge 오류가 아니다.
반면 engagement 축이 채널 내부의 상대 성과와 전혀 맞지 않으면 성과 정합성이
검증되지 않은 것이다.

## 3. 평가 입력

Judge가 보는 정보:

- 익명 candidate ID
- 후보 시작·종료 시각과 길이
- 원본 전체의 장면 순서·장면명·설명
- 후보 내부 Vpick 장면 설명
- 후보 내부 대사
- 경계 확인용 직전·직후 문맥
- 시각 장면 설명의 제공 여부

Judge가 보지 않는 정보:

- 채널명
- 조회수와 좋아요
- 채널 성과 백분위
- Pos/Neg
- 숏폼 제목과 URL
- 실제 게시 여부
- 후보 생성 시스템

후보마다 API 요청을 하나씩 보내며, 다른 후보와 비교하거나 배치 안에서 순위를
매기지 않는다.

## 4. v9 평가 기준

### 4.1 편집·구간 선택 품질

| 차원 | 질문 |
|---|---|
| source_salience | 원본 전체에서 핵심적이고 대표적인 구간인가 |
| self_contained_clarity | 원본을 몰라도 상황·논점·반응을 이해할 수 있는가 |
| progression_payoff | 후보 안에서 전개와 의미 있는 도착점이 완성되는가 |
| boundary_integrity | 필요한 맥락에서 시작하고 도착점 직후 자연스럽게 끝나는가 |

### 4.2 내재적 확산 잠재력

| 차원 | 질문 |
|---|---|
| opening_pull | 첫 의미 단위에 시청 이유가 제시되는가 |
| change_or_surprise | 발견·반전·갈등·관점 변화가 실제로 일어나는가 |
| emotional_or_information_gain | 감정 또는 정보 가치가 분명한 정점에 도달하는가 |
| memorable_specificity | 제목·요약·인용이 가능한 구체적 순간이 있는가 |

각 차원은 0~4의 명시적 앵커를 사용한다. 4점은 단순히 괜찮은 후보가 아니라
같은 장르의 실제 공개 숏폼 중에서도 드문 대표 사례 수준이다.

모델은 총점을 만들지 않는다. 코드는 다음 고정식을 사용한다.

```text
editorial_score_100 = editorial 4개 차원의 동일 가중 평균
engagement_score_100 = engagement 4개 차원의 동일 가중 평균
judge_score_100 = 0.5 * editorial + 0.5 * engagement
```

Pos/Neg를 보고 가중치를 학습하지 않는다. 라벨에 맞춰 가중치를 조정하면 Judge
검증과 성과 분류기 학습이 섞이기 때문이다.

## 5. 평가체계의 네 검증축

### 5.1 데이터·누수 검증

- 60개 모두 실제 게시 롱폼-숏폼 페어인지 확인
- 후보 구간 매핑과 Vpick/자막 근거 출처 기록
- 블라인드 입력에 채널·성과·라벨·URL이 없는지 자동 확인
- 정보가 부족하면 0점 대신 `abstain`

### 5.2 반복 신뢰도

같은 60개를 후보당 독립 요청으로 2회 평가한다.

| 지표 | 개발 통과 기준 |
|---|---:|
| 채점 coverage | 0.95 이상 |
| editorial 반복 Spearman | 0.80 이상 |
| engagement 반복 Spearman | 0.80 이상 |
| 각 축 반복 MAE | 10점 이하 |

반복 신뢰도는 같은 판단을 재현하는지를 말할 뿐, 그 판단이 맞는지는 증명하지
않는다.

### 5.3 인간 기준 정합성

사람이 60개 전부를 평가하지 않는다. 채널과 Pos/Neg를 층화한 12개 anchor를
2명이 같은 v9 루브릭으로 독립 평가하면 총 24회다.

| 지표 | 개발 통과 기준 |
|---|---:|
| 사람 간 각 축 Spearman | 0.50 이상 |
| Judge-사람 editorial Spearman | 0.70 이상 |
| Judge-사람 engagement Spearman | 0.60 이상 |

사람 간 일치도가 먼저 낮으면 모델 프롬프트가 아니라 평가 기준이 모호한 것이므로
루브릭을 먼저 수정한다. 기존 v6 인간평가는 항목 정의가 달라 v9의 최종 검증값으로
그대로 재사용하지 않고, 앵커 선정과 모호한 항목 진단에만 참고한다.

### 5.4 실제 성과 외부 타당도

성과 검증에는 `engagement_score_100`만 사용한다. editorial 점수에 Pos/Neg
일치를 강요하지 않는다.

- Pos/Neg는 같은 채널 내 성과 상·하위군이다.
- 채널별 AUC를 계산하고, Pos와 Neg가 각각 3개 이상인 채널의 Macro AUC를
  주 지표로 사용한다.
- Judge 점수와 채널 내 성과 백분위를 채널 평균 중심화한 뒤 상관을 계산한다.
- 전체 pooled AUC는 채널 규모 차이가 섞이므로 참고값으로만 보고한다.

| 지표 | 개발 통과 기준 |
|---|---:|
| 안정 채널 Macro AUC | 0.60 이상 |
| 채널 중심화 성과 상관 | 0.20 이상 |

Pos/Neg는 절대적인 좋은/나쁜 콘텐츠 정답이 아니다. 이 검증은 Judge의
engagement 축이 실제 성과와 최소한 같은 방향의 신호를 갖는지 확인하는
외부 타당도 검사다.

## 6. Pointwise와 Pairwise의 역할

- `pointwise`: 최종 Judge의 본체. 새 후보 하나에 단독 점수를 줄 수 있다.
- `pairwise`: 위치 편향과 기준 민감도를 확인하는 보조 진단.

모든 후보 조합을 비교하지 않는다. 공개 숏폼과 같은 롱폼의 경계 절단·payoff
제거 후보 등 예상 방향이 명확한 12쌍만 A/B 순서를 뒤집어 검사한다. 이는
Ours/Vpick 개선 실험이 아니라 Judge가 명백한 품질 저하를 감지하는지 보는
criterion-sensitivity 테스트다.

## 7. 최종 채택 상태

| 상태 | 의미 |
|---|---|
| pending_human_anchor_scores | 반복 실행은 끝났지만 인간 앵커가 비어 있음 |
| needs_revision_reliability | 같은 입력에 판단이 안정적으로 재현되지 않음 |
| needs_revision_editorial_validity | 사람의 구간 품질 판단과 맞지 않음 |
| editorial_only_engagement_unvalidated | 편집 Judge로만 사용 가능, 성과 관련 주장 불가 |
| development_pass_pending_locked_test | 현재 60개 개발셋 통과, 새 데이터 검증 대기 |
| validated | 미사용 잠금 테스트셋에서도 모든 기준 통과 |

현재 60개는 여러 프롬프트 실험에 이미 사용했으므로 개발셋이다. 이 데이터에서
통과하더라도 바로 `validated`로 부르지 않는다. 프롬프트를 고정한 뒤 새로 모은
20~30개 실제 페어를 한 번만 평가해 최종 상태를 결정한다.

## 8. 현재 판정

- v6 GPT: 반복 신뢰도는 통과했지만 인간 종합점수 상관 0.0697, Pos/Neg AUC
  0.4731로 타당도 실패
- v8 Codex: 안정 채널 Macro AUC 0.4927로 성과 외부 타당도 실패
- Opus 4.8 6축 고정 총점: AUC 0.504
- Opus 4.8 v7 체크리스트: AUC 0.364

따라서 현재 검증된 Judge는 없다. 위 결과는 모델을 더 크게 바꾸기 전에 평가
축과 검증 대상을 분리해야 한다는 근거다.

### v9 Opus 실행 상태

- 1회차 60/60 완료
- 2회차 11/60 완료
- Anthropic 워크스페이스 사용 한도로 49건 대기
- 2회 공통 11개의 예비 Spearman: editorial 0.8488, engagement 0.8654
- 2회 공통 coverage가 11/60이므로 신뢰도 판정은 `incomplete_repeat_run`
- 반복 완료 전에는 Pos/Neg 외부 타당도를 열지 않음

부분 결과는 `results/shortform_judge_v9/PARTIAL_RUN_REPORT.md`에 기록한다.

## 9. 다음 실행 순서

1. v9 프롬프트와 점수식을 고정한다.
2. Opus 4.8로 60개를 후보당 단독 요청, 2회 반복 평가한다.
3. 반복 신뢰도 기준을 확인한다.
4. 12개 anchor에 대해 2인 인간 평가를 완료한다.
5. editorial 인간 정합성과 engagement 인간·성과 정합성을 각각 계산한다.
6. 12개 통제 변형 쌍으로 기준 민감도와 A/B 위치 편향을 확인한다.
7. 개발 기준을 통과하면 프롬프트를 동결하고 새 20~30개 잠금 테스트셋에서
   한 번만 최종 검증한다.
8. `validated` 이후에만 Ours와 Vpick 후보를 같은 Judge로 비교한다.

## 10. 구현 파일

```text
prompts/shortform_judge_v9_ko.md
config/shortform_judge_v9_opus.json
src/shortform_judge_v9.py
src/run_shortform_judge_v9.py
src/evaluate_shortform_judge_v9.py
```

실행:

```bash
python src/run_shortform_judge_v9.py \
  --repeat-count 2 \
  --no-cache

python src/evaluate_shortform_judge_v9.py \
  --scores results/shortform_judge_v9/shortform_judge_v9_scores.csv \
  --targets deliverables/2026-07-24/performance_judge_v1/candidate_targets_PRIVATE.csv \
  --human-scores deliverables/2026-07-24/shortform_judge_v9/human_anchor_scores.csv \
  --dataset-role development \
  --out-dir results/shortform_judge_v9/validation
```

## 11. 참고 방법

- QVHighlights: query-relevant clip의 5점 saliency와 다중 하이라이트 관점
- TVSum: 여러 평가자의 shot importance 평가와 순위 상관
- G-Eval: 명시적 평가 단계와 구조화된 form filling
- CheckEval: 모호한 종합평가를 세부 기준으로 분해
- NIST LLM relevance 논의: LLM 판단을 검증 없이 정답으로 사용하지 않음

출처:

- https://proceedings.neurips.cc/paper/2021/hash/62e0973455fd26eb03e91d5741a4a3bb-Abstract.html
- https://openaccess.thecvf.com/content_cvpr_2015/html/Song_TVSum_Summarizing_Web_2015_CVPR_paper.html
- https://aclanthology.org/2023.emnlp-main.153/
- https://aclanthology.org/2025.emnlp-main.796/
- https://www.nist.gov/publications/dont-use-llms-make-relevance-judgments

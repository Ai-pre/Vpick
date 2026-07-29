# Vpick Evaluation System v1

## 목적

이 모듈은 하이라이트 선택 방식을 개선하지 않는다. 고정된 후보를 어떤 방식으로
평가해야 하는지 다섯 체계를 동일 데이터에서 비교한다.

핵심 질문은 두 개다.

1. Judge 점수가 채널 규모를 제거한 실제 쇼츠 성과와 관계가 있는가?
2. Judge가 독립 숏폼 품질과 원본에서의 선택 적합성을 구분하는가?

## 데이터

현재 기준 데이터:

- 공개 숏폼 60개
- 롱폼 54개
- 채널 6개
- 채널 비교 코호트 302개
- Vpick 장면 분석 또는 yt-dlp transcript fallback
- 업로드일 27/60
- 7일·30일 고정 관측 조회수 0/60

Judge 입력은 `results/evaluation_system_v1/prepared/candidates_blind.jsonl`이고,
성과·채널 정보는 `behavior_labels_private.csv`에만 둔다.

### Blind candidate schema

```json
{
  "candidate_id": "PJ_...",
  "longform_id": "...",
  "start_ms": 0,
  "end_ms": 0,
  "duration_ms": 0,
  "longform_overview": [],
  "description": "...",
  "transcript": "...",
  "before_context": "...",
  "after_context": "...",
  "visual_evidence_available": false
}
```

Standalone Case는 `longform_overview`, `before_context`, `after_context`를 제거한다.

### Private behavior schema

주요 필드:

- `short_views`: 관측 시점 누적 조회수
- `upload_date`, `stats_snapshot_date`, `upload_age_days`
- `channel_median_views`
- `relative_log_view_score`
- `channel_view_percentile`
- `performance_tier`: 연속값에서 파생한 top25/middle50/bottom25
- `behavior_label_status`: 현재는 `exploratory`
- `dataset_split`: longform 단위 train/validation/test

`legacy_performance_label`은 과거 추적용이다. Judge 입력이나 주 성과 지표로 쓰지 않는다.

## 성과 정규화

```text
relative_log_view_score
= log2((views + 1) / (same_channel_median_views + 1))
```

- 0: 채널 중앙값과 같음
- 1: 중앙값의 약 2배
- -1: 중앙값의 약 절반

`channel_view_percentile`은 같은 채널 코호트에서 mid-rank 방식으로 계산한다.
두 값을 합치지 않고 각각 검증한다.

고정 7일·30일 조회수가 없기 때문에 업로드 경과일이 있는 27개는 보조 변수를
보존하고, 나머지 33개와 함께 현재 누적 조회수 결과는 탐색적으로만 해석한다.

## 다섯 Case

### Case 1 - channel baseline

LLM을 사용하지 않는다. 채널 중앙값 대비 점수와 백분위의 일관성, leave-one-out,
상·하단 극단값 제거, 코호트 bootstrap 민감도를 검사한다.

### Case 2 - standalone pointwise

후보 설명·대사·길이만 보고 hook, engagement, self-contained, payoff, density,
boundary를 0~4점으로 평가한다.

프롬프트:
`prompts/evaluation_standalone_pointwise_v1_ko.md`

출력 스키마:
`schemas/evaluation_standalone_pointwise_v1.schema.json`

### Case 3 - source-conditioned pointwise

원본 전체 개요, 직전·직후 문맥까지 보고 source salience, relative competitiveness,
hook, self-contained, payoff, density, boundary를 평가한다.

프롬프트:
`prompts/evaluation_source_pointwise_v1_ko.md`

출력 스키마:
`schemas/evaluation_source_pointwise_v1.schema.json`

### Case 4 - source-conditioned pairwise

같은 `longform_id`의 A/B만 비교한다. 각 pair는 AB와 BA 두 순서를 만들어
순서 일관성을 측정한다. 현재 공개 숏폼끼리 비교 가능한 pair는 8개다.

프롬프트:
`prompts/evaluation_source_pairwise_v1_ko.md`

출력 스키마:
`schemas/evaluation_source_pairwise_v1.schema.json`

### Case 5 - hybrid

Standalone 점수와 Source Selection 점수를 별도로 유지한다.

```text
hybrid = alpha * standalone + (1 - alpha) * source
alpha in {0.3, 0.5, 0.7}
```

alpha는 validation split의 성과 백분위 Spearman으로만 선택하고 test에서는 고정한다.

## 검증 지표

- 성과: Spearman, Top-25 ROC-AUC, PR-AUC, 상·하위 25% Macro F1
- 일반화: 채널별 지표, channel holdout, longform group split
- 불확실성: longform 단위 bootstrap 95% CI
- 반복성: pointwise repeat Spearman과 MAE
- 순서 안정성: pairwise AB/BA consistency
- 동일 롱폼 순위: Top-1과 NDCG
- 인간 정합성: 항목별 Spearman, weighted kappa 또는 alpha

현재 인간 응답지는 비어 있으므로 인간 정합성은 N/A다. 값은 만들지 않았다.

## 인간 라벨

생성 파일:

- `deliverables/2026-07-24/evaluation_system_v1/human_source_pointwise_tasks.jsonl`
- `deliverables/2026-07-24/evaluation_system_v1/human_source_pointwise_responses.csv`
- `deliverables/2026-07-24/evaluation_system_v1/human_source_pairwise_tasks.jsonl`
- `deliverables/2026-07-24/evaluation_system_v1/human_source_pairwise_responses.csv`

두 평가자는 동일 task를 독립적으로 평가한다. 채널·조회수·성과 라벨은 보지 않는다.

## 실제 결과

| Case | 실제 범위 | 성과 백분위 rho | Top-25 AUC | 안정성 |
|---|---:|---:|---:|---:|
| Case 1 | 60 | 0.9555* | N/A | LOO 백분위 변화 0.2025p |
| Case 2 | 29 current overlap | 0.1083 | 0.4944 | repeat rho 0.8070 |
| Case 3 | 57 scored | 0.0525 | 0.4865 | repeat rho 0.8611 on 11 |
| Case 4 | 124 synthetic pairs | N/A | N/A | order consistency 0.6532 |
| Case 5 | 28 overlap | 0.0971 | 0.5234 | alpha 0.7 selected on validation |

\* Case 1의 rho는 두 성과 정규화 신호 간 일관성이지 Judge-성과 상관이 아니다.

## 해석

- Case 1의 정규화는 현재 코호트 안에서 안정적이다.
- Case 2와 3은 반복 가능한 채점이 가능하지만 성과를 맞히지 못한다.
- Case 4는 A/B 순서 민감도가 높다.
- Case 5는 구성 요소보다 일관되게 좋아지지 않았다.
- 따라서 성과 예측 Judge로 승인된 Case는 없다.

잠정 운영 후보는 Case 3이지만 용도는 `원본 조건부 하이라이트 품질 진단`이다.
성과 예측 타당성을 주장하려면 고정 관측 기간 성과, 현재 루브릭 2인 인간 라벨,
Case 2 전체 반복, Case 3 미완료 반복이 필요하다.

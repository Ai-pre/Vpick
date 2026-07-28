# Vpick 성과 예측 Judge v14 개발 패키지

## 한 줄 결론

v14는 v13이 실패했던 **중간 성과 구간 순위화**를 개선했다. 그러나 표본이 34건뿐이고
95% CI가 0을 포함하므로 아직 검증된 Judge가 아니다.

## 1. 왜 v14를 만들었는가

v13의 전체 채널 중심 Spearman은 `0.3125`였지만, mid 34건에서는 채널 중심
Spearman이 `-0.2084`였다. 쉬운 POS-vs-NEG 구분이 전체 수치를 올리고 실제로 어려운
중간 구간에서는 순서를 맞히지 못했다.

v14는 다음을 바꿨다.

1. `mid↔mid` 학습 pair를 3배 가중
2. 성과 차이가 비슷한 local pair를 2배 가중
3. 너무 쉬운 `neg↔pos` pair를 0.25배로 축소
4. 채널마다 pair 총가중치를 동일하게 정규화
5. 타임스탬프·화자표시·괄호·줄 수 프록시 특징 제거
6. 정규화한 후보 설명·자막·전후 문맥만 사용
7. outer fold 안에서 표현 구조와 C를 inner CV로만 선택

## 2. 개발 OOF 결과

| 지표 | v13 | v14 | 변화 |
|---|---:|---:|---:|
| 전체 채널 중심 Spearman | 0.3125 | 0.3355 | +0.0230 |
| **mid 채널 중심 Spearman** | **-0.2084** | **+0.2495** | **+0.4579** |
| mid pairwise | 0.4333 | 0.6000 | +0.1667 |
| 같은 라벨 내부 pairwise | 0.4732 | 0.5536 | +0.0804 |
| local pairwise | 0.5877 | 0.5213 | -0.0664 |
| POS-vs-NEG AUC | 0.7144 | 0.7367 | +0.0223 |

부트스트랩 95% CI:

- mid 채널 중심 Spearman: `[-0.1912, 0.6148]`
- mid pairwise: `[0.3213, 0.8409]`
- local pairwise: `[0.3737, 0.6555]`
- POS-vs-NEG AUC: `[0.5909, 0.8606]`

mid 점추정치는 크게 개선됐지만 CI가 0을 포함한다. local pairwise도 우연 0.5와
구별되지 않는다. 따라서 v14는 **현재 가장 나은 개발 후보**일 뿐 최종 승인 모델이
아니다.

## 3. 가중치 ablation

같은 94건에서 3-seed 파일럿을 수행했다.

| 변형 | mid rho | mid pairwise | local pairwise | 해석 |
|---|---:|---:|---:|---|
| clean 무가중 | 0.0742 | 0.5167 | 0.5450 | 프록시 제거만으로는 부족 |
| mid2/local3 | 0.0338 | 0.5500 | 0.5687 | local은 회복, mid 약함 |
| mid2/local4 | 0.0638 | 0.5667 | 0.5735 | local 보존형 |
| **mid3/local2, 10-seed** | **0.2495** | **0.6000** | 0.5213 | mid 목표에 가장 적합 |

local 보존형을 10-seed 단일 텍스트 구조로 다시 돌리면 mid rho `0.1873`,
local pairwise `0.5355`였다. v13과 결합하면 mid 성능이 다시 낮아졌으므로 최종
개발 후보는 mid 집중형을 유지했다.

이 비교 자체도 같은 개발 데이터에서 수행됐다. 가중치 개선 수치는 홀드아웃 증거가
아니다.

## 4. 배포 후보 artifact

전체 94건에서 등록된 10개 seed로 내부 CV를 반복한 결과:

- 10/10회 `field_aware_clean_text_only` 선택
- 최다 C 선택: `3.0` 6회
- 후보별 평균 내부 선택 점수도 `C=3.0`이 최고

따라서 배포 후보를 다음처럼 고정했다.

```text
description + transcript -> semantic char/word TF-IDF
before_context + after_context -> context char TF-IDF
-> mid/local 가중 Pairwise Logistic Regression (C=3.0)
-> 개발 참조 분포 대비 0~100 percentile
```

입력에서 다음 값은 금지된다.

- 채널명, 조회수, 좋아요
- POS/NEG/mid 라벨과 성과 백분위
- 영상 URL/ID
- 자막 출처

artifact는 private 경로에만 저장한다. 공개 패키지에는 학습 어휘가 포함된
`joblib` 대신 구조와 입력 정책만 담은 metadata를 제공한다.

## 5. 검증 상태

```text
development_only_not_validated
```

이유:

1. v14 설계는 v13 실패를 같은 94건에서 확인한 뒤 만들어졌다.
2. mid 표본이 34건뿐이다.
3. mid rho와 local pairwise CI가 귀무값을 포함한다.
4. 실제 배포 artifact를 fresh holdout에 적용한 결과가 없다.

엄격 검증 규약은 `config/performance_judge_validation_v14.json`에 고정했다.

- 전체 최소 180건
- mid-enriched 표본 최소 150건
- mid 채널 중심 rho ≥ 0.20, CI 하한 > 0
- local pairwise ≥ 0.60, CI 하한 > 0.50
- 극단 AUC ≥ 0.65, CI 하한 > 0.50

## 6. 파일

| 파일 | 내용 |
|---|---|
| `data/oof_predictions_PRIVATE.csv` | 94건 v14 nested OOF |
| `data/nested_tuning_log_PRIVATE.json` | fold별 구조·C 선택과 inner 점수 |
| `results/v14_summary_PUBLIC.json` | 핵심 수치·CI·반복 안정성 |
| `results/v14_model_comparison_PUBLIC.csv` | v13·duration·가짜 판정자 비교 |
| `results/weighting_ablation_summary.json` | 3개 가중치 파일럿 |
| `results/local_preserving_summary_PUBLIC.json` | local 보존형 10-seed 결과 |
| `results/deployment_artifact_METADATA.json` | 고정 artifact 구조와 금지 입력 |
| `results/artifact_training_scores.json` | 94건 artifact 스모크 테스트 |
| `independent_oof_audit.py` | OOF CSV만 이용한 표준 라이브러리 재계산 |
| `verify_package.py` | 개수·구조·핵심 수치·해시 검증 |

## 7. 재현

```bash
python src/train_performance_calibrator_v14_dev.py

python src/fit_shortform_success_judge_v14_dev.py

python src/predict_shortform_success_v14_dev.py \
  --input new_candidates.jsonl \
  --output holdout_predictions.json \
  --allow-development-candidate

python src/evaluate_shortform_success_holdout_v14.py \
  --predictions holdout_predictions.json \
  --targets holdout_targets_PRIVATE.csv \
  --output holdout_result.json \
  --holdout-id fresh_holdout_v1 \
  --confirm-fresh-untouched
```

홀드아웃 점수를 확인한 뒤 구조·가중치·게이트를 바꾸면 그 데이터는 더 이상
holdout이 아니다.

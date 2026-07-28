# Claude/Fable5 독립 검토 요청

당신은 Vpick 숏폼 성과 예측 Judge의 독립 검증자입니다. 이 패키지는 이전 v13의
문제를 지적받은 뒤 만든 v14 **개발 후보**입니다. 좋은 수치를 찾아 칭찬하는 것이
아니라, 코드·OOF·튜닝 로그를 근거로 이 결과가 무엇을 입증하고 무엇을 입증하지
못하는지 엄격하게 판정해 주십시오.

## 프로젝트 목표

새로운 익명 숏폼 후보 하나가 들어왔을 때 채널명·조회수·좋아요·성과 라벨을 보지
않고, Vpick 설명·자막·전후 문맥만으로 같은 채널 안에서의 상대 성과 잠재력을
0~100으로 출력하는 Judge를 만드는 것이 목표입니다.

조회수는 채점 후 검증에만 사용합니다. 채널별 조회수 백분위를 정답 신호로 사용하며,
중간 구간까지 실제 순서를 맞히는지가 핵심입니다.

## 배경

v13은 전체 채널 중심 Spearman `0.3125`였지만:

- mid-only 채널 중심 Spearman `-0.2084`
- 같은 라벨 내부 pairwise 약 우연 수준
- local pairwise `0.5877`, CI가 0.5 포함

즉 POS/NEG 극단 구분이 전체 지표를 올렸고 중간 구간은 실패했습니다.

v14는 다음을 변경했습니다.

- mid-mid pair 3배
- local pair 2배
- neg-pos 쉬운 pair 0.25배
- 채널별 pair 총가중치 균형화
- 자막 형식 프록시 수치 특징 제거
- 정규화한 설명·자막·전후 문맥만 사용
- outer longform GroupKFold 안에서 구조와 C를 inner CV로 선택
- 10개 seed 반복 OOF 평균

## 검토할 파일

1. `README.md`
2. `data/oof_predictions_PRIVATE.csv`
3. `data/nested_tuning_log_PRIVATE.json`
4. `results/v14_summary_PUBLIC.json`
5. `results/v14_model_comparison_PUBLIC.csv`
6. `results/weighting_ablation_summary.json`
7. `results/local_preserving_summary_PUBLIC.json`
8. `results/deployment_artifact_METADATA.json`
9. 저장소의 다음 코드
   - `src/train_performance_calibrator_v14_dev.py`
   - `src/fit_shortform_success_judge_v14_dev.py`
   - `src/predict_shortform_success_v14_dev.py`
   - `src/evaluate_shortform_success_holdout_v14.py`
   - `config/performance_calibrator_v14_dev.json`
   - `config/performance_judge_validation_v14.json`

## 반드시 독립 재계산할 항목

1. 94개 candidate ID와 OOF 누락·중복 여부
2. 같은 `longform_id`가 outer train/test에 동시에 들어가는 누수가 없는지
3. 다음 OOF 지표
   - 전체/채널 중심 Spearman
   - mid pooled 및 mid 채널 중심 Spearman
   - mid-only pairwise
   - 같은 채널 local pairwise
   - 같은 성과 라벨 내부 pairwise
   - POS-vs-NEG AUC
4. longform 군집 bootstrap CI
5. seed별 결과 분산과 특정 seed 의존성
6. outer test를 보고 spec/C를 선택한 흔적이 없는지
7. 프록시 특징을 숫자 열뿐 아니라 텍스트 경로에서도 실제 제거했는지
8. artifact 추론 입력에 채널·성과·출처 누출이 가능한 우회 경로가 없는지
9. nested OOF 절차와 최종 fixed artifact 사이의 불일치가 어느 정도인지

## 집중적으로 반박해 볼 주장

```text
v14는 v13보다 중간 성과 구간을 더 잘 순위화한다.
```

이 주장이 다음 이유로 거짓일 가능성을 점검하십시오.

- 34개 mid에 대한 선택 과적합
- 10-seed 평균화가 만든 비배포성 이득
- 특정 채널 또는 자막 생성 방식에 의존
- 텍스트 길이·문체·ASR 품질을 성과로 오인
- fold별 구조 선택 불안정
- 가중치 ablation을 같은 데이터에서 보고 고른 사후 선택

## 원하는 출력 형식

### 1. 최종 판정

다음 중 하나:

- `REJECTED`
- `DEVELOPMENT_CANDIDATE_ONLY`
- `READY_FOR_FRESH_HOLDOUT`
- `VALIDATED`

현재 fresh holdout이 없으므로 `VALIDATED`를 선택하려면 그 판단이 왜 가능한지
엄격한 근거를 제시해야 합니다.

### 2. 발견 사항

심각도 순서로 작성:

- `[CRITICAL]`
- `[HIGH]`
- `[MEDIUM]`
- `[LOW]`

각 항목에 파일·필드·재계산 수치를 포함하십시오.

### 3. 재계산 표

보고값과 독립 재계산값을 나란히 표시하십시오.

### 4. 다음 한 번의 실험

사람 평가나 GPU 없이 지금 데이터와 코드로 할 수 있는 실험 중, 불확실성을 가장
많이 줄이는 실험 하나만 선택해 구체적 명령·통과 기준과 함께 제안하십시오.

### 5. holdout 전 동결 목록

모델 입력, 텍스트 정규화, pair 생성, 가중치, 구조, C, seed 처리, 점수 변환,
검증 지표, 게이트를 명시하십시오.

## 금지

- POS/NEG AUC만 보고 연속 성과 Judge를 승인하지 마십시오.
- pooled 상관만으로 결론 내리지 마십시오.
- 현재 94건에서 더 좋아 보이는 가중치를 새로 찾고 그것을 검증값처럼 보고하지
  마십시오.
- artifact의 학습 데이터 점수를 일반화 성능으로 해석하지 마십시오.

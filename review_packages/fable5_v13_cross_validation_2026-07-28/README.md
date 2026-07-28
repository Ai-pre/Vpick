# Fable5 Cross-Validation Package: Shortform Success Judge v13

## 목적

이 패키지는 현재 동결한 `Shortform Success Judge v13`을 팀원 또는 외부
모델이 독립적으로 검토할 수 있도록 만든 재현 자료다.

검토 대상은 다음 질문이다.

> Vpick 설명·자막·앞뒤 문맥과 Codex 품질 특징만으로, 신규 숏폼의 같은
> 채널 내 상대 성과 순위를 유의미하게 예측할 수 있는가?

출력 `shortform_success_potential_0_100`은 예상 조회수가 아니다. 비슷한
게시 조건을 가정한 콘텐츠 기반 상대 성과 잠재력 백분위다.

## 현재 결론

- 내부 개발 상태: `development_frozen_pending_holdout`
- 내부 출처 비의존 게이트 5개: 통과
- 최종 외부 검증: 아직 통과하지 않음
- 신규 미공개 holdout 전 배포 가능 주장: 금지

같은 94개 데이터가 모델 구조 선택과 개발 평가에 반복 사용됐다. 따라서
이 패키지는 **개발 결과의 재현·오류 탐지용**이지 새로운 미공개
holdout이 아니다.

## 데이터 구성

- 후보 94개
- 롱폼 85개
- 채널 6개
- 성과 라벨: `pos 30`, `mid 34`, `neg 30`
- 자막 출처: `Vpick 47`, `yt-dlp fallback 47`
- 과거 역할 표기: `dev 19`, `locked_test 75`

과거 `locked_test` 75개도 이미 분석과 구조 선택 과정에서 확인됐으므로
현재는 더 이상 미공개 테스트셋으로 취급하면 안 된다.

### 채널별 후보 수

| 채널 | 후보 수 |
|---|---:|
| 숏박스 | 21 |
| 워크맨 | 21 |
| OOTB | 17 |
| 안원잘부 | 15 |
| BDNS | 10 |
| 피식대학 | 10 |

## 디렉터리

```text
data/reproduction/
  candidates_blind_94.jsonl
  codex_direct_v10_dimensions.csv
  codex_direct_v10_judgments_94.jsonl
  codex_direct_v10_scores_94.csv
  validation_targets_94_PRIVATE.csv
  group_split_94_PRIVATE.csv
  performance_controls_94.json
  preparation_summary.json
  ...

data/diagnostics/
  oof_predictions_PRIVATE.csv
  tuning_log_PRIVATE.json

reference_results/
  v11~v13 공개 요약과 v13 모델 비교
```

`PRIVATE`는 정답 누설 방지를 위한 역사적 파일명이다. 이 리뷰 패키지는
정답을 포함하므로 모델의 블라인드 채점 입력으로 통째로 넘기면 안 된다.

학습된 `joblib` artifact는 Git에 넣지 않았다. Pickle 계열 artifact는
신뢰하지 않는 저장소에서 역직렬화하면 위험하고 scikit-learn 버전에도
민감하기 때문이다. 제공된 코드와 재현 데이터로 다시 생성하며, 구조와
선택된 C 값은 `reference_results/deployment_artifact_METADATA.json`에서
확인할 수 있다.

## 모델 입력과 정답 분리

실제 모델 입력:

- 후보 설명
- 후보 자막
- 후보 직전·직후 문맥
- 구간 길이
- Codex 7개 품질 특징

금지 입력:

- 채널명
- 조회수·좋아요 수
- POS/NEG/MID
- 채널 내 성과 백분위
- 영상 URL과 ID
- 데이터 역할
- 자막 출처

성과 정답과 채널명은 학습 쌍 구성 및 사후 검증에만 사용한다.

## v13 구조

```text
익명 설명·자막·앞뒤 문맥
+ Codex 7개 특징
+ 구간 구조 특징
-> Pairwise Logistic Regression 3개
-> 동일 가중치 평균
-> 상대 성과 잠재력
```

앙상블 멤버:

1. `numeric_050`: 수치 특징 비중 0.50
2. `channel_balanced`: 채널별 학습 쌍 총 가중치 균등
3. `numeric_025`: 수치 특징 비중 0.25

## 보고된 내부 결과

| 지표 | v13 |
|---|---:|
| 채널 중심 Spearman | 0.3125 |
| 채널 Macro Spearman | 0.3203 |
| 같은 채널 Pairwise | 0.6146 |
| Local Pairwise | 0.5877 |
| Top-quintile precision | 0.4762 |
| 채널 중심 Spearman 95% CI | [0.0661, 0.5050] |
| Leave-One-Channel-Out 채널 중심 Spearman | 0.2969 |

## 추가 독립 진단에서 발견된 제한

전체 지표만 보면 내부 게이트를 통과하지만, 같은 성과 버킷 내부의 세밀한
순위 변별은 아직 확인되지 않았다.

| 추가 지표 | v13 |
|---|---:|
| 같은 POS/MID/NEG 버킷 내부 Pairwise | 0.4892 |
| MID 34개 내부 Spearman | -0.0696 |

이는 현재 성능의 상당 부분이 `pos/mid/neg`에 해당하는 큰 성과 구간을
나누는 데서 나올 가능성을 뜻한다. 모델 입력에 라벨이 들어간다는 뜻은
아니지만, 데이터의 극단 분리로 인해 일반 지표가 실제보다 좋아 보일 수
있으므로 반드시 검토해야 한다.

`label_bucket_oracle_invalid_for_deployment`는 정답 라벨을 고의로 읽는
무효 기준선이다. 이 가짜 판정자가 매우 높은 일반 지표를 내는 것을 통해
단순 상관과 Pairwise만으로 Judge 신뢰성을 주장하면 안 된다는 점을
확인한다.

## 빠른 검증

패키지 무결성:

```bash
python review_packages/fable5_v13_cross_validation_2026-07-28/verify_package.py
```

저장된 OOF 예측 독립 재계산:

```bash
python review_packages/fable5_v13_cross_validation_2026-07-28/independent_oof_audit.py
```

두 스크립트는 표준 라이브러리만으로 실행된다.

## 전체 v13 재현

권장 환경:

```bash
python -m pip install -r \
  review_packages/fable5_v13_cross_validation_2026-07-28/requirements-review.txt
```

참조 결과는 Python `3.13.9`에서 생성됐다. 정확한 환경과 별도 Amazon CPU
재현 차이는 `REPRODUCIBILITY_NOTE.md`를 확인한다.

결과를 Git 작업 파일과 분리해 재현한다.

```bash
python src/train_performance_calibrator_v13.py \
  --private-dir review_packages/fable5_v13_cross_validation_2026-07-28/data/reproduction \
  --public-dir review_packages/fable5_v13_cross_validation_2026-07-28/workspace/reproduced_public \
  --private-output review_packages/fable5_v13_cross_validation_2026-07-28/workspace/reproduced_private \
  --report review_packages/fable5_v13_cross_validation_2026-07-28/workspace/reproduced_report.md
```

재현 결과는 `reference_results/summary_PUBLIC.json`과 비교한다.

## 반드시 검토할 위험

1. 같은 94개에서 앙상블 구조와 가중치를 선택한 선택 편향
2. 과거 `locked_test`가 실제 코드에서 완전히 격리되지 않은 문제
3. 반복 OOF 평균과 전체 데이터로 한 번 fit한 배포 artifact의 동등성
4. 텍스트 속 인물명·프로그램 고유어를 통한 암묵적 채널 식별 가능성
5. POS/MID/NEG 큰 구간 분리 외의 세부 성과 순위 변별 부족
6. 85개 롱폼 중 79개가 숏폼 후보 1개뿐인 군집 구조
7. 텍스트 입력만으로 표정·음성·편집·썸네일·업로드 효과를 볼 수 없는 한계
8. 신규 채널과 기존 채널에서 의미가 같은 `0~100` 점수인지 여부
9. Python·NumPy·pandas·joblib 버전에 따른 소폭의 비결정적 결과 차이

## 핵심 소스

- `config/performance_calibrator_v13.json`
- `src/train_performance_calibrator_v11.py`
- `src/train_performance_calibrator_v12.py`
- `src/run_performance_calibrator_v12_phase2.py`
- `src/train_performance_calibrator_v13.py`
- `src/fit_shortform_success_judge_v13.py`
- `src/predict_shortform_success_v13.py`
- `src/evaluate_shortform_success_holdout_v13.py`
- `docs/shortform_success_judge_v13_frozen.md`
- `docs/shortform_success_judge_v13_team_handoff.md`
- `reports/performance_calibrator_v13_2026-07-28.md`
- `review_packages/fable5_v13_cross_validation_2026-07-28/REPRODUCIBILITY_NOTE.md`

외부 검토자는 기존 보고서 수치를 정답처럼 받아들이지 말고, 코드와 제공
데이터에서 독립적으로 재계산해야 한다.

# Fable5 전달 프롬프트

아래 GitHub 저장소에 있는 Vpick 프로젝트의 `Shortform Success Judge v13`을
독립 검토해 주세요. 기존 결론을 요약하는 데 그치지 말고, 데이터·코드·OOF
예측을 직접 대조하여 방법론적으로 유효한지 반박 가능한 형태로 검증해
주세요.

```text
Repository: https://github.com/Ai-pre/Vpick
Branch: main
Review package:
review_packages/fable5_v13_cross_validation_2026-07-28
```

검토를 시작할 때 `git rev-parse HEAD`를 기록하고, 최종 답변에 사용한 commit
hash를 명시하세요.

## 0. 가장 중요한 태도

- 기존 README와 보고서의 수치를 정답으로 전제하지 마세요.
- 좋은 수치를 방어하려 하지 말고 잘못된 주장을 찾아내는 red-team 관점으로
  검토하세요.
- 코드 실행 결과와 문서 설명이 다르면 코드를 기준으로 판단하세요.
- 확실하지 않은 부분은 추측으로 메우지 말고 `확인됨 / 추정 / 미확인`으로
  분리하세요.
- 데이터 94개는 신규 holdout이 아니라 이미 구조 선택에 사용된 개발
  데이터입니다.
- 현재 상태를 최종 검증 완료나 상용 배포 가능으로 승격하지 마세요.
- 이번 검토에서 가중치나 모델을 다시 최적화해 성능을 높이지 마세요.
  먼저 현재 동결 모델의 사실성을 검증하는 것이 목적입니다.
- GPU, 외부 LLM API, YouTube 영상 다운로드는 사용하지 마세요. 제공된
  텍스트·CSV·JSON과 CPU 코드만으로 검토하세요.

## 1. 프로젝트 목표

목표는 신규 숏폼 후보에 대해 다음 값을 출력하는 Judge를 만드는 것입니다.

```text
shortform_success_potential_0_100
```

이 값은 예상 조회수가 아니라, 비슷한 게시 조건에서의 콘텐츠 기반 상대
성과 잠재력 백분위입니다.

현재 입력은 다음뿐입니다.

- Vpick 또는 fallback으로 준비한 후보 설명
- 후보 구간 자막
- 후보 직전·직후 문맥
- 구간 길이
- Codex가 독립 채점한 7개 콘텐츠 품질 특징

다음 정보는 모델 입력에서 금지됩니다.

- 채널명
- 조회수·좋아요 수
- POS/MID/NEG
- 채널 내 성과 백분위
- 영상 URL과 영상 ID
- 데이터 역할
- 자막 출처

단, 같은 채널 후보끼리 학습 쌍을 만들고 성과를 사후 검증하기 위해
채널명과 성과 백분위는 학습 파이프라인 내부에서 사용됩니다. 이것이
추론 입력 누설과 구분되어 올바르게 구현됐는지 확인해 주세요.

## 2. 먼저 읽을 파일

다음 순서대로 읽어 주세요.

1. `review_packages/fable5_v13_cross_validation_2026-07-28/README.md`
2. `docs/shortform_success_judge_v13_team_handoff.md`
3. `review_packages/fable5_v13_cross_validation_2026-07-28/REPRODUCIBILITY_NOTE.md`
4. `docs/shortform_success_judge_v13_frozen.md`
5. `config/performance_calibrator_v11.json`
6. `config/performance_calibrator_v12.json`
7. `config/performance_calibrator_v13.json`
8. `src/train_performance_calibrator_v11.py`
9. `src/train_performance_calibrator_v12.py`
10. `src/run_performance_calibrator_v12_phase2.py`
11. `src/train_performance_calibrator_v13.py`
12. `src/fit_shortform_success_judge_v13.py`
13. `src/predict_shortform_success_v13.py`
14. `src/evaluate_shortform_success_holdout_v13.py`
15. 패키지의 `data/reproduction/`, `data/diagnostics/`
16. 마지막으로 기존 `reports/`와 `reference_results/`

보고서를 먼저 읽고 그 설명에 맞춰 코드를 해석하는 순서를 피해주세요.

## 3. 데이터 사실관계

제공 데이터의 예상 상태는 다음과 같습니다.

- 후보 94개
- 롱폼 85개
- 채널 6개
- 채널별 후보 수:
  - 숏박스 21
  - 워크맨 21
  - OOTB 17
  - 안원잘부 15
  - BDNS 10
  - 피식대학 10
- 성과 라벨:
  - pos 30
  - mid 34
  - neg 30
- 자막 출처:
  - Vpick scene API 47
  - yt-dlp fallback 47
- 역사적 데이터 역할:
  - dev 19
  - locked_test 75

`locked_test` 75개는 이미 이전 분석과 구조 선택에서 사용됐으므로 현재
진짜 미공개 테스트셋이 아닙니다. 코드가 이 역할을 실제 분할에 사용하지
않았다는 멘토 피드백도 있었습니다. 이 문제를 정확히 확인해 주세요.

## 4. 1단계: 패키지 무결성 검사

먼저 다음 명령을 실행하세요.

```bash
python review_packages/fable5_v13_cross_validation_2026-07-28/verify_package.py
```

다음을 확인하세요.

- 모든 핵심 파일에 94개 후보가 있는가
- 모든 파일의 `candidate_id` 집합이 정확히 같은가
- 중복 후보 또는 중복 ID가 없는가
- 익명 후보 입력에 금지 컬럼이 없는가
- 설명·자막·앞뒤 문맥이 실제로 94개 모두 채워졌는가
- API 키, 비밀번호, 계정정보가 포함되지 않았는가
- OOF 파일의 정답과 검증 정답 파일이 일치하는가

검사 실패 시 이후 결과 계산을 중단하고 실패 원인을 먼저 보고하세요.

## 5. 2단계: 데이터 누설 감사

다음 경로를 따라 금지 정보가 특징 행렬에 들어가는지 추적하세요.

```text
load_bundle
-> compose_candidate_text
-> build_structure_features
-> prepare_fold
-> build_pair_matrix
-> fit_prepared
-> predict
```

특히 다음을 검사하세요.

1. `channel_name`, 성과 백분위, POS/MID/NEG, URL, 자막 출처가 TF-IDF 또는
   수치 특징에 포함되는가
2. TF-IDF vocabulary, imputer, scaler가 각 outer fold의 train 데이터에서만
   fit되는가
3. C 선택이 outer test를 보지 않고 inner GroupKFold 안에서만 수행되는가
4. 같은 `longform_id`가 train과 test에 동시에 들어가는 fold가 있는가
5. 여러 seed의 OOF 평균 과정에서 후보 자신의 정답 또는 예측이 학습으로
   되돌아가는가
6. 채널 균형 가중치가 train fold에서만 계산되는가
7. 채널명은 모델 입력이 아니더라도, 자막 속 프로그램명·출연자명·고정
   유행어가 암묵적인 채널 식별자로 작동할 가능성이 있는가
8. `source_salience`, `overview_support`, transcript 품질처럼 데이터 준비
   상태를 나타내는 특징이 성과 점수에 부당하게 섞이는가

각 항목을 `통과 / 실패 / 판단 보류`로 표로 정리하고 파일·함수·가능하면
라인 근거를 제시하세요.

## 6. 3단계: 전체 v13 재현

필요하면 다음 환경을 사용하세요.

```bash
python -m pip install -r \
  review_packages/fable5_v13_cross_validation_2026-07-28/requirements-review.txt
```

참조 결과의 Python 버전은 `3.13.9`입니다. Amazon에서 Python 3.10.20과
일부 다른 수치 라이브러리 버전으로 재실행했을 때 모든 게이트는
통과했지만 channel-centered Spearman이 `0.3125`에서 `0.3151`로
달라졌습니다. `REPRODUCIBILITY_NOTE.md`와
`reference_results/server_cpu_reproduction/`을 확인하고, 이 차이를
무시하지 마세요.

그다음 Git 파일을 덮어쓰지 않도록 별도 workspace에 재현하세요.

```bash
python src/train_performance_calibrator_v13.py \
  --private-dir review_packages/fable5_v13_cross_validation_2026-07-28/data/reproduction \
  --public-dir review_packages/fable5_v13_cross_validation_2026-07-28/workspace/reproduced_public \
  --private-output review_packages/fable5_v13_cross_validation_2026-07-28/workspace/reproduced_private \
  --report review_packages/fable5_v13_cross_validation_2026-07-28/workspace/reproduced_report.md
```

다음 기준값과 재현값을 소수점 넷째 자리까지 비교하세요.

| 지표 | 기준값 |
|---|---:|
| pooled Spearman | 0.2907 |
| channel-centered Spearman | 0.3125 |
| channel Macro Spearman | 0.3203 |
| same-channel Pairwise | 0.6146 |
| Local Pairwise | 0.5877 |
| Top-quintile precision | 0.4762 |
| bootstrap 95% CI lower | 0.0661 |
| LOCO channel-centered Spearman | 0.2969 |

재현되지 않으면 라이브러리 버전, seed, GroupKFold shuffle, tie 처리,
부동소수점 차이를 순서대로 확인하세요. 단순히 “환경 차이”라고 끝내지
말고 최초로 달라지는 중간 산출물을 찾으세요.

정확한 참조 환경에서도 byte-identical하게 재현되는지, 같은 환경에서
두 번 실행했을 때 동일한 결과가 나오는지 별도로 확인하세요. 필요하면
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`을 고정한
결과도 비교하세요.

## 7. 4단계: 저장된 OOF의 독립 평가

다음 스크립트는 모델 학습 코드의 평가 함수를 import하지 않고 핵심 지표를
다시 계산합니다.

```bash
python review_packages/fable5_v13_cross_validation_2026-07-28/independent_oof_audit.py
```

현재 추가로 관찰된 결과는 다음과 같습니다.

- 같은 POS/MID/NEG 버킷 내부 Pairwise: `0.4892`
- MID 34개 내부 Spearman: `-0.0696`

전체 Pairwise `0.6146`과 채널 중심 Spearman `0.3125`가 좋아 보이더라도,
같은 버킷 안에서는 무작위 수준이고 MID 내부 순위는 음수입니다.

이 결과가 다음 중 무엇을 의미하는지 판단하세요.

1. 모델이 실제 연속 성과를 예측한 것이 아니라 큰 성과 구간만 구분하는가
2. MID 표본 34개가 너무 작아 진단 변동성이 큰 것뿐인가
3. POS/MID/NEG가 채널 백분위로 만들어졌기 때문에 일반 지표가 구조적으로
   부풀려지는가
4. 현재 0~100 점수를 연속적인 성공 잠재력으로 해석해도 되는가

MID-only와 within-bucket 지표의 bootstrap 신뢰구간도 추가 계산해 주세요.
단, 그 결과를 보고 모델 가중치를 바꾸지 마세요.

## 8. 5단계: 가짜 판정자 sanity check

`independent_oof_audit.py`의
`label_bucket_oracle_invalid_for_deployment`는 POS/MID/NEG 정답을 고의로
읽고, 버킷 내부는 사실상 임의 순서로 두는 무효 모델입니다.

이 모델은 배포할 수 없지만 일반 상관·Pairwise·Top-quintile 지표가 매우
높게 나옵니다. 따라서 다음을 확인하세요.

- v13이 이 가짜 판정자보다 세밀한 순위 정보를 실제로 추가하는가
- 전체 지표 외에 within-bucket, MID-only, 인접 백분위 pair 정확도가
  반드시 필요한가
- 현재 내부 게이트 5개가 “극단 버킷 구분기”를 걸러낼 수 있는가
- 새로운 검증 게이트를 추가한다면 threshold를 현재 데이터에 사후
  최적화하지 않고 어떻게 정할 것인가

가짜 판정자의 높은 수치를 v13의 성과처럼 인용하지 마세요. 이것은 지표
취약성을 보여주는 음성 대조군입니다.

## 9. 6단계: 검증 설계 감사

다음 쟁점을 각각 판단하세요.

### A. 그룹 분할

- `longform_id` GroupKFold가 같은 원본 롱폼 누설은 막는가
- 85개 롱폼 중 79개가 후보 1개뿐이라 같은 롱폼 내 선택 능력을 충분히
  검증하지 못하는가
- 하나의 롱폼에서 여러 숏폼이 나오는 실제 서비스 상황을 평가하려면
  어떤 추가 데이터가 필요한가

### B. 채널 일반화

- 채널 중심 Spearman은 채널별 평균 차이를 제거하지만 암묵적인 채널
  vocabulary 누설까지 제거하지는 못하는가
- LOCO 0.2969가 신규 채널 일반화의 충분한 증거인가
- 6개 채널만으로 LOCO 평균을 안정적으로 해석할 수 있는가
- 기존 채널용 Judge와 신규 채널용 Judge를 별도 모드로 봐야 하는가

### C. 반복 OOF와 배포 artifact

- v13 OOF는 10개 seed의 서로 다른 fold 모델 예측을 평균한다
- 배포 artifact는 전체 94개로 각 앙상블 멤버를 한 번 fit한다
- 반복 cross-fitted ensemble의 OOF 성능이 단일 full-fit 배포 모델의
  성능을 과대평가할 가능성이 있는가
- 배포 동등성을 높이려면 실제 배포도 fold ensemble로 저장해야 하는가,
  아니면 신규 holdout에서 full-fit artifact만 검증하면 충분한가

### D. 모델 선택 편향

- v11/v12/v12 phase2에서 같은 94개로 여러 표현·가중치·앙상블을 비교했다
- v13은 그 결과로 선택한 세 모델을 다시 같은 94개에서 평가했다
- nested C 선택은 올바르더라도 모델 계열과 앙상블 선택까지 완전히 nested된
  것은 아닌가
- 현재 수치는 개발 성능으로만 보고 신규 holdout 전에는 어느 수준의
  표현까지 허용해야 하는가

### E. 목표 정의

- 채널 내 조회 백분위는 “콘텐츠 품질”과 “실제 성과” 중 무엇을 측정하는가
- 업로드 시점, 추천 노출, 제목·썸네일, 시청자 규모를 보지 않는 모델이
  실제 성공 가능성을 어느 범위까지 주장할 수 있는가
- 현재 명칭 `shortform_success_potential`이 과장이라면 더 정확한 이름을
  제안하세요

## 10. 7단계: 추가 기준선

현재 구조를 바꾸지 않고 다음 진단 기준선을 같은 fold와 지표로 비교하세요.

1. 난수 점수 1,000회 분포
2. 상수 점수
3. duration-only
4. Codex 7개 특징만 사용
5. 익명 텍스트 TF-IDF만 사용
6. 구조 특징만 사용
7. label permutation 1,000회
8. 채널별 target permutation
9. POS/MID/NEG 버킷만 알고 버킷 내부는 무작위인 무효 oracle
10. v13 세 멤버 각각과 동일 가중치 앙상블

가능하면 다음을 보고하세요.

- observed metric
- null mean
- null 95% 구간
- empirical p-value
- 효과 크기
- 후보 수와 pair 수

표본이 작으므로 p-value 하나만으로 결론 내리지 마세요.

## 11. 최종 판정 기준

아래 중 하나로 명확히 판정하세요.

### A. 코드 또는 데이터 누설로 무효

성과 정답이나 대리 변수가 test fold 특징 생성에 들어가거나, 같은 롱폼이
train/test에 겹치거나, OOF가 실제 out-of-fold가 아니면 이 판정을 내립니다.

### B. 개발 후보로는 유효하지만 최종 Judge로는 미검증

코드 누설은 없고 내부 성능은 재현되지만, 모델 선택 편향·세부 순위 변별
부족·fresh holdout 부재가 남으면 이 판정을 내립니다.

### C. 신규 holdout에 바로 잠금 검증할 수 있는 동결 후보

코드와 데이터가 재현되고, 현재 구조를 더 수정할 필요 없이 fresh holdout에
한 번만 적용할 준비가 됐다고 판단될 때 선택합니다. 이는 최종 성능 검증
통과를 의미하지 않습니다.

현재 자료만으로 `상용 배포 검증 완료` 판정을 내리면 안 됩니다.

## 12. 최종 답변 형식

다음 순서로 작성해 주세요.

1. **한 문장 판정**
2. **치명적 문제**: 있으면 파일·함수 근거와 함께
3. **재현 결과 표**: 기존값, 재현값, 차이
4. **누설 감사 표**: 항목별 통과/실패/보류
5. **추가 sanity check 결과**
6. **within-bucket 및 MID-only 해석**
7. **반복 OOF와 배포 artifact 동등성 판단**
8. **현재 주장 가능한 것**
9. **현재 주장하면 안 되는 것**
10. **fresh holdout 전에 반드시 고칠 사항**
11. **fresh holdout에서 잠글 사항**
12. **우선순위가 있는 다음 작업 5개**

각 문제는 심각도를 `Critical / High / Medium / Low`로 표시하세요.
코드 수정이 필요하면 바로 수정하지 말고 먼저 수정안과 그 수정이 기존
검증을 무효화하는지 설명하세요.

마지막에는 다음 문장을 완성해 주세요.

> 현재 v13은 ________ 용도로는 사용할 수 있지만, ________ 근거가 없으므로
> ________이라고 주장할 수 없다.

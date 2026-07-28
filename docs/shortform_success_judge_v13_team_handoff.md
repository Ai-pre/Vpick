# Shortform Success Judge v13 - Team Handoff

## 1. 개발 목표

신규 숏폼 후보의 익명 콘텐츠 근거를 이용해
`shortform_success_potential_0_100`을 출력하는 Judge를 개발했다.

이 값은 조회수의 절대 예측값이 아니다. 게시 조건이 비슷하다고 가정했을 때
해당 후보가 보일 수 있는 콘텐츠 기반 상대 성과 잠재력 백분위다.

현재 상태는 `development_frozen_pending_holdout`이다. 기존 94개 개발
데이터에서 내부 검증 게이트는 통과했지만, 구조 선택에도 같은 데이터가
사용됐으므로 신규 미공개 holdout 검증 전에는 최종 검증 완료로 주장하지
않는다.

## 2. 전체 파이프라인

```text
Vpick 후보 설명 + 자막 + 앞뒤 문맥 + 구간 길이
-> Codex 7개 품질 특징
-> 익명 문자 TF-IDF와 수치 특징 결합
-> Pairwise Logistic Regression 3개
-> 동일 가중치 앙상블
-> 성공 잠재력 0~100 + 모델 불일치
```

Codex 7개 특징은 다음과 같다.

1. `self_contained_clarity`
2. `progression_payoff`
3. `boundary_integrity`
4. `opening_pull`
5. `change_or_surprise`
6. `emotional_or_information_gain`
7. `memorable_specificity`

채점 입력에서는 채널명, 조회수, 좋아요 수, POS/NEG, 성과 백분위, URL,
데이터 역할, 자막 출처를 모두 제외한다. 성과 정보는 모델 학습과 사후
검증에만 사용한다.

## 3. 고정 모델

세 Pairwise 모델은 같은 익명 텍스트와 Codex·구조 특징을 사용한다.

| 모델 | 차이 |
|---|---|
| `numeric_050` | Codex·구조 수치 특징 비중 0.50 |
| `channel_balanced` | 채널별 학습 쌍의 총 가중치를 균등화 |
| `numeric_025` | Codex·구조 수치 특징 비중 0.25 |

세 모델의 예측 순위를 각각 `1/3`씩 평균한다. Pairwise 학습은 POS/NEG
이진 분류가 아니라, 같은 채널 후보 사이에서 실제 채널 내 성과 순서를
학습하는 방식이다.

## 4. 기존 방식에서 바꾼 점

초기 nested 실험에서는 fold마다 모델 계열을 자동 선택했다. 데이터가
작아 fold별 선택 결과가 크게 달라졌고, 전체 OOF 채널 중심 Spearman은
`0.1048`에 그쳤다. 다만 같은 개발 데이터에서 가장 좋았던 고정 Pairwise
계열은 `0.2577`이었다.

최종 v13에서는 다음을 변경했다.

- 불안정한 자동 모델 계열 선택을 제거하고 Pairwise 계열로 고정
- Codex·구조 특징 비중이 다른 두 모델을 결합
- 특정 채널의 데이터 수가 학습을 지배하지 않도록 채널 균형 모델 추가
- 세 모델을 동일 가중치로 앙상블
- 10개 seed의 롱폼 단위 GroupKFold OOF를 평균
- 성과 누설 가능성이 있는 출처·채널·조회 정보 제거

따라서 실제 알고리즘 개선 비교는 고정 Pairwise `0.2577`에서 v13
`0.3125`로 보는 것이 타당하다. `0.1048`은 불안정했던 자동 선택
파이프라인 전체 결과이므로 직접적인 단일 모델 기준선으로 과장하면 안 된다.

## 5. 내부 검증

- 후보 94개
- 롱폼 85개
- 채널 6개
- 외부 5-fold, 내부 4-fold GroupKFold
- 롱폼 ID 기준 분할
- 10개 seed 반복 OOF 평균
- 2,000회 롱폼 bootstrap
- Leave-One-Channel-Out 진단

| 지표 | 결과 |
|---|---:|
| 채널 중심 Spearman | 0.3125 |
| 채널 Macro Spearman | 0.3203 |
| 같은 채널 Pairwise 정확도 | 0.6146 |
| Local Pairwise 정확도 | 0.5877 |
| Top-quintile precision | 0.4762 |
| NDCG | 0.8672 |
| 채널 중심 Spearman 95% CI | [0.0661, 0.5050] |
| LOCO 채널 중심 Spearman | 0.2969 |

설정한 출처 비의존 내부 게이트 5개는 모두 통과했다.

## 6. 해석 및 제한

- `0~100`은 예상 조회수가 아니라 94개 개발 참조 분포에 대한 상대 백분위다.
- 텍스트·구조 근거만 사용하므로 표정, 음성 톤, 화면 편집, 썸네일, 제목,
  업로드 시점 및 추천 알고리즘 효과는 직접 평가하지 못한다.
- `ensemble_disagreement_std`가 높으면 세 모델의 판단이 불안정하다는 뜻이다.
- 내부 게이트 통과는 개발 완료 신호일 뿐 최종 외부 타당도 검증은 아니다.
- 신규 holdout 결과를 본 뒤 구조나 가중치를 바꾸면 해당 데이터는 더 이상
  holdout이 아니다.

## 7. 다음 검증에서 잠글 항목

- Codex 7개 특징 정의
- 세 Pairwise 앙상블 멤버
- 멤버별 `1/3` 가중치
- Logistic Regression의 C 후보 범위
- 채널 내 성과 정규화 방식
- 내부 및 외부 검증 게이트

신규 holdout의 후보 점수를 먼저 저장하고, 이후 비공개 실제 성과 파일을
한 번만 결합해 평가해야 한다.

## 8. 주요 파일

- `docs/shortform_success_judge_v13_frozen.md`: 모델 사용 및 검증 원칙
- `config/performance_calibrator_v13.json`: 동결 설정
- `src/train_performance_calibrator_v13.py`: 반복 OOF 내부 검증
- `src/fit_shortform_success_judge_v13.py`: 전체 개발 데이터 artifact 생성
- `src/predict_shortform_success_v13.py`: 신규 후보 추론
- `src/evaluate_shortform_success_holdout_v13.py`: 미공개 holdout 평가
- `results/performance_calibrator_v13/summary_PUBLIC.json`: 공개 결과 요약
- `reports/performance_calibrator_v13_2026-07-28.md`: 실험 보고서

로컬 Git 기준 최신 반영 커밋은 `de93ff6`이다.

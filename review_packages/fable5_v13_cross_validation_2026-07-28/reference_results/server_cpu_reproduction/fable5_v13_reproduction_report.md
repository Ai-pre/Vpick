# Vpick 성과 보정기 v13 동결 후보 검증

## 1. 고정 구조

2차 개발 실험이 끝난 뒤 다음 세 Pairwise 모델을 동일 가중치로 고정했다.

1. Codex·구조 수치 특징 비중 0.50
2. 채널별 학습 쌍 총 가중치 균등
3. Codex·구조 수치 특징 비중 0.25

세 모델 모두 익명 설명·자막·경계 문맥의 문자 TF-IDF와 Codex 7개 특징을
사용한다. 채널명, 조회수, 좋아요, 성과 라벨·백분위, URL, 데이터 역할,
자막 출처는 입력에서 제외했다.

## 2. 검증

- 롱폼 ID 기반 외부 5-fold·내부 4-fold GroupKFold
- 10개 seed 반복 후 OOF 평균
- C는 각 외부 학습 fold 내부에서만 선택
- 2,000회 롱폼 bootstrap
- leave-one-channel-out 별도 진단

## 3. 결과

| 모델 | 채널 중심 rho | 채널 Macro rho | Pairwise | Local Pairwise | 선택 점수 |
|---|---|---|---|---|---|
| frozen_equal_weight_ensemble | 0.3151 | 0.3237 | 0.6161 | 0.5924 | 0.2857 |
| numeric_050 | 0.3049 | 0.3168 | 0.6085 | 0.5877 | 0.2759 |
| channel_balanced | 0.2975 | 0.3032 | 0.6009 | 0.5877 | 0.2665 |
| numeric_025 | 0.3084 | 0.3093 | 0.6085 | 0.5640 | 0.2679 |

- 채널 중심 Spearman 95% CI:
  [0.0685, 0.5050]
- Leave-one-channel-out 채널 중심 Spearman:
  0.2994

## 4. 내부 게이트

| 게이트 | 관측값 | 최소 기준 | 통과 |
|---|---|---|---|
| channel_centered_spearman | 0.3151 | 0.3000 | True |
| channel_macro_spearman | 0.3237 | 0.2000 | True |
| same_channel_pairwise_accuracy | 0.6161 | 0.5800 | True |
| same_channel_local_pairwise_accuracy | 0.5924 | 0.5500 | True |
| bootstrap_primary_ci_lower | 0.0685 | 0.0000 | True |

내부 게이트 통과: **True**

## 5. 판정

현재 상태는 `development_frozen_pending_holdout`이다. 내부 게이트를 모두
통과하더라도 같은 94개에서 구조를 선택했으므로 최종 검증 통과로 주장하지
않는다. 다음 신규 미공개 holdout에서는 구조·가중치·C 후보를 바꾸지 않고
한 번만 평가한다.

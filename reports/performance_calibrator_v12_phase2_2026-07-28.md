# Vpick 성과 보정기 v12 2차 집중 실험

## 목적

1차 ablation에서 가장 강했던 기존 Pairwise 문자 TF-IDF + 수치 특징 구조
주변의 단일 변경만 비교했다. 모든 실험은 동일한 롱폼 GroupKFold와 내부 C
선택을 사용한다.

## 결과

| 실험 | 채널 중심 rho | 채널 Macro rho | Pairwise | Local Pairwise | 선택 점수 |
|---|---|---|---|---|---|
| baseline_numeric_050 | 0.3029 | 0.3017 | 0.6206 | 0.6351 | 0.2884 |
| baseline_concat_char_word | 0.2740 | 0.2983 | 0.6282 | 0.6540 | 0.2838 |
| baseline_channel_balanced | 0.3014 | 0.3150 | 0.6206 | 0.6066 | 0.2832 |
| baseline_numeric_025 | 0.3174 | 0.3053 | 0.6206 | 0.5640 | 0.2739 |
| baseline_gap03 | 0.2915 | 0.2897 | 0.6115 | 0.6114 | 0.2704 |
| baseline_local_boost | 0.2907 | 0.2687 | 0.6131 | 0.6019 | 0.2614 |
| baseline_char_3_5 | 0.2832 | 0.2668 | 0.6085 | 0.6114 | 0.2593 |
| baseline_extended_c | 0.2817 | 0.2713 | 0.6100 | 0.6066 | 0.2591 |
| baseline_reliability | 0.2800 | 0.2685 | 0.6055 | 0.5924 | 0.2519 |
| baseline_char_2_6 | 0.2561 | 0.2550 | 0.6100 | 0.6303 | 0.2510 |
| baseline_gap08 | 0.2779 | 0.2688 | 0.6009 | 0.5877 | 0.2484 |
| baseline_numeric_200 | 0.2341 | 0.2545 | 0.6009 | 0.6114 | 0.2337 |
| baseline_train_zscore | 0.2036 | 0.2078 | 0.5797 | 0.6019 | 0.1982 |

사후 최고 개발 실험은 `baseline_numeric_050`이다.

- 채널 중심 Spearman: 0.3029
- 채널 Macro Spearman: 0.3017
- Pairwise 정확도: 0.6206
- Local Pairwise 정확도:
  0.6351
- 채널 중심 Spearman 2,000회 bootstrap 95% CI:
  [0.0603, 0.4909]

## 해석

이 표는 개선 방향을 고르기 위한 개발 결과다. 같은 94개에서 최고안을 선택했기
때문에 최종 검증값으로 사용하지 않는다. 선택한 구조를 고정한 뒤 새 미공개
holdout에서 한 번만 평가해야 한다.

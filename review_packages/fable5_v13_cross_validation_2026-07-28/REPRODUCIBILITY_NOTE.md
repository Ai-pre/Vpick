# Reproducibility Note

## 참조 결과 생성 환경

Git의 기존 v13 결과는 다음 환경에서 생성됐다.

```text
Python 3.13.9
NumPy 2.2.4
pandas 2.2.3
SciPy 1.15.2
scikit-learn 1.6.1
joblib 1.4.2
```

`requirements-review.txt`는 Python을 제외한 이 버전을 고정한다.

## Amazon CPU 재현 환경

별도 Amazon CPU 환경에서도 같은 코드와 데이터를 다시 실행했다.

```text
Python 3.10.20
NumPy 2.2.6
pandas 2.3.3
SciPy 1.15.2
scikit-learn 1.6.1
joblib 1.5.3
```

모든 내부 게이트는 다시 통과했지만 일부 지표는 소폭 달라졌다.

| 지표 | 기존 참조 | Amazon 재현 | 차이 |
|---|---:|---:|---:|
| pooled Spearman | 0.290666 | 0.291150 | +0.000484 |
| channel-centered Spearman | 0.312474 | 0.315075 | +0.002601 |
| channel Macro Spearman | 0.320330 | 0.323666 | +0.003336 |
| same-channel Pairwise | 0.614568 | 0.616085 | +0.001517 |
| Local Pairwise | 0.587678 | 0.592417 | +0.004739 |
| Top-quintile precision | 0.476190 | 0.476190 | 0 |
| channel Macro NDCG | 0.867181 | 0.867665 | +0.000483 |
| LOCO channel-centered Spearman | 0.296918 | 0.299353 | +0.002435 |

Amazon 재현 산출물은
`reference_results/server_cpu_reproduction/`에 보존했다.

## 해석

- 코드·데이터·seed가 같아도 Python 및 수치 라이브러리 환경에 따라
  fold 모델의 예측 순위가 소폭 달라질 수 있다.
- 게이트 통과 여부는 바뀌지 않았지만 byte-identical 재현은 실패했다.
- 외부 검토자는 먼저 참조 환경을 맞춰야 하며, 그래도 차이가 나면
  solver·BLAS·thread 설정까지 확인해야 한다.
- 최종 제출에서는 소수점 넷째 자리의 단일 수치를 절대값처럼 과장하지
  않고, 재현 범위와 bootstrap 구간을 함께 보고하는 것이 타당하다.

# Performance Calibrator v14 개발 진단

## 결론

이 결과는 새 홀드아웃 검증이 아니라 기존 94건 개발 데이터의 nested OOF 진단이다.
모델 구조와 pair 가중치는 v13 실패를 본 뒤 설계했으므로 최종 성능 주장에 사용할 수
없다.

| 모델 | 전체 채널 중심 rho | mid 채널 중심 rho | mid pairwise | local pairwise | 극단 AUC |
|---|---:|---:|---:|---:|---:|
| v14_nested_procedure | 0.3355 | 0.2495 | 0.6000 | 0.5213 | 0.7367 |
| v13_repeated_grouped_oof | 0.3125 | -0.2084 | 0.4333 | 0.5877 | 0.7144 |
| duration_only | -0.0476 | -0.2096 | 0.4000 | 0.4692 | 0.4633 |
| label_bucket_oracle_invalid | 0.9239 | 0.0465 | 0.5500 | 0.8104 | 1.0000 |

## v14 핵심 수치

- mid 채널 중심 Spearman: 0.2495
- mid pairwise: 0.6000
- local pairwise: 0.5213
- POS-vs-NEG AUC: 0.7367
- mid rho 95% CI:
  [-0.1912,
   0.6148]

## 설계

- 롱폼 단위 outer 5-fold / inner 4-fold
- outer fold 안에서 표현 구조와 C를 함께 선택
- 모든 자막 형식 프록시 수치 특징 제외
- 정규화된 자막·설명만 사용
- mid-mid pair 3배, local pair 2배, pos-neg 쉬운 pair 0.25배
- 채널별 pair 총가중치 균형화
- 10개 seed 반복 OOF 평균

## 상태

`development_only_not_validated`. 새 mid-enriched holdout에서 실제 배포 artifact를
검증하기 전에는 성과 예측 Judge로 승인하지 않는다.

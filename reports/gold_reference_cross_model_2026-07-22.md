# Gold Reference Judge v6 크로스 모델 비교

## 실험 조건

- 평가 대상: 동일한 블라인드 Gold 후보 45개
- 입력: 장면 설명, 후보 구간 transcript, before_context, after_context
- 공통 기준: `shortform_reference_judge_v6_ko`
- 공통 점수: 현저성 1~5 + 8개 Boolean 체크리스트의 동일 가중 평균
- 인간 검증: 15개 후보, 평가자 2명 완료 상태의 예비 결과
- GPT는 2회 반복 평균, Gemini와 Claude는 현재 1회 결과
- Claude 모델명은 사용자가 확인한 `Claude Fable`이며 세부 API 모델 ID는 기록되지 않음

## 모델별 결과

| 모델 | 채점 커버리지 | 평균 점수 | Suitable 비율 | 인간 점수 Spearman | 인간 Suitable 정확도 | 인간 Suitable F1 | 반복 Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.6 Terra | 45/45 | 52.2917 | 0.5111 | 0.0697 | 0.7778 | 0.8571 | 0.8722 |
| Gemini 3.6 Flash High | 44/45 | 55.9659 | 0.4773 | 0.2982 | 0.6667 | 0.7692 | 미측정 |
| Claude Fable | 45/45 | 59.4444 | 0.4000 | 0.6466 | 0.8889 | 0.9231 | 미측정 |

## 모델 간 일치도

| 모델 조합 | 공통 후보 | Reference Spearman | Suitable 일치율 | 평균 절대 점수 차이 |
|---|---:|---:|---:|---:|
| GPT - Gemini | 44 | 0.6077 | 0.6818 | 22.3722 |
| GPT - Claude | 45 | 0.6551 | 0.7556 | 14.0972 |
| Gemini - Claude | 44 | 0.6494 | 0.7955 | 16.7614 |

## 해석

1. 현재 2인 인간평가 기준으로는 Claude Fable이 가장 높은 정렬을 보였다. Reference 점수 Spearman은 0.6466이고 Suitable 정확도는 0.8889다.
2. GPT는 2회 반복 Spearman 0.8722로 자체 일관성이 가장 잘 검증됐지만, 인간의 연속 점수와는 거의 정렬되지 않았다.
3. Gemini 3.6 High는 GPT보다 인간 연속 점수 상관이 높지만 Suitable 정확도는 더 낮다. ASR이 심하게 손상된 후보 1개는 abstain했다.
4. 세 모델의 Pos/Neg AUC는 각각 0.4731, 0.5037, 0.4892로 무작위 수준이다. 현재 기준은 숏폼의 편집·완결 품질을 평가하지만 조회수 성과를 직접 설명하는 기준은 아니다.
5. 아직 최종 Judge를 확정할 수 없다. 인간평가 3번째 평가자와 Claude·Gemini 반복 평가가 완료되어야 신뢰도 게이트를 최종 판정할 수 있다.

## 현재 결론

- 예비 주 Judge 후보: Claude Fable
- 안정성 기준 모델: GPT-5.6 Terra
- 보조 크로스체크 모델: Gemini 3.6 Flash High
- 다음 검증: 인간 H3 완료, Claude 2회차, Gemini 2회차, 경계·payoff 불일치 항목 재검토

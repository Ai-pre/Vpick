# Gold Pairwise Judge v5 멀티모달 실험 보고

## 실험 설정

- 같은 채널의 Pos-Neg Gold 18쌍을 익명 쌍대비교했다.
- Gemini 3.1 Flash-Lite가 공개 YouTube 원본의 지정 구간을 화면과 음성으로 직접 확인했다.
- 한 요청에 비교 4개, 즉 영상 구간 최대 8개를 입력했고 10번의 실제 API 요청으로 36개 판정을 얻었다.
- 각 비교를 2회 평가했으며 2회차는 LEFT/RIGHT 위치를 반대로 제시한 뒤 원래 후보 방향으로 복원했다.
- 모델 입력에서 Pos/Neg, 조회수, 좋아요, 성과 백분위, Gold pair ID를 제외했다.

## 결과

| 지표 | 결과 |
|---|---:|
| 평가 커버리지 | 1.0000 |
| 편집 판단 반복 일치도 | 0.5000 |
| 성과 판단 반복 일치도 | 0.6111 |
| Pos 엄격 선호 정확도 | 0.3889 |
| 결정된 비교의 Pos 정확도 | 0.6364 |
| 평균 신뢰도 | 4.6667 / 5 |
| 반복 신뢰도 게이트 | 실패 |
| 인간 일치도 게이트 | 평가 대기 |

채널별 Pos 엄격 선호 정확도는 숏박스 0.3333, Pilot 채널 0.0000, 워크맨 0.6250이었다. Gemini는 18쌍 중 Pos 7쌍, Neg 4쌍을 선택했고 7쌍은 반복 판단이 달라 불안정으로 분류됐다.

## 진단

- 좌우 반전 후 편집 판단 9쌍과 성과 판단 7쌍이 달라졌다.
- 같은 화면 위치만 두 번 선택한 위치 편향은 편집 4쌍, 성과 5쌍에서 확인됐다.
- 편집 세부 점수의 83.3%가 4점 이상이었고 좌우 평균 총점도 84점대로 점수 포화가 컸다.
- 평균 신뢰도 4.67점과 낮은 반복 일치도가 함께 나타나 모델의 자기 신뢰도를 품질 지표로 사용할 수 없었다.
- 실제 영상 입력은 transcript 누락 문제를 해결했지만 Flash-Lite의 판단 안정성까지 보장하지는 않았다.

## 결론

Gemini 멀티모달 결과는 영상·음성 근거를 제공하는 보조 진단으로는 유용하지만 현재 상태로 주 Judge가 될 수 없다. 자동 모델 중 반복 신뢰도 기준 0.80을 통과한 것은 Terra뿐이다. 다만 Terra도 인간 일치도 검증이 끝나지 않았으므로 최종 Judge로 확정하지 않는다. 다음 필수 단계는 `human_pairwise_labels.csv`의 18쌍 x 3명, 총 54개 블라인드 인간 평가를 완료하고 모델별 인간 다수결 일치도와 Fleiss' kappa를 계산하는 것이다.

## 결과 위치

```text
results/gold_pairwise_judge_v5_multimodal_batch/scores/pairwise_judge_scores.csv
results/gold_pairwise_judge_v5_multimodal_batch/scores/pairwise_judge_usage.csv
results/gold_pairwise_judge_v5_multimodal_batch/validation/pairwise_validation_summary.json
results/gold_pairwise_judge_v5_multimodal_batch/validation/pairwise_validation_report.md
```

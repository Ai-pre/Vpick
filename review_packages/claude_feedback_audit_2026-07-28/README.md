# Claude 피드백 독립 검증 및 수정 기록

## 결론

Claude의 핵심 비판은 대체로 맞다. 다만 실험 3의 `Vpick Core@5`가 틀렸다는
주장은 재계산 결과 사실이 아니었다.

- **실험 3**: Vpick 예측 중복은 실제로 존재했지만, 제거 후에도
  `Core@5 = 0.1739`로 동일했다. Ours의 `Core@5 = 0.4783` 결론도 유지된다.
- **성과 예측 Judge v13**: 전체 채널 중심 Spearman `0.3125`는 재현되지만,
  중간 성과 34건에서는 채널 중심 Spearman이 `-0.2084`다. 따라서 v13은
  연속 성과 순위 Judge로 승인할 수 없다.
- 현재 데이터가 지지하는 범위는 상위/하위 극단 판별
  `POS-vs-NEG AUC = 0.7144`까지다.

## 1. 실험 3 중복 재계산

동일한 `pair_id`, 실행 설정 안에서 `(pred_start_sec, pred_end_sec)`가 같은 행은
동일 후보로 간주했다. 원래 순위를 유지하면서 첫 행만 남기고, 6위 이후의 고유
후보를 당겨 Top-5를 다시 채웠다.

| 항목 | 기존 | 중복 제거·보충 | 변화 |
|---|---:|---:|---:|
| Vpick Top-1 Core | 0.0435 | 0.0435 | 0 |
| Vpick Core@3 | 0.1304 | 0.1304 | 0 |
| Vpick Best IoU@3 | 0.07650 | 0.07805 | +0.00155 |
| **Vpick Core@5** | **0.1739** | **0.1739** | **0** |
| Vpick Best IoU@5 | 0.12715 | 0.12715 | 0 |
| Ours Core@5 | 0.4783 | 0.4783 | 비교값 |

세부 감사 결과:

- 입력 129행 중 정확히 같은 구간 10행 제거
- Top-5 안에 중복이 있던 쌍: `G002`, `G007`
- 기존 평가에서 고유 후보가 5개 미만이던 쌍: 5/23
- 후순위 고유 후보로 보충한 뒤에도 5개 미만인 쌍: 3/23
- 남은 3건은 중복 문제가 아니라 Vpick이 애초에 고유 후보를 4개만 생성한 경우

따라서 보고 문구는 다음처럼 고친다.

> Vpick 후보 목록에 중복 구간이 있어 평가 전 중복 제거가 필요했다. 중복 제거와
> 후순위 후보 보충 후 핵심 지표를 재계산했으며, Vpick Core@5 0.1739와 Ours
> Core@5 0.4783은 변하지 않았다.

## 2. v13 타당성 재검증

동일한 94건 OOF 예측을 사용했다.

| 검증 뷰 | 결과 | 해석 |
|---|---:|---|
| 전체 채널 중심 Spearman | +0.3125 | 양극단 영향을 포함한 개발 수치 |
| POS+NEG 채널 중심 Spearman | +0.3919 | 극단 구분 신호 존재 |
| POS-vs-NEG AUC | 0.7144 | 극단 판별은 우연보다 높음 |
| mid 원시 pooled Spearman | -0.0696 | 중간 구간 순위화 실패 |
| **mid 채널 중심 Spearman** | **-0.2084** | 채널 정규화 후에도 실패 |
| 같은 채널 local pairwise | 0.5877 | 95% CI가 0.5를 포함 |
| 같은 라벨 내부 pairwise | 0.4892 | 극단 버킷 내부 변별은 우연 수준 |

`-0.0696`과 `-0.2084`는 충돌하는 값이 아니다. 전자는 mid 34건을 그대로
계산한 pooled 상관이고, 후자는 채널 평균을 제거한 뒤 계산한 프로젝트의 주
정규화 지표다. 성과 정규화 목표에는 후자를 우선 보고한다.

## 3. Claude 지적 중 정정할 부분

### “Vpick Core@5가 틀렸다”

중복 오염과 기회 수 불균형은 확인됐지만 `Core@5` 수치는 재계산 전후 동일했다.
설계 결함은 수정해야 하나, 이미 보고한 핵심 수치를 정정할 필요는 없다.

### “10 seed 평균화가 동결 목록에서 빠졌다”

10개 seed와 반복 OOF 평균은 이미
`config/performance_calibrator_v13.json` 및
`src/train_performance_calibrator_v13.py`에 고정되어 있다.

실제 문제는 별개다. 개발 수치 `0.3125`는 반복 OOF 예측 평균이고, 현재 배포
artifact는 전체 데이터에 한 번 적합한 3개 모델이다. 두 절차의 동등성이 검증되지
않았다. 다음 검증에서는 **실제로 배포할 artifact가 만든 예측**만 홀드아웃에
사용해야 한다.

### “앙상블이 사실상 약하다”

맞다. 세 멤버는 수치 특징 비중과 채널 균형 여부만 다르며 표현 계열이 같다.
결합 이득도 약 `+0.004`이므로 강한 모델 다양성의 증거로 주장하지 않는다.

## 4. 이번 수정

1. `src/evaluate_predictions.py`
   - 동일 시간 구간 자동 중복 제거
   - 원래 순위를 유지하고 후순위 고유 후보로 Top-k 보충
   - 중복 제거 수와 Top-5 미충족 그룹을 감사 CSV/JSON에 기록
   - 과거 수치 재현이 필요할 때만 `--keep-duplicate-intervals` 사용
2. `independent_oof_audit.py`
   - mid pooled와 mid 채널 중심 Spearman을 분리
   - POS+NEG 채널 중심 Spearman과 AUC 추가
3. `performance_judge_validation_v14.json`
   - pooled 수치가 아닌 mid와 local pairwise를 주 게이트로 승격
   - 점추정치와 군집 bootstrap CI를 동시에 요구
   - 다음 모델에서 자막 형식 프록시 특징 제외
   - OOF 평균이 아니라 실제 배포 artifact의 홀드아웃 예측을 검증하도록 고정
4. `evaluate_shortform_success_holdout_v14.py`
   - fresh holdout, 최소 표본 수, mid 성능, local pairwise, 극단 AUC를 함께 판정

## 5. 현재 승인 상태

| 구성요소 | 상태 |
|---|---|
| 고정 루브릭 기반 편집·콘텐츠 품질 점수 | 사용 가능 |
| v13 상·하위 극단 판별기 | 연구용 진단 가능 |
| v13 연속 성과 백분위 Judge | **승인 불가** |
| 실험 3 Ours-vs-Vpick 비교 | 중복 제거 후에도 방향 유지 |
| v14 검증 규약 | 구현 완료, 새 데이터·새 모델 검증 대기 |

현재 v13 OOF를 v14 규약에 진단용으로 넣은 결과도 불합격이다.

| v14 게이트 | 관측 | 95% CI 하한 | 판정 |
|---|---:|---:|---|
| mid 채널 중심 Spearman | -0.2109 | -0.5942 | 실패 |
| 같은 채널 local pairwise | 0.5877 | 0.4419 | 실패 |
| POS-vs-NEG AUC | 0.7144 | 0.5657 | 통과 |
| 전체/중간 표본 수 | 94 / 34 | 요구 180 / 150 | 실패 |

이 결과는 새 홀드아웃 검증이 아니라 기존 개발 OOF를 더 엄격한 규약으로 다시 본
진단이다. 최종 승인은 새 데이터에서 실제 배포 artifact를 한 번만 평가해 결정한다.

## 재현

```bash
python src/evaluate_predictions.py \
  --dataset review_packages/claude_feedback_audit_2026-07-28/data/comparable_dataset.csv \
  --predictions review_packages/claude_feedback_audit_2026-07-28/data/vpick_predictions_original.csv \
  --out-dir review_packages/claude_feedback_audit_2026-07-28/results/vpick_deduplicated

python review_packages/fable5_v13_cross_validation_2026-07-28/independent_oof_audit.py \
  --output review_packages/fable5_v13_cross_validation_2026-07-28/reference_results/independent_oof_audit_corrected.json

python review_packages/claude_feedback_audit_2026-07-28/audit.py

python review_packages/claude_feedback_audit_2026-07-28/run_v14_on_current_v13_oof.py
```

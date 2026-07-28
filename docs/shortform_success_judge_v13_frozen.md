# Shortform Success Judge v13

## 역할

신규 숏폼 후보의 익명 콘텐츠 근거를 받아
`shortform_success_potential_0_100`을 출력한다. 이 값은 실제 조회수 예측값이
아니라, 게시 조건이 비슷하다고 가정한 콘텐츠 기반 상대 성과 잠재력이다.

```text
Vpick 후보 설명·자막·앞뒤 문맥
-> Codex 7개 품질 특징
-> 고정 Pairwise 성과 보정기 3개
-> 동일 가중치 앙상블
-> 성공 잠재력 백분위 + 모델 불일치
```

## 입력

모든 후보는 다음을 포함해야 한다.

- `candidate_id`
- `description`
- `transcript`
- `before_context`
- `after_context`
- `duration_sec`
- Codex 7개 특징

Codex 특징은 다음과 같다.

1. `self_contained_clarity`
2. `progression_payoff`
3. `boundary_integrity`
4. `opening_pull`
5. `change_or_surprise`
6. `emotional_or_information_gain`
7. `memorable_specificity`

채널명, 조회수, 좋아요, 성과 라벨·백분위, URL, 데이터 역할, 자막 출처는
입력할 수 없다.

## 고정 앙상블

세 멤버는 동일한 익명 문자 TF-IDF와 Codex·구조 수치 특징을 사용한다.

| 멤버 | 차이 |
|---|---|
| `numeric_050` | 수치 특징 비중 0.50 |
| `channel_balanced` | 채널별 학습 쌍 총 가중치 균등 |
| `numeric_025` | 수치 특징 비중 0.25 |

세 점수를 `1/3`씩 평균한다. 모델이나 가중치는 신규 holdout 결과를 보기 전에
고정했다.

## 내부 검증

- 후보 94개, 롱폼 85개, 채널 6개
- 롱폼 ID 기반 외부 5-fold·내부 4-fold GroupKFold
- 10개 seed 반복 OOF
- C는 각 외부 학습 fold 내부에서만 선택
- 2,000회 롱폼 bootstrap
- leave-one-channel-out 진단

| 지표 | 결과 |
|---|---:|
| 채널 중심 Spearman | 0.3125 |
| 채널 Macro Spearman | 0.3203 |
| 같은 채널 Pairwise | 0.6146 |
| Local Pairwise | 0.5877 |
| Top-quintile precision | 0.4762 |
| 채널 중심 Spearman 95% CI | [0.0661, 0.5050] |
| Leave-one-channel-out 채널 중심 Spearman | 0.2969 |

내부 게이트 5개는 모두 통과했다. 그러나 같은 94개에서 구조를 선택했기
때문에 최종 상태는 `development_frozen_pending_holdout`이다.

## 실행

동결 구조 검증:

```bash
python src/train_performance_calibrator_v13.py
```

연구용 전체 데이터 artifact 생성:

```bash
python src/fit_shortform_success_judge_v13.py \
  --allow-pending-holdout
```

신규 후보 채점:

```bash
python src/predict_shortform_success_v13.py \
  --input examples/shortform_success_candidate.example.json \
  --allow-pending-holdout
```

신규 미공개 holdout의 예측을 먼저 저장한 뒤 정답 파일과 한 번만 결합한다.

```bash
python src/evaluate_shortform_success_holdout_v13.py \
  --predictions holdout_predictions.json \
  --targets holdout_targets_PRIVATE.csv \
  --output holdout_result.json \
  --holdout-id holdout_2026_08 \
  --confirm-fresh-untouched
```

정답 파일 형식은
`examples/shortform_success_holdout_targets.example.csv`를 따른다.

주요 출력:

```json
{
  "shortform_success_potential_0_100": 84.0426,
  "member_percentile_min": 84.0426,
  "member_percentile_max": 85.1064,
  "ensemble_disagreement_std": 0.5015,
  "judge_status": "development_frozen_pending_holdout"
}
```

`ensemble_disagreement_std`가 클수록 세 보정기의 판단이 불안정하다는 뜻이다.

## 다음 잠금 검증

신규 holdout에서는 다음을 바꾸지 않는다.

- Codex 7개 특징 정의
- 세 앙상블 멤버
- 멤버 가중치
- C 후보 범위
- 검증 게이트
- 성과 정규화 방식

holdout 결과를 확인한 뒤 구조를 바꾸면 해당 데이터는 다시 개발셋이 되므로,
별도의 다음 holdout이 필요하다.

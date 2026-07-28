# Vpick Shortform Success Judge and Highlight Selection

Vpick 장면 분석 결과를 기반으로 새 숏폼 후보의 성공 잠재력을 평가하는
하이브리드 Judge를 구축하고, 검증된 Judge로 하이라이트 선택 파이프라인을
개선하는 프로젝트입니다. Codex LLM은 콘텐츠·편집 특징을 추출하고, 별도
성과 보정기는 채널 내 연속 성과 백분위 순서를 학습합니다.

이 레포는 원본 유튜브 영상을 다운로드하거나 mp4 파일을 저장하지 않습니다. 대신 Vpick이 제공하는 scene timestamp, description, transcript, speech timestamp를 활용합니다.

```text
Vpick evidence -> Codex seven-axis feature extraction
Codex features + anonymous text -> continuous performance ranker
new candidate -> shortform_success_potential_0_100
```

## Shortform Success Judge v11 (2026-07-28)

최종 목표는 신규 후보가 들어왔을 때 채널·조회수·성과 라벨을 보지 않고
`shortform_success_potential_0_100`을 출력하는 것입니다. 이 값은 실제 조회수
예측이 아니라 동일한 게시 조건을 가정한 콘텐츠 기반 상대 성과 잠재력입니다.

현재 검증은 Pos/Neg와 AUC를 완전히 제외하고 채널 내 연속 성과 백분위만
사용합니다. `longform_id` GroupKFold를 외부 5-fold·내부 4-fold로 구성하고,
모델 종류와 하이퍼파라미터를 내부 fold에서만 선택합니다.

완전 중첩 파이프라인의 채널 중심 Spearman은 `0.1048`, 채널 Macro Spearman은
`0.1345`, 같은 채널 Pairwise 정확도는 `0.5463`입니다. 출처 존재 여부만 점수로
쓴 사후 대조군의 채널 중심 Spearman `0.1780`을 이기지 못했으므로 현재 상태는
**`experimental_rejected`**입니다. 연구용 후보 리랭킹은 가능하지만 검증된
성과 예측기로 주장하지 않습니다.

- 설계 및 사용법:
  [`docs/shortform_success_judge_v11_continuous.md`](docs/shortform_success_judge_v11_continuous.md)
- 연속 성과 검증:
  [`results/performance_calibrator_v11/summary_PUBLIC.json`](results/performance_calibrator_v11/summary_PUBLIC.json)
- 실행:
  `python src/train_performance_calibrator_v11.py`

## Prior Validation Protocol (2026-07-28)

이전 단계에서 Judge의 역할은 **조회수 예측**이 아니라, 같은 롱폼에서 나온 숏폼 후보의
편집·구간 선택 품질과 텍스트 근거상 콘텐츠 흡인력을 진단하고 최종 순서를
보정하는 것입니다.

검증은 세 실험으로 분리했습니다.

1. `성과 정합성`: 94개 실제 Shorts와의 대조. 보조 진단이며 낮아도 Judge의
   구간 진단 능력을 바로 기각하지 않습니다.
2. `구간 정합성`: 동일 롱폼의 Vpick 자동 후보 사이에서 실제 채택 구간의
   Hit@1, Hit@3, MRR을 측정합니다. 현재 주 검증입니다.
3. `파이프라인 비교`: 정답 주입 없이 Vpick, Ours, Ours+Judge를 비교합니다.

94개 데이터는 `neg 30 / mid 34 / pos 30`, 고유 롱폼 85개입니다. 컷 기준은
상·하위 25%가 아니라 **상위 20% / 중간 60% / 하위 20%**입니다.
`longform_id` 그룹 분할로 `dev 19 / locked_test 75`, 롱폼 중복 0을
확정했습니다.

현재 저장된 Vpick 자동 결과로 16개 동일 롱폼 풀, 총 98개 후보의 주 검증
입력을 구축했습니다. 영상별 자동 후보 수가 일정하지 않아 우연 기준은
일괄 `1/9`가 아니라 실제 풀 크기별 정확값을 사용합니다.

### Validation Results

멘토 피드백을 반영해 94개 데이터 전체를 그대로 재사용하지 않고
`longform_id` 기준 `dev 19 / locked_test 75`로 분리했습니다. 잠금 테스트에서
v10 Judge의 성과 정합성은 3분류 정확도 `0.333`, QWK `0.042`, 채널별 macro
Spearman `0.014`로 우연 수준이었습니다. 따라서 이 Judge를 조회수 예측기로
사용하지 않습니다.

`POS는 높게, NEG는 낮게, MID는 무작위`로 점수를 주는 가짜 판정자는 중간군을
포함한 locked test에서도 pooled Spearman `0.731`, QWK `0.731`을 기록했습니다.
중간군 추가만으로 극단 라벨 shortcut이 사라지지 않는다는 sanity check이며,
성과 상관을 Judge 채택 근거로 쓰지 않는 이유입니다.

동일 롱폼 구간 정합성 파일럿은 16개 풀 98개 후보를 라벨 비공개로 채점했습니다.
실제 채택 구간의 Hit@3은 `0.719`로 풀 크기별 정확 우연 기준 `0.498`보다
높았고, 정확 우연 위치 검정은 `p=0.0424`였습니다. 그러나 Hit@1은 `0.125`
(우연 `0.166`), MRR은 `0.414`(우연 `0.406`)로 최종 1등 선정 능력은
확인되지 않았습니다. 현재 근거가 지지하는 용도는 **Top3 후보 압축**까지입니다.

이 파일럿의 16개 풀은 모두 Vpick 후보가 gold IoU 기준을 통과하지 못해 실제
구간을 추가한 Vpick-miss 표본입니다. 또한 독립 API 요청이 아니라 한 세션에서
후보를 한 건씩 직접 평가했으므로 프로덕션 동등 검증으로 과장하지 않습니다.

Claude, Gemini 멀티모달, mR3 비교는 폐기하지 않고 모델·입력 ablation으로
보존합니다. Gemini 멀티모달은 점수 포화와 반복 불안정이 있어 실험 1에서
제외하고, 텍스트 baseline 이후 실험 2의 후속 ablation으로만 검토합니다.

- 검증 명세:
  [`docs/llm_judge_validation_protocol_2026-07-28.md`](docs/llm_judge_validation_protocol_2026-07-28.md)
- 설정:
  [`config/judge_validation_protocol_v1.json`](config/judge_validation_protocol_v1.json)
- 준비 감사:
  [`results/judge_validation_protocol_2026-07-28/preparation_summary_PUBLIC.json`](results/judge_validation_protocol_2026-07-28/preparation_summary_PUBLIC.json)
- 94개 성과 정합성:
  [`results/judge_validation_protocol_2026-07-28/codex_direct_v10_performance_94_PUBLIC.json`](results/judge_validation_protocol_2026-07-28/codex_direct_v10_performance_94_PUBLIC.json)
- 98개 동일 롱폼 구간 정합성:
  [`results/judge_validation_protocol_2026-07-28/codex_direct_v10_within_video_98_PUBLIC.json`](results/judge_validation_protocol_2026-07-28/codex_direct_v10_within_video_98_PUBLIC.json)

## Frozen Judge Implementation (2026-07-27)

현재 채택한 평가체계는 **원본 문맥을 포함한 후보 단독 Pointwise
LLM-as-a-Judge**입니다. 아래 60개 실행은 v10 루브릭을 확정한 역사적
기준 실행이며, 2026-07-28부터는 위의 94개/동일 롱폼 검증 프로토콜을
우선합니다.

```text
Final 60 pairs
-> Vpick evidence or timestamped transcript fallback
-> candidate-specific description
-> one label-blind candidate per request
-> editorial 4 axes + engagement 4 axes
-> pre-registered 50:50 aggregation
-> reliability / human reference / performance diagnostics
```

Fable 5 피드백을 반영해 집계식을 실행 전에 고정했고, 점수 분포 제약과
중간값 기본 규칙을 삭제했습니다. 증거 충분성은 총점에서 분리하고, 이유를 먼저
작성한 뒤 점수를 부여하며, `abstain`은 순위 계산에서 제외합니다.

현재 v10은 후보의 **편집 품질과 콘텐츠 흡인 잠재력 진단용**으로 사용합니다.
채널 내 성과 백분위와의 상관이 낮았고, POS/NEG를 유지한 V1~V5 실험도 채택
기준을 통과하지 못했으므로 조회수·좋아요 예측기로 사용하지 않습니다.

- 전체 명세:
  [`docs/best_judge_pipeline_2026-07-27.md`](docs/best_judge_pipeline_2026-07-27.md)
- 프롬프트:
  [`prompts/shortform_judge_v10_ko.md`](prompts/shortform_judge_v10_ko.md)
- 확정 데이터:
  [`data/processed/goldlabel_60_replaced_v6_channel_normalized_2026-07-23.csv`](data/processed/goldlabel_60_replaced_v6_channel_normalized_2026-07-23.csv)
- 패키지 감사:
  `python src/audit_best_judge_pipeline.py`
- 서버 실행:
  `BEST_JUDGE_REPEAT_COUNT=2 bash scripts/run_best_judge_pipeline.sh`

아래 내용은 현재 결론에 도달하기까지의 설계와 이전 실험 기록입니다.

## Evaluation System v1: five-case comparison

현재 공식 비교 결과는 [`results/evaluation_system_v1/EVALUATION_SYSTEM_COMPARISON_REPORT.md`](results/evaluation_system_v1/EVALUATION_SYSTEM_COMPARISON_REPORT.md)에 있습니다.
동일한 60개 공개 쇼츠를 기준으로 다음 다섯 체계를 비교합니다.

1. 채널 상대 성과 baseline
2. 숏폼 단독 Pointwise Judge
3. 원본 조건부 Pointwise Judge
4. 원본 조건부 Pairwise Judge
5. 독립 품질과 원본 선택 점수를 분리한 Hybrid Judge

성과 정답은 하나의 Pos/Neg 라벨로 합치지 않습니다.

```text
relative_log_view_score
= log2((short_views + 1) / (channel_median_views + 1))

channel_view_percentile
= same-channel empirical percentile
```

두 값을 별도의 연속 성과 신호로 유지하고, 상·하위 25%는 AUC/F1 계산을 위한
파생 구간으로만 사용합니다. 현재 데이터에는 업로드 후 7일·30일 고정 조회수가
없으므로 성과 검증은 모두 `exploratory`입니다.

실행:

```bash
python -m evaluation.prepare_data --config configs/evaluation.yaml
python -m evaluation.build_behavior_labels --config configs/evaluation.yaml
python -m evaluation.build_annotation_tasks --config configs/evaluation.yaml

python -m evaluation.run_case --case channel_baseline
python -m evaluation.run_case --case standalone_pointwise
python -m evaluation.run_case --case source_pointwise
python -m evaluation.run_case --case source_pairwise
python -m evaluation.run_case --case hybrid

python -m evaluation.compare_cases --config configs/evaluation.yaml
```

새 LLM 실행 전 입력 누출과 스키마만 확인:

```bash
python -m evaluation.run_judge \
  --case source_pointwise \
  --dry-run
```

실제 API 실행은 `configs/evaluation.yaml`의 `judge.provider`, `judge.model`을
지정하고 해당 API 키를 환경변수에 설정한 뒤 `--dry-run` 없이 실행합니다.
Mock 결과는 파이프라인 테스트용이며 검증 결과에 포함되지 않습니다.

현재 실제 비교에서는 어떤 Case도 성과 예측 타당성을 통과하지 못했습니다.
Case 3은 11개 반복 표본에서 Spearman `0.8611`로 안정적이었지만, 성과 백분위와의
상관은 `0.0525`, Top-25 AUC는 `0.4865`였습니다. 따라서 잠정적으로
`원본 조건부 품질 진단`에만 사용할 수 있고 `성과 예측 점수`로 부르면 안 됩니다.

설계, 데이터 스키마, N/A 사유는
[`docs/evaluation_system_v1.md`](docs/evaluation_system_v1.md)에 정리했습니다.

## Judge-first v9

현재 최우선 과제는 새 후보의 조회수를 맞히는 분류기가 아니라, 후보 하나의
`편집·구간 선택 품질`과 `내재적 확산 잠재력`을 반복 가능하게 채점하는
LLM-as-a-Judge를 확립하는 것입니다.

- 메인 평가는 후보당 단독 요청인 pointwise 절대평가입니다.
- editorial 4축과 engagement 4축을 분리합니다.
- Pos/Neg는 engagement 축의 외부 타당도 검증에만 사용합니다.
- 인간 평가는 12개 anchor x 2명만 사용하며 LLM Judge를 대체하지 않습니다.
- Ours와 Vpick 후보는 Judge가 검증되기 전에는 평가체계 구축에 넣지 않습니다.

```bash
python src/run_shortform_judge_v9.py \
  --repeat-count 2 \
  --no-cache

python src/evaluate_shortform_judge_v9.py \
  --scores results/shortform_judge_v9/shortform_judge_v9_scores.csv \
  --targets deliverables/2026-07-24/performance_judge_v1/candidate_targets_PRIVATE.csv \
  --dataset-role development \
  --out-dir results/shortform_judge_v9/validation
```

상세 정의와 합격 기준은
[`docs/llm_judge_first_protocol_v1.md`](docs/llm_judge_first_protocol_v1.md)에
있습니다.

현재 Opus 4.8 실행은 1회차 60/60, 2회차 11/60까지 완료됐습니다. Anthropic
워크스페이스 사용 한도로 남은 49건은 대기 중이며, 2회 공통 coverage가
11/60이므로 상태는 `incomplete_repeat_run`입니다. 부분 결과는
[`results/shortform_judge_v9/PARTIAL_RUN_REPORT.md`](results/shortform_judge_v9/PARTIAL_RUN_REPORT.md)에
기록했습니다.

## Rejected Performance Judge v1

새로 생성된 단일 숏폼 후보의 채널 내 고성과 가능성을 평가하기 위한 별도
Judge입니다. 실제 공개 숏폼 60개(Pos 30, Neg 30)를 사용하며, LLM은 성과를
직접 선언하는 대신 훅, 반전, 감정 고점, 인용성, payoff, 시작·종료 경계 같은
관찰 가능한 특징을 추출합니다. 최종 점수는 이 특징을 학습한 L2 정규화
로지스틱 모델이 계산합니다.

```text
새 숏폼 후보
-> Vpick 장면 설명·대사·타임스탬프
-> blind pointwise LLM 평가
-> 검증된 특징만 결합
-> high_performance_score_0_100
```

누수 방지 원칙:

- LLM 입력에는 채널명, 조회수, 백분위, Pos/Neg를 넣지 않습니다.
- 같은 롱폼의 다른 숏폼을 통째로 제외하는 Leave-One-Longform-Out 검증을
  주 평가로 사용합니다.
- Leave-One-Channel-Out 결과는 처음 보는 채널에 대한 스트레스 테스트로
  따로 기록합니다.
- 30/30 극단 표본으로 학습했으므로 출력은 정확한 조회수나 실제 확률이
  아닙니다. 고성과·저성과 경계는 LOLO 예측에서 각 극단군 정밀도 75%를
  목표로 산출하며, 두 경계 사이는 `불확실`로 해석합니다.

학습 및 검증:

```bash
python src/build_performance_judge_dataset_v1.py
python src/train_performance_judge_v1.py
```

단일 후보 추론:

```bash
python src/predict_performance_judge_v1.py \
  --candidate-json candidate.json \
  --judge-json judgment.json \
  --model-artifact deliverables/2026-07-24/performance_judge_v1/model_artifact.json
```

주요 결과:

```text
deliverables/2026-07-24/performance_judge_v1/
├── candidates_blind.jsonl
├── candidate_targets_PRIVATE.csv
├── model_comparison.csv
├── cross_validated_predictions_PRIVATE.csv
├── model_artifact.json
└── PERFORMANCE_JUDGE_REPORT.md
```

현재 검증 상태는 **`rejected`**입니다. 격리된 Claude Opus 4.8 API 평가에서
6축 고정 총점 AUC는 `0.504`, 동일 v7 체크리스트 총점 AUC는 `0.364`였고,
학습 보정 모델도 배포 기준을 통과하지 못했습니다. 기존 Codex 세션 점수는 더
높았지만 비공개 라벨과 실행 맥락이 완전히 격리된 API 실행이 아니므로 배포
근거에서 제외합니다. `predict_performance_judge_v1.py`는 검증 실패 artifact를
기본적으로 거부합니다.

전체 설계와 파인튜닝 보류 근거는
[`docs/performance_judge_v1_design.md`](docs/performance_judge_v1_design.md)에
정리했습니다. 검증 실패 이후 평가체계의 역할을 다시 나눈 설계는
[`docs/performance_judge_next_iteration.md`](docs/performance_judge_next_iteration.md)에
있습니다.

한 줄로 정리하면:

```text
Vpick의 멀티모달 장면 분석 결과 위에서 작동하는 LLM 기반 숏폼 하이라이트 선택 파이프라인
```

## What Is Included

이 레포는 제출/공유용 clean version입니다.

포함:

- 베스트 파이프라인 코드
- LLM rerank prompt/config
- 파일럿 평가 데이터와 결과
- gold dataset v1
- control dataset
- Vpick asset batch upload script
- 평가 지표 및 실행 문서

제외:

- 원본 유튜브 영상
- 다운로드된 mp4 클립
- Vpick 계정 정보
- API key
- raw LLM response cache
- 전체 Vpick raw scene dump
- 실험 중간 산출물 전체

## Dataset

### Pilot Ready Dataset

파일:

```text
data/processed/ready/ready_dataset_pairs.csv
```

초기 실험에 사용한 파일럿 데이터입니다.

- gold short pairs: 11
- unique long-form videos: 3
- label confidence: high 11

이 데이터로 Vpick baseline과 ours adaptive coverage pipeline의 성능을 비교했습니다.

### Gold Dataset v1

파일:

```text
data/processed/gold_dataset_pairs_main.csv
```

팀원 PR #1에서 추가된 확장 gold dataset입니다.

- gold short pairs: 27
- unique long-form videos: 26
- channels: 숏박스, 피식대학, OOTB_Studio, 빠더너스, 워크맨
- label confidence: high 22, medium 2, low 3

각 row는 실제 Shorts 성과와 원본 롱폼 내 gold segment를 함께 기록합니다.

주요 column:

```text
pair_id
long_video_id
short_video_id
channel_name
long_video_url
short_video_url
gold_start_sec
gold_end_sec
short_views
short_likes
label_confidence
label_notes
vpick_project_id
vpick_asset_id
```

`short_views`, `short_likes`, `label_notes`의 채널 내 백분위 정보는 gold label 신뢰도 판단에 사용합니다. 장르/채널마다 조회수 규모가 다르기 때문에 raw view count만 절대 비교하지 않고, 가능하면 채널 내 상대 성과를 함께 봅니다.

### Negative Gold Dataset

파일:

```text
data/processed/gold_dataset_pairs_control.csv
```

실제로 게시된 숏폼 중 성과가 낮은 `Neg` Gold 데이터입니다.

- negative Gold pairs: 7
- unique long-form videos: 5
- channels: 숏박스, 워크맨

45개 구간은 모두 실제 게시 숏폼에서 얻은 Gold입니다. Gold 여부와 성과 라벨을 혼동하지 않도록 통합 CSV의 `evaluation_role`은 모두 `gold`이며, 별도 `performance_label`로 `pos`, `neg`, `unlabeled`를 기록합니다.

2026-07-21 기준 Pilot 11개도 동일 채널 Shorts 50개 코호트의 조회수 백분위로 분류했습니다. 현재 전체 라벨은 `Pos 31`, `Neg 9`, `Unlabeled 5`이며, 기준은 상위 25%를 Pos, 하위 25%를 Neg, 중간 50%를 Unlabeled로 유지합니다. 수집 시점과 통계 출처는 `label_notes`와 `data/processed/pilot_channel_short_stats_2026-07-21.csv`에 기록합니다.

## Vpick Asset Upload

팀원 PR #1에서 gold dataset 롱폼을 Vpick에 일괄 업로드하고 CSV에 `vpick_project_id`, `vpick_asset_id`를 자동 기입하는 스크립트가 추가되었습니다.

파일:

```text
scripts/batch_upload_assets.py
```

사용 예:

```bash
export VPICK_EMAIL="..."
export VPICK_PASSWORD="..."

python scripts/batch_upload_assets.py \
  --csv data/processed/gold_dataset_pairs_main.csv \
  --csv data/processed/gold_dataset_pairs_control.csv
```

또는 이미 발급된 token이 있으면:

```bash
export VPICK_ACCESS_TOKEN="..."
```

동작:

1. CSV에서 `vpick_asset_id`가 비어 있는 고유 롱폼 목록 추출
2. 채널별 Vpick project 생성 또는 재사용
3. YouTube long-form asset 업로드 요청
4. asset status polling
5. `READY` 상태가 되면 scene 개수 확인
6. CSV에 `vpick_project_id`, `vpick_asset_id` 자동 기입
7. 진행 상태를 `data/raw/vpick/upload_state.json`에 저장

중간에 실패하거나 인증이 만료되어도 같은 명령을 다시 실행하면 state file을 기반으로 이어서 진행할 수 있습니다.

## Best Pipeline

현재 베스트 파이프라인은 아래 조합입니다.

```text
Vpick scene data
+ timeline_bins_bridge candidate generation
+ variable trim windows
+ deterministic rerank
+ adaptive_coverage Top5 slate
+ LLM rerank
```

### 1. Scene Data

Vpick API에서 장면 분석 결과를 수집합니다.

- scene id
- scene start/end timestamp
- scene description
- speech transcript
- speech-level timestamp

원본 영상을 직접 분석하는 것이 아니라, Vpick이 만든 장면 데이터 위에서 선택 파이프라인을 구축합니다.

### 2. Candidate Generation

실제 Shorts는 한 scene만 잘라 쓰는 경우보다 앞뒤 setup이 붙는 경우가 많습니다. 그래서 여러 후보 window를 만듭니다.

- `scene_i` 단독 후보
- `scene_{i-1} + scene_i` bridge 후보
- 30s / 45s / 60s / 75s variable windows
- scene boundary windows
- speech boundary windows

gold timestamp는 후보 생성에 사용하지 않습니다.

### 3. Deterministic Rerank

후보별로 다음 신호를 조합해 1차 점수를 계산합니다.

- speech density
- duration fit
- titleability
- rank prior
- speech boundary bonus
- filler penalty
- source band diversity

### 4. Adaptive Coverage Top5

단순히 점수 높은 후보만 고르면 특정 시간대에 후보가 몰릴 수 있습니다. `adaptive_coverage`는 한 롱폼 안에서 여러 숏폼 후보가 나올 수 있다는 전제를 반영합니다.

- 겹치는 후보를 event cluster로 묶음
- 각 event에서 대표 후보 선택
- intro 구간 과집중 방지
- 서로 다른 시간대와 이벤트를 커버하는 Top5 slate 구성

### 5. LLM Rerank

Top5 후보만 LLM에 전달해 최종 순서를 조정합니다.

현재 config:

```text
config/llm_rerank_top5_genre_lang.json
```

모델:

- GPT-4o mini
- Claude Haiku 4.5

LLM 판단 기준:

- hook
- setup / payoff
- standalone 이해 가능성
- interaction / reaction / conflict
- ending completeness
- titleability

## Pilot Evaluation Results

파일럿 평가 데이터 기준:

- long-form videos: 3
- gold short pairs: 11
- baseline: Vpick auto-generated shortform result
- ours: Vpick scene data 기반 candidate generation + adaptive coverage + LLM rerank

| System | Model | Top1 Core | Core@3 | Tight@3 | IoU@3 | Core@5 | Tight@5 | IoU@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vpick baseline | Vpick auto | 0.091 | 0.091 | 0.000 | 0.026 | 0.091 | 0.000 | 0.046 |
| Ours | adaptive_coverage + Claude Haiku 4.5 | 0.182 | 0.455 | 0.455 | 0.313 | 0.545 | 0.545 | 0.393 |
| Ours | adaptive_coverage + GPT-4o mini | 0.182 | 0.364 | 0.364 | 0.257 | 0.545 | 0.545 | 0.391 |

요약:

```text
Vpick baseline Core@5: 0.091
Ours adaptive_coverage + Claude Core@5: 0.545
```

즉, Vpick의 영상 분석 자체를 대체하는 것이 아니라, Vpick이 제공한 장면 데이터 위에서 더 나은 하이라이트 선택기를 만든 것입니다.

## Historical LLM Judge Experiments

아래 v4~v8 실험은 v9 이전의 기록입니다. 반복 안정성, 인간 정합성, 성과
정합성을 분리해야 한다는 근거로 보존하며 현재 주 Judge로 사용하지 않습니다.

```text
docs/evaluation_system_pointwise_v4.md
prompts/shortform_pointwise_judge_v4_ko.md
config/gold_pointwise_judge_v4_terra.json
config/gold_pointwise_judge_v4_gemini_multimodal.json
scripts/run_gold_pointwise_judge_v4.sh
reports/gold_pointwise_judge_v4_2026-07-22.md
results/gold_pointwise_judge_v4/
```

현재 결과에서 Terra의 반복 Spearman은 편집 0.8263·성과 0.7968이고, Gemini 멀티모달은 편집 0.3617·성과 0.5172입니다. Gemini의 Pos>Neg AUC는 0.6333으로 더 높지만 반복 신뢰성이 낮아 주 Judge로 사용하지 않습니다. 최종 검증은 층화 표본 15개 x 3명의 인간 절대평가와의 상관으로 결정합니다.

```text
src/build_judge_candidates.py
src/run_llm_judge.py
src/evaluate_llm_judge.py
prompts/shortform_candidate_judge_v2_ko.md
config/gold_judge_v1.json
docs/llm_as_judge.md
```

Gold 45개에 대한 평가체계 검증은 다음과 같이 실행합니다.

```bash
REPEAT_COUNT=2 bash scripts/run_gold_judge_validation.sh
```

Judge는 고정된 익명 Gold 구간만 입력받으며 Pos/Neg 라벨, 조회수, 좋아요를 볼 수 없습니다. 현재 통합 데이터의 Pos 31개와 Neg 9개로 외부 성과 정합성을 사후 계산하고, Unlabeled 5개는 구분력 계산에서 제외합니다.

Current consolidated Gold data and the first full Judge run are available at:

```text
data/processed/gold_dataset_pairs_all.csv
data/processed/gold_dataset_pairs_all_summary.json
reports/gold_judge_v1_2026-07-21.md
results/gold_judge_v1/
```

The matched pairwise v4 evaluation separates editorial interval quality from intrinsic performance potential, reverses left/right presentation on the second repeat, and validates the Judge against three independent human raters. See:

```text
docs/evaluation_system_v4.md
prompts/shortform_pairwise_judge_v4_ko.md
config/gold_pairwise_judge_v4.json
scripts/run_gold_pairwise_judge_v4.sh
reports/gold_pairwise_judge_v4_2026-07-21.md
results/gold_pairwise_judge_v4/
```

The v5 multimodal Judge sends the same 18 blind comparisons to Gemini as clipped public YouTube inputs. The completed Gemini 3.1 Flash-Lite run produced 36 real judgments in 10 batched API requests. It evaluates audio, expressions, actions, on-screen text, and boundaries instead of relying on transcript alone.

The experiment reached full coverage, but left/right reversal agreement was only 0.5000 for editorial quality and 0.6111 for performance potential. It therefore remains a multimodal diagnostic, not a validated primary Judge. Terra and Claude remain independent text diagnostics, and three-person blind human evaluation remains the final validity gate.

```text
docs/evaluation_system_v5_multimodal.md
prompts/shortform_pairwise_judge_v5_multimodal_ko.md
prompts/shortform_pairwise_judge_v5_multimodal_batch_ko.md
config/gold_pairwise_judge_v5_multimodal_batch.json
scripts/run_gold_pairwise_gemini_multimodal_batch_v5.sh
reports/gold_pairwise_judge_v5_multimodal_2026-07-21.md
results/gold_pairwise_judge_v5_multimodal_batch/
```

The same-model modality ablation compares Gemini 3.1 Flash-Lite with text evidence only against the completed video/audio run. Multimodal input improved strict Pos preference from 0.1667 to 0.3889 and decisive Pos accuracy from 0.3000 to 0.6364, but editorial repeat agreement fell from 0.6667 to 0.5000. It is therefore used as a performance-potential diagnostic rather than a standalone editorial Judge.

```text
config/gold_pairwise_judge_v5_gemini_text_batch.json
src/run_gemini_text_batch_judge.py
scripts/run_gold_pairwise_gemini_text_ab_v5.sh
reports/gemini_multimodal_ablation_2026-07-21.md
results/gold_pairwise_judge_v5_text_batch/
results/gold_pairwise_judge_v5_ab/
```

## Highlight Quality Evaluation v1

성과 예측과 별도로, **같은 롱폼 안에서 어떤 구간이 더 좋은 하이라이트인가**를 평가하는
블라인드 Judge입니다. 채널명, 조회수, Pos/Neg, 게시 여부, 후보 출처를 모델 입력에서
제외하고 다음 6개 항목을 0~4점으로 채점합니다.

| Dimension | Weight |
|---|---:|
| source salience | 0.20 |
| hook | 0.20 |
| payoff | 0.20 |
| self-contained | 0.15 |
| density | 0.15 |
| boundary | 0.10 |

총점은 모델이 직접 만들지 않고 코드가 0~100점으로 계산합니다. Pointwise가 기본 평가이고,
pairwise는 같은 롱폼 후보끼리만 비교하며 A/B 순서를 뒤집어 순서 민감도를 진단합니다.

```text
config/highlight_quality_judge_v1.json
prompts/highlight_quality_pointwise_v1_ko.md
prompts/highlight_quality_pairwise_v1_ko.md
src/highlight_quality_judge_v1.py
src/build_highlight_quality_eval_v1.py
src/run_highlight_quality_judge_v1.py
deliverables/2026-07-24/highlight_quality_v1/
```

평가셋은 30개 롱폼에서 154개 후보와 동일 롱폼 비교 124쌍을 구성합니다. 후보 출처는
published short 30개, boundary shift 60개, hard negative 30개, random 30개,
Vpick baseline 2개, existing model 2개입니다. 공개된 고성과 숏폼은 신뢰 가능한 정답
신호이지만 완전한 정답으로 강제하지 않습니다.

```bash
python src/build_highlight_quality_eval_v1.py

python src/run_highlight_quality_judge_v1.py \
  --mode pointwise \
  --provider gemini \
  --model gemini-3.1-flash-lite

python src/run_highlight_quality_judge_v1.py \
  --mode pairwise \
  --provider gemini \
  --model gemini-3.1-flash-lite \
  --repeat-swapped
```

Gemini 3.1 Flash-Lite 전체 검증은 pointwise 154개와 pairwise 124쌍을 모두 완료했습니다.
Pointwise에서 published short의 boundary shift 대비 엄격 승률은 35%였습니다. Pairwise에서
A/B 반전까지 일치한 81쌍만 사용하면 boundary shift 대비 승률은 80.7%였지만, 전체 순서
일치율이 65.3%에 그쳤습니다. 직접 비교는 경계 차이에 더 민감했으나 위치 편향이 커서 현재
모델을 최종 검증된 주 Judge로 간주하지 않습니다. 자세한 구현 범위·결과·한계는
`deliverables/2026-07-24/highlight_quality_v1/IMPLEMENTATION_AUDIT.md`에 기록합니다.

```text
results/highlight_quality_judge_v1_full/EVALUATION_REPORT.md
results/highlight_quality_judge_v1_full/evaluation_summary.json
```

### Scene Evidence Coverage

60개 숏폼 후보의 54개 고유 롱폼 중 43개는 실제 Vpick 장면 분석을 사용합니다. Vpick 재시도
후에도 실패한 11개 롱폼은 `yt-dlp`의 타임코드 자막을 장면으로 묶고 Gemini가 자막에 근거한
설명을 생성해 보완합니다. 이 fallback은 시각 근거가 없다고 명시하며 실제 Vpick 결과를
덮어쓰거나 `vpick_available=1`로 기록하지 않습니다.

```text
src/build_subtitle_fallback_scenes.py
data/raw/subtitle_fallback_scenes/
deliverables/2026-07-24/performance_ranker/vpick_enrichment_summary.json
```

## Repository Structure

```text
.
├── src/
│   ├── select_diverse_candidates.py
│   ├── expand_trim_windows.py
│   ├── rerank_trim_candidates.py
│   ├── build_longform_slate.py
│   ├── llm_rerank_top5.py
│   ├── evaluate_predictions.py
│   ├── llm_client.py
│   ├── vpick_client.py
│   └── segments.py
├── scripts/
│   ├── run_best_no_api_pipeline.sh
│   └── batch_upload_assets.py
├── config/
│   └── llm_rerank_top5_genre_lang.json
├── prompts/
│   ├── highlight_reranker_v1_ko_variety_vlog.md
│   ├── highlight_reranker_v1_ko_lecture.md
│   └── highlight_reranker_v1_en.md
├── data/
│   ├── processed/
│   │   ├── ready/ready_dataset_pairs.csv
│   │   ├── gold_dataset_pairs_main.csv
│   │   └── gold_dataset_pairs_control.csv
│   ├── raw/.gitkeep
│   └── templates/
├── results/
│   ├── vpick_baseline/
│   └── ours_adaptive_coverage/
├── docs/
└── reports/
```

## How To Run

### 1. Environment

Python 3.10 이상을 권장합니다. 핵심 파이프라인은 Python standard library 중심으로 작성되어 있습니다.

LLM rerank를 실제 API로 실행하려면:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

Vpick upload/API 작업을 하려면:

```bash
export VPICK_EMAIL="..."
export VPICK_PASSWORD="..."
```

또는:

```bash
export VPICK_ACCESS_TOKEN="..."
```

### 2. Prepare Vpick Scene JSON

이 레포에는 전체 Vpick raw scene dump를 포함하지 않습니다. 재현 실행을 위해서는 아래 위치에 scene JSON을 넣어야 합니다.

```text
data/raw/vpick/{long_video_id}_scenes.json
```

예:

```text
data/raw/vpick/NS7tSrMrWsc_scenes.json
data/raw/vpick/OrCOflk2QmQ_scenes.json
data/raw/vpick/heifaIjlSUc_scenes.json
```

gold dataset v1을 새로 분석하려면 먼저 `scripts/batch_upload_assets.py`로 Vpick asset을 준비한 뒤 scene JSON을 수집합니다.

### 3. Run Best Pipeline

API 비용 없이 구조만 확인하는 dry-run:

```bash
bash scripts/run_best_no_api_pipeline.sh \
  data/processed/ready/ready_dataset_pairs.csv \
  data/processed/best_pipeline
```

실제 LLM rerank 실행:

```bash
LLM_RERANK_DRY_RUN=0 bash scripts/run_best_no_api_pipeline.sh \
  data/processed/ready/ready_dataset_pairs.csv \
  data/processed/best_pipeline_real_llm
```

gold dataset v1에 적용하려면 dataset path를 교체합니다.

```bash
bash scripts/run_best_no_api_pipeline.sh \
  data/processed/gold_dataset_pairs_main.csv \
  data/processed/gold_dataset_v1_pipeline
```

## Evaluation Metrics

주요 지표:

- Top1 Core Hit
- Core Recall@3
- Tight Recall@3
- Core Recall@5
- Tight Recall@5
- Mean Best IoU@5

`Core Hit`은 예측 구간이 gold 구간의 핵심 부분을 충분히 포함하는지 보는 지표입니다.  
`Tight Hit`은 Core Hit보다 더 엄격하게 IoU까지 요구합니다.  
`IoU`는 예측 구간과 gold 구간의 시간상 겹침 정도입니다.

```text
IoU = overlap / union
```

## Multimodal Evaluation

평가체계에는 로컬 mp4를 내려받지 않고 Gemini의 공개 YouTube 구간 입력을 사용한 멀티모달 쌍대비교가 구현되어 있습니다. 이는 고정된 Gold 후보를 평가하는 Judge이며 Ours의 후보 생성·선택 파이프라인에는 아직 포함하지 않습니다.

향후 합법적으로 후보 클립을 확보하거나 동일한 구간 입력 방식을 안정화하면 아래 구조로 선택 단계까지 확장할 수 있습니다.

```text
Top15 candidate clips
-> Qwen-Omni video/audio verifier
-> text score + multimodal score fusion
-> final Top5
```

현재 후보 선택은 Vpick의 멀티모달 장면 분석 결과를 사용하고, 평가는 Gemini 영상 Judge를 보조 진단으로 함께 보고합니다. Flash-Lite의 낮은 좌우 반전 일치도 때문에 최종 판정은 3인 인간 블라인드 평가 전까지 보류합니다.

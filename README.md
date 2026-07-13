# Vpick Highlight Selection Pipeline

Vpick 장면 분석 결과를 기반으로 유튜브 롱폼 영상에서 숏폼 하이라이트 후보를 선택하고, 실제 Shorts로 업로드된 gold 구간과 비교 평가하는 프로젝트입니다.

이 레포는 원본 유튜브 영상을 다운로드하거나 mp4 파일을 저장하지 않습니다. 대신 Vpick이 제공하는 scene timestamp, description, transcript, speech timestamp를 활용합니다.

```text
Long-form YouTube video
-> Vpick scene analysis
-> scene / speech timestamp, description, transcript
-> candidate window generation
-> adaptive coverage Top5 slate
-> LLM rerank
-> evaluation against real Shorts gold pairs
```

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

### Control Dataset

파일:

```text
data/processed/gold_dataset_pairs_control.csv
```

대조군 데이터입니다.

- control short pairs: 7
- unique long-form videos: 5
- channels: 숏박스, 워크맨

control set은 메인 평가용 gold가 아니라 judge 검증, 신뢰도 점검, 낮은 성과 short와의 비교 실험에 사용합니다.

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

## Multimodal Future Work

Qwen-Omni 같은 멀티모달 모델을 붙이려면 후보 구간의 실제 mp4 클립이 필요합니다. 현재는 유튜브 영상 다운로드 및 재배포가 법적/기술적으로 부담되므로 베스트 파이프라인에는 포함하지 않았습니다.

향후 합법적으로 후보 클립을 확보할 수 있다면 아래 구조를 추가할 수 있습니다.

```text
Top15 candidate clips
-> Qwen-Omni video/audio verifier
-> text score + multimodal score fusion
-> final Top5
```

현재 버전에서는 Vpick의 멀티모달 장면 분석 결과를 사용하고, 그 위에서 후보 선택과 평가 체계를 개선합니다.

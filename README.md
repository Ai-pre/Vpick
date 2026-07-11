# Vpick Highlight Selection Pipeline

Vpick 장면 분석 결과를 기반으로 유튜브 롱폼 영상에서 숏폼 하이라이트 후보를 고르고, 실제 업로드된 Shorts 구간과 비교해 평가하는 프로젝트입니다.

## 핵심 아이디어

이 레포는 원본 유튜브 영상을 직접 다운로드하거나 보관하지 않습니다. 대신 Vpick이 제공하는 장면 분석 결과를 사용합니다.

```text
Long-form YouTube video
-> Vpick scene analysis
-> scene timestamp / description / transcript
-> candidate window generation
-> adaptive coverage Top5 slate
-> LLM rerank
-> evaluation against gold Shorts pairs
```

따라서 현재 파이프라인은 다음처럼 정의합니다.

```text
Vpick의 멀티모달 분석 결과(scene, timestamp, description, transcript)를 활용한
LLM 기반 숏폼 하이라이트 선택 파이프라인
```

## 현재 베스트 파이프라인

현재까지 가장 안정적인 조합은 아래입니다.

```text
Vpick scene data
+ timeline_bins_bridge candidate generation
+ variable trim windows
+ deterministic rerank
+ adaptive_coverage Top5 slate
+ LLM rerank
```

### 1. Vpick 장면 데이터

Vpick API에서 아래 정보를 수집합니다.

- scene id
- scene start/end timestamp
- scene description
- speech transcript
- speech-level timestamp

원본 mp4나 유튜브 영상을 레포에 포함하지 않습니다.

### 2. Gold Pair 데이터셋

실제 Shorts로 업로드된 구간을 사람이 원본 롱폼 안에서 매칭해 gold segment로 기록합니다.

포함 정보:

- long_video_url
- short_video_url
- gold_start_sec
- gold_end_sec
- short views / likes
- genre / channel metadata

### 3. 후보 생성

Vpick scene 하나만 쓰면 setup이 잘릴 수 있으므로 여러 후보를 만듭니다.

- `scene_i` 단독 후보
- `scene_{i-1} + scene_i` bridge 후보
- 30s / 45s / 60s / 75s variable window
- scene boundary window
- speech boundary window

정답 timestamp는 후보 생성에 사용하지 않습니다.

### 4. Deterministic Rerank

후보별로 아래 신호를 조합해 1차 점수를 만듭니다.

- speech density
- duration fit
- titleability
- rank prior
- speech boundary bonus
- filler penalty
- source band diversity

### 5. Adaptive Coverage Top5

단순히 점수 높은 후보만 고르면 특정 시간대에 몰릴 수 있습니다. 그래서 long-form timeline을 넓게 커버하는 Top5 slate를 구성합니다.

- 겹치는 후보를 event cluster로 묶음
- 각 event에서 대표 후보 선택
- intro 구간 과집중 방지
- Top5가 서로 다른 시간대와 이벤트를 커버하도록 선택

### 6. LLM Rerank

Top5 후보만 LLM에 전달해 최종 순서를 조정합니다.

현재 config:

- GPT-4o mini
- Claude Haiku 4.5

LLM은 아래 기준으로 후보를 비교합니다.

- hook
- setup/payoff
- standalone 이해 가능성
- interaction / reaction / conflict
- ending completeness
- titleability

## 실험 결과

평가 데이터:

- long-form videos: 3
- gold short pairs: 11
- baseline: Vpick auto-generated shortform result
- ours: Vpick scene data 위에서 자체 후보 생성 및 LLM rerank

| System | Model | Top1 Core | Core@3 | Tight@3 | IoU@3 | Core@5 | Tight@5 | IoU@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vpick baseline | Vpick auto | 0.091 | 0.091 | 0.000 | 0.026 | 0.091 | 0.000 | 0.046 |
| Ours | adaptive_coverage + Claude Haiku 4.5 | 0.182 | 0.455 | 0.455 | 0.313 | 0.545 | 0.545 | 0.393 |
| Ours | adaptive_coverage + GPT-4o mini | 0.182 | 0.364 | 0.364 | 0.257 | 0.545 | 0.545 | 0.391 |

요약하면, 현재 베스트 파이프라인은 Vpick baseline 대비 Core@5 기준 `0.091 -> 0.545`로 개선되었습니다.

## 레포 구조

```text
.
├── src/
│   ├── select_diverse_candidates.py
│   ├── expand_trim_windows.py
│   ├── rerank_trim_candidates.py
│   ├── build_longform_slate.py
│   ├── llm_rerank_top5.py
│   └── evaluate_predictions.py
├── config/
│   └── llm_rerank_top5_genre_lang.json
├── prompts/
│   ├── highlight_reranker_v1_ko_variety_vlog.md
│   ├── highlight_reranker_v1_ko_lecture.md
│   └── highlight_reranker_v1_en.md
├── scripts/
│   └── run_best_no_api_pipeline.sh
├── data/
│   ├── templates/
│   └── processed/ready/ready_dataset_pairs.csv
├── results/
│   ├── vpick_baseline/
│   └── ours_adaptive_coverage/
├── docs/
└── reports/
```

## 실행 방법

### 1. 환경 준비

Python 3.10 이상을 권장합니다. 현재 핵심 파이프라인은 Python standard library 중심으로 작성되어 별도 패키지 설치 없이 동작하도록 구성했습니다.

LLM rerank를 실제 API로 돌리려면 환경변수가 필요합니다.

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

API 비용 없이 dry-run으로 구조만 확인할 수도 있습니다.

### 2. Vpick scene JSON 준비

이 레포에는 원본 영상과 전체 Vpick raw scene JSON을 포함하지 않습니다.

실제 재현을 위해서는 아래 위치에 Vpick scene JSON을 넣어야 합니다.

```text
data/raw/vpick/{long_video_id}_scenes.json
```

예:

```text
data/raw/vpick/NS7tSrMrWsc_scenes.json
data/raw/vpick/OrCOflk2QmQ_scenes.json
data/raw/vpick/heifaIjlSUc_scenes.json
```

### 3. 베스트 파이프라인 실행

```bash
bash scripts/run_best_no_api_pipeline.sh \
  data/processed/ready/ready_dataset_pairs.csv \
  data/processed/best_pipeline
```

기본값은 LLM API를 호출하지 않는 dry-run입니다.

실제 LLM rerank를 실행하려면:

```bash
LLM_RERANK_DRY_RUN=0 bash scripts/run_best_no_api_pipeline.sh \
  data/processed/ready/ready_dataset_pairs.csv \
  data/processed/best_pipeline_real_llm
```

## 평가 지표

주요 평가지표:

- Top1 Core Hit
- Core Recall@3
- Tight Recall@3
- Core Recall@5
- Tight Recall@5
- Mean Best IoU@5

`Core Hit`은 예측 구간이 gold 구간의 핵심 부분을 충분히 포함하는지 보는 지표입니다.  
`Tight Hit`은 Core Hit보다 더 엄격하게 IoU까지 요구합니다.

## Qwen-Omni / Multimodal Future Work

Qwen-Omni 같은 멀티모달 모델을 사용하려면 후보 구간의 실제 mp4 클립이 필요합니다.

현재는 유튜브 영상 다운로드 및 재배포가 법적/기술적으로 부담되기 때문에, 이 레포의 베스트 파이프라인에는 포함하지 않았습니다.

향후 합법적으로 후보 클립을 확보할 수 있다면 아래 구조를 추가할 수 있습니다.

```text
Top15 candidate clips
-> Qwen-Omni video/audio verifier
-> text score + multimodal score fusion
-> final Top5
```

## 포함하지 않은 것

의도적으로 제외한 파일:

- 원본 유튜브 영상
- 다운로드된 mp4 클립
- Vpick 계정 정보
- API key
- raw LLM response cache
- 전체 Vpick raw scene dump
- 실험 중간 산출물 전체

이 레포는 제출/공유를 위한 clean version입니다.

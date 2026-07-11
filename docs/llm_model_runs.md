# LLM 모델 실행 계획

## 기본 선택

이번 프로젝트의 기본 비교 모델은 2개로 둔다.

1. OpenAI: `gpt-4o-mini`
2. Claude: `claude-haiku-4-5-20251001`

설정 파일:

- `config/llm_runs.json`

## Claude 모델 선택 이유

팀당 Claude API 예산이 $25로 제한되어 있으므로 기본 모델은 Claude Haiku 4.5로 둔다.

이유:

- Anthropic 문서 기준 Claude Haiku 4.5는 최신 비교표에서 가장 빠른 모델군으로 분류된다.
- 가격은 input $1 / output $5 per million tokens로, Sonnet/Opus/Fable보다 저렴하다.
- Anthropic의 model selection guide도 비용 민감, 대량 처리, 초기 프로토타입은 Haiku 4.5부터 시작하라고 권장한다.
- 이 과제는 후보 구간 top-k 선택 문제라 초반에는 최고가 모델보다 평가 반복량이 더 중요하다.

운영 방식:

- 기본 실험: `claude-haiku-4-5-20251001`
- Haiku가 계속 gold 구간을 놓치는 실패 케이스가 보이면 일부 pair만 `claude-sonnet-5` 또는 그 시점의 Sonnet 모델로 재실험
- 발표에서는 "비용 제한 내 반복 실험을 위해 Haiku를 기본으로 두고, 실패 케이스에 한해 상위 모델 ablation을 수행"이라고 설명

## 비용 가드

`llm_select_candidates.py`는 Claude run에 대해 대략 비용을 계산한다.

현재 설정:

```json
{
  "run_id": "claude_haiku45_v1",
  "provider": "anthropic",
  "model": "claude-haiku-4-5-20251001",
  "max_estimated_usd": 25.0
}
```

예상 비용이 `max_estimated_usd`를 넘으면 실행을 중단한다.

주의:

- 비용은 실행 전 대략 추정치다.
- 실제 청구 비용은 provider usage 값과 과금 정책을 기준으로 확인해야 한다.
- 그래도 실수로 대량 실행하는 것을 막는 1차 안전장치로 충분하다.

## 실행 방법

API 키는 코드에 저장하지 않고 환경변수로 넣는다.

PowerShell:

```powershell
$env:OPENAI_API_KEY="..."
$env:ANTHROPIC_API_KEY="..."
```

Bash:

```bash
export OPENAI_API_KEY='...'
export ANTHROPIC_API_KEY='...'
```

API 호출 없이 파이프라인만 검증:

```bash
python3 src/llm_select_candidates.py \
  --dataset data/processed/pilot_dataset_pairs.csv \
  --runs config/llm_runs.json \
  --output data/processed/llm_runs/pilot_llm_predictions.csv \
  --dry-run
```

OpenAI만 실제 실행:

```bash
python3 src/llm_select_candidates.py \
  --dataset data/processed/pilot_dataset_pairs.csv \
  --runs config/llm_runs.json \
  --provider openai \
  --output data/processed/llm_runs/openai_predictions.csv
```

Claude만 실제 실행:

```bash
python3 src/llm_select_candidates.py \
  --dataset data/processed/pilot_dataset_pairs.csv \
  --runs config/llm_runs.json \
  --provider anthropic \
  --output data/processed/llm_runs/claude_predictions.csv
```

둘 다 실행:

```bash
python3 src/llm_select_candidates.py \
  --dataset data/processed/pilot_dataset_pairs.csv \
  --runs config/llm_runs.json \
  --output data/processed/llm_runs/predictions.csv
```

평가:

```bash
python3 src/evaluate_predictions.py \
  --dataset data/processed/pilot_dataset_pairs.csv \
  --predictions data/processed/llm_runs/predictions.csv \
  --out-dir data/processed/llm_runs/evaluation
```

## 출력 파일

LLM 선택 결과:

- `data/processed/llm_runs/predictions.csv`

사용량 및 예상 비용:

- `data/processed/llm_runs/usage.csv`

원본 LLM 응답:

- `data/processed/llm_runs/raw_responses/*.json`

LLM에 실제로 들어간 candidate prompt:

- `data/processed/llm_runs/prompts/*.json`

평가 결과:

- `data/processed/llm_runs/evaluation/metrics.csv`
- `data/processed/llm_runs/evaluation/summary.json`

## 실험 설계

최소 실험표는 다음처럼 구성한다.

| run_id | provider | model | prompt_id | 목적 |
|---|---|---|---|---|
| vpick_auto_20260705 | vpick | vpick_auto | N/A | 기업 baseline |
| openai_gpt4o_mini_v1 | openai | gpt-4o-mini | highlight_selector_v1 | 저비용 OpenAI LLM baseline |
| claude_haiku45_v1 | anthropic | claude-haiku-4-5-20251001 | highlight_selector_v1 | $25 제한 내 Claude baseline |

이후 prompt를 개선하면 `prompt_id`만 `highlight_selector_v2`처럼 늘린다.

## 해석 기준

모델 비교는 Final Score 하나로 끝내지 않는다.

- Coverage가 높고 IoU가 낮으면: 핵심 장면은 찾았지만 너무 길게 잡음
- Coverage가 낮으면: 아예 다른 장면을 고름
- Start Error가 낮고 End Error가 높으면: 시작 훅은 맞췄지만 끝 경계를 못 맞춤
- Top-3 안에 Core Hit이 있으면: LLM ranking 개선 여지가 있음

따라서 발표에서는 `Top-1 Core Hit`, `Top-1 IoU`, `Hit@3`, `Final Score`를 같이 보여준다.

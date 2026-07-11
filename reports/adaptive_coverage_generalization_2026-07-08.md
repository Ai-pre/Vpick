# Vpick 하이라이트 선택 파이프라인 개선 정리

작성일: 2026-07-08

## 핵심 결론

기존 `late_quota` 전략은 현재 파일럿 데이터에서는 성능이 좋았지만, 이름과 로직상 "후반 후보를 더 보장한다"는 인상이 있어 특정 영상셋에 맞춘 규칙처럼 보일 수 있었다.

그래서 최종 파이프라인 기본 전략을 `adaptive_coverage`로 교체했다. 이 전략은 gold timestamp를 보지 않고, 특정 영상 ID나 특정 시간대를 하드코딩하지 않는다. 대신 "롱폼 하나에서 여러 개의 숏폼 후보가 나올 수 있다"는 일반적인 문제 구조에 맞춰 Top5 후보가 서로 다른 이벤트와 시간대를 커버하도록 만든다.

## 최종 파이프라인

1. Vpick 장면 분석 결과 수집
   - scene description
   - speech transcript
   - scene start/end timestamp
   - speech start/end timestamp

2. Stage 1 후보 생성
   - Vpick scene 단위 후보를 기반으로 여러 하이라이트 후보를 만든다.
   - `scene_i`가 선택되면 `scene_{i-1}+scene_i` 같은 bridge 후보도 duration limit 안에서 추가한다.
   - 목적은 실제 숏폼이 한 장면보다 약간 앞의 setup을 포함하는 경우를 커버하는 것이다.

3. Trim 후보 확장
   - 30s, 45s, 60s, 75s 후보를 생성한다.
   - scene boundary window와 speech boundary window를 함께 만든다.
   - 정답 timestamp를 보고 자르는 것이 아니라, Vpick이 준 장면/대사 timestamp 안에서 가능한 후보 window를 만든다.

4. Deterministic rerank
   - 후보별 점수는 다음 신호를 조합한다.
   - speech density
   - duration fit
   - titleability
   - rank prior
   - speech boundary bonus
   - filler penalty
   - source band diversity

5. Longform Top5 slate 생성: `adaptive_coverage`
   - 겹치는 후보들을 이벤트 클러스터로 묶는다.
   - 각 이벤트에서 대표 구간을 고른다.
   - 너무 초반 intro에 후보가 몰리지 않게 일반적인 intro-safe cutoff를 적용한다.
   - 남은 후보를 긴 영상의 여러 시간대로 나눠 Top5가 서로 다른 이벤트/구간을 커버하게 한다.
   - 이 단계도 gold timestamp, 영상 ID, 특정 정답 위치를 사용하지 않는다.

6. LLM rerank
   - Top5 slate를 Claude Haiku 4.5와 GPT-4o mini에 넣어 최종 순서를 고른다.
   - 현재 config: `config/llm_rerank_top5_genre_lang.json`
   - 결과 비교 시 Claude와 GPT를 각각 별도 run으로 평가한다.

## 실제 LLM 평가 결과

평가 데이터:

- longform 3개
- gold short pair 11개
- Vpick baseline은 Vpick이 자동 생성한 숏폼 구간
- ours는 Vpick 장면 분석 데이터 위에서 만든 자체 후보 선택 파이프라인

| System | Model | Top1 Core | Core@3 | Tight@3 | IoU@3 | Core@5 | Tight@5 | IoU@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vpick baseline | Vpick auto | 0.091 | 0.091 | 0.000 | 0.026 | 0.091 | 0.000 | 0.046 |
| Previous ours | late_quota + Claude | 0.091 | 0.364 | 0.364 | 0.257 | 0.545 | 0.545 | 0.391 |
| New ours | adaptive_coverage + Claude | 0.182 | 0.455 | 0.455 | 0.313 | 0.545 | 0.545 | 0.393 |
| New ours | adaptive_coverage + GPT-4o mini | 0.182 | 0.364 | 0.364 | 0.257 | 0.545 | 0.545 | 0.391 |

해석:

- `adaptive_coverage`는 Vpick baseline보다 Core@5 기준 0.091 -> 0.545로 상승했다.
- `late_quota`와 비교하면 Core@5는 유지했고, Claude 기준 Top1/Core@3/IoU@3가 개선됐다.
- 즉, 성능을 잃지 않으면서 설명 가능한 일반화 전략으로 바꾼 상태다.

## 실제 Top5 후보

### NS7tSrMrWsc

| Rank | Start | End | Duration |
|---:|---:|---:|---:|
| 1 | 3:42 | 4:28 | 45.735 |
| 2 | 7:50 | 8:37 | 47.205 |
| 3 | 15:39 | 16:26 | 47.168 |
| 4 | 20:09 | 20:54 | 45.000 |
| 5 | 21:50 | 22:36 | 45.105 |

### OrCOflk2QmQ

| Rank | Start | End | Duration |
|---:|---:|---:|---:|
| 1 | 3:36 | 4:21 | 45.000 |
| 2 | 10:36 | 11:26 | 49.520 |
| 3 | 13:00 | 13:45 | 44.980 |
| 4 | 19:48 | 20:32 | 44.830 |
| 5 | 24:06 | 24:50 | 44.840 |

### heifaIjlSUc

| Rank | Start | End | Duration |
|---:|---:|---:|---:|
| 1 | 5:51 | 6:40 | 48.515 |
| 2 | 8:43 | 9:28 | 45.000 |
| 3 | 10:09 | 10:53 | 44.785 |
| 4 | 13:14 | 13:59 | 45.000 |
| 5 | 17:59 | 18:44 | 45.000 |

## 일반화 가능하다고 말할 수 있는 이유

- 영상 ID, gold timestamp, 특정 정답 구간을 코드에 넣지 않았다.
- "후반에 정답이 많다" 같은 데이터셋 전용 가정도 최종 기본 전략에서 제거했다.
- 후보 생성은 Vpick의 장면/대사 timestamp를 기반으로 한다.
- 후보 선택은 이벤트 중복 제거, speech boundary, duration fit, timeline coverage처럼 다른 영상에도 적용 가능한 신호만 사용한다.
- 한 롱폼에서 여러 숏폼이 나오는 현실을 반영해 Top5가 서로 다른 이벤트를 커버하도록 만들었다.

## 다음 개선 방향

1. 평가 데이터 확대
   - 현재 3개 롱폼/11개 pair라 성능 수치가 흔들릴 수 있다.
   - 최소 10개 롱폼, 30개 이상 gold short pair로 늘려야 한다.

2. 장르 라우터 추가
   - 예능/vlog는 setup-payoff, 반응, 대화 밀도 중심
   - 강연/정보형은 핵심 주장, 문제 제기, 결론성 문장 중심
   - 인터뷰는 질문-답변 완결성, 감정/논쟁 포인트 중심

3. Top5 평가 유지
   - 한 롱폼에서 숏폼 하나만 뽑는 문제가 아니므로 Top1만 보면 과제가 왜곡된다.
   - Core@5, Tight@5, mean IoU@5를 메인 지표로 두는 것이 적절하다.

4. Vpick baseline 재수집
   - Vpick 자동 숏폼 생성 결과가 여러 개 나올 수 있으므로 같은 롱폼에서 가능한 baseline 숏폼을 더 모아야 한다.
   - baseline도 Top5 기준으로 맞춰야 ours와 공정하게 비교할 수 있다.

## 산출물 위치

- 코드: `vpick/src/build_longform_slate.py`
- 실행 스크립트: `vpick/scripts/run_best_no_api_pipeline.sh`
- 실제 LLM 결과: `vpick/data/processed/best_pipeline_adaptive_coverage_real_llm`
- 평가 요약: `vpick/data/processed/best_pipeline_adaptive_coverage_real_llm/llm_rerank_top5/evaluation/summary.json`

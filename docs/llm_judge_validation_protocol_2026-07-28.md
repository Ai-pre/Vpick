# LLM-as-a-Judge 검증 프로토콜

## 결론

현재 채택한 구현은 `shortform_judge_v10_ko`의 후보 단독 Pointwise
Judge다. 다만 이 Judge를 조회수 예측기로 사용하지 않는다. 주 역할은 같은
롱폼에서 생성된 후보들의 편집·구간 선택 품질과 텍스트 근거상 콘텐츠 흡인력을
설명 가능하게 진단하고 최종 순서를 보정하는 것이다.

Judge의 검증은 다음 세 실험을 분리한다.

1. 성과 정합성: 94개 실제 Shorts의 채널 내 성과와 점수를 대조하는 보조 진단
2. 구간 정합성: 같은 롱폼의 Vpick 후보 중 실제 채택 구간의 순위를 재는 주 검증
3. 파이프라인 비교: 정답 주입 없이 Vpick, Ours, Ours+Judge를 비교하는 개선 실험

## 기존 실험의 위치

| 실험 | 확인한 것 | 현재 판단 |
|---|---|---|
| Codex v10, 60개 | 설명 가능한 8항목 Pointwise 채점 | 루브릭·집계식 유지 |
| Claude Opus 4.8 | 별도 모델의 성과 정합성 | 성과 Judge로 채택 실패 |
| mR3-Qwen3-8B | 오픈소스 Judge와 입력 방식 민감도 | 주 Judge 대체 근거 없음 |
| Gemini 멀티모달 | 영상·음성 추가의 효과 | 점수 포화·반복 불안정으로 주 검증에서 제외 |
| 동일 롱폼 hard negative 23쌍 | 같은 영상 안의 1:1 순서 | tie-aware 0.5435로 우연 수준, 새 다후보 실험의 선행 진단 |

멀티모달은 폐기한 것이 아니다. 조회수 노이즈가 큰 실험 1에 넣지 않고,
동일 롱폼 안에서 통제되는 실험 2의 후속 ablation으로만 사용한다.

## 데이터

- 전체 94개: `neg 30 / mid 34 / pos 30`
- 채널 6개, 고유 롱폼 85개
- 컷 기준: 하위 20% / 중간 60% / 상위 20%
- 분할: `dev 19 / locked_test 75`
- 분할 단위: `longform_id`
- dev/test 롱폼 중복: 0

`pos/neg`는 모델 입력이 아니다. 채점 후 성과 정합성과 실험 2의 층화 결과를
해석할 때만 결합한다.

## 멘토 피드백 반영

| 피드백 | 반영 내용 | 상태 |
|---|---|---|
| 양극단만으로는 지표가 부풀 수 있으므로 중간 데이터 필요 | 채널 백분위 20~80 구간 34개 추가 | 완료 |
| main/control 분할이 실제 코드에 적용되지 않음 | `longform_id` 그룹 기준 dev 19 / locked test 75 적용, 롱폼 중복 0 | 완료 |
| 가짜 판정자 sanity check가 보고서에 없음 | 무작위, 라벨 누설 상한선, POS/NEG shortcut, 자막 소스 shortcut을 각각 분리 보고 | 완료 |

중간군을 추가하면 모든 문제가 해결된다는 가정은 채택하지 않았다. 실제로
POS는 높게, NEG는 낮게 두고 MID만 무작위로 배치한 가짜 판정자도 locked
test에서 pooled Spearman `0.731`을 기록했다. 양극단 60개가 그대로 남아 있어
극단 구분만으로도 상관이 높아질 수 있기 때문이다.

## 실험 1: 성과 정합성

94개를 후보당 한 번씩 독립 채점하고 QWK, 3분류 정확도, 채널 중심화
Spearman, 채널별 macro Spearman을 계산한다. 이 실험은 조회수 예측 가능성을
확인하는 보조 진단이며 Judge 채택의 주 기준이 아니다.

현재 데이터 감사에서 다음 교락이 확인됐다.

```text
mid: yt-dlp 34 / Vpick 0
neg: yt-dlp 10 / Vpick 20
pos: yt-dlp 3 / Vpick 27
```

문맥은 앞뒤 각각 최대 200자로 통일했고, `longform_id` 그룹 분할도 완료했다.
그러나 자막 소스만 보는 얕은 대조군이 locked test 3분류 정확도 `0.64`를
기록했다. 따라서 소스별 표현 차이가 라벨의 대리 신호가 될 위험은 아직 크다.

또한 94개 인수인계 입력에는 v10이 요구하는 전체 `longform_overview`가 없다.
현재 정규화 파일은 실험 1 준비본이지만, 전체 overview를 동일 방식으로 복원하기
전에는 “프로덕션과 완전히 동일한 v10 검증”이라고 보고하지 않는다.

### 가짜 판정자 해석 수정

`pos/neg/mid` 라벨을 직접 아는 판정자는 94개에서도 높은 점수를 얻는다. 실제
계산에서 라벨 누설 판정자의 전체 Spearman은 `0.8970`이었다. 따라서 “중간
데이터를 넣으면 라벨을 아는 가짜 판정자가 0.2대로 붕괴한다”는 가정은 사용하지
않는다.

- 라벨 판정자: 의도적인 누설 상한선
- 무작위 판정자: 우연 기준
- 자막 소스 판정자: 실제 데이터 교락 탐지

이 세 대조군의 역할을 분리한다.

### 94개 잠금 테스트 결과

| 측정 | v10 Judge | 우연 기준 |
|---|---:|---:|
| 3분류 정확도 | 0.333 | 0.335 |
| QWK | 0.042 | 0.001 |
| 채널 중심화 Spearman | 0.096 | 0.002 |
| 채널별 macro Spearman | 0.014 | 0.001 |

v10 Judge는 실제 성과 순위를 복원하지 못했다. 이는 실패 결과이며 숨기지
않는다. 조회수에는 업로드 시점, 추천 노출, 제목·썸네일 등 현재 텍스트 입력에
없는 요인이 섞이므로 이 Judge를 성과 예측기로 사용하지 않는다.

대조군 결과도 같은 결론을 지지한다.

- POS 높음 / NEG 낮음 / MID 무작위 가짜 판정자: pooled Spearman `0.731`,
  QWK `0.731`
- 자막 소스만 사용한 shortcut: 3분류 정확도 `0.640`
- 완전 라벨 oracle: 의도적 누설 상한선일 뿐 배포 가능한 Judge가 아님

따라서 성과 정합성은 Judge의 품질을 증명하는 주 지표가 아니라 데이터와
지표가 얼마나 쉽게 속는지 확인하는 보조 진단이다.

## 실험 2: 구간 정합성

각 실제 Shorts마다 같은 롱폼의 Vpick 자동 후보를 모으고, 실제 채택 구간과
IoU가 0.5 이상인 후보가 없을 때만 정답 구간을 추가한다. 모든 후보는 같은
롱폼 scene 데이터에서 설명·대사·전후 문맥을 조립한다.

저장된 Vpick 결과는 영상마다 후보 수가 다르다. 현재 구축 결과는:

- 평가 가능 풀: 16개
- 후보: 98개
- 풀 크기: 5개 2풀, 6개 12풀, 7개 1풀, 9개 1풀

그러므로 우연 Hit@1을 일괄 `1/9`로 두지 않는다. 각 풀의 실제 후보 수와 정답
동치 후보 수로 정확한 우연 Hit@1·Hit@3·MRR을 계산한 뒤 평균한다.

동점은 candidate ID로 임의 해소하지 않는다. 정답 최고 점수의 동점 집단 안에서
무작위 순서를 가정한 기대 credit을 사용한다.

### 98개 블라인드 직접 평가 결과

| 측정 | v10 Judge | 풀 크기별 정확 우연 | 한쪽 우연검정 p |
|---|---:|---:|---:|
| Hit@1 | 0.125 | 0.166 | 0.6792 |
| Hit@3 | **0.719** | 0.498 | **0.0424** |
| MRR | 0.414 | 0.406 | 0.4351 |

Hit@3의 풀 단위 bootstrap 95% 구간은 `0.500~0.906`이다. 실제 채택 구간을
상위 3개 후보로 압축하는 신호는 확인됐지만, Hit@1과 MRR은 우연을 넘지
못했다. 따라서 현재 결과가 지지하는 용도는 **Top3 후보 압축 및 진단**이며,
최종 1개 자동 선택기로 검증됐다고 말하지 않는다.

이 결과에는 다음 제한이 있다.

1. 16개 풀 모두 저장된 Vpick 후보가 gold IoU 0.5를 넘지 못해 실제 구간을
   별도로 추가한 Vpick-miss 표본이다. Vpick Hit@K 0은 이 표본의 구성 특성이며,
   비편향 head-to-head 성능값이 아니다.
2. 명세는 후보당 독립 요청을 요구하지만 이번에는 한 세션에서 후보를 한 건씩
   직접 평가했다. 라벨·성과는 비공개였으나 프로덕션 동등 실행은 아니다.
3. 94개 중 저장된 연속 Vpick 후보가 충분한 16개 풀만 평가했다.
4. 성과 bucket은 pos 14, mid 1, neg 1이므로 bucket별 결론을 내릴 수 없다.

## 실험 3: 파이프라인

이 단계에는 정답 구간을 주입하지 않는다.

```text
Vpick auto
vs Ours 후보생성 + 리랭커
vs Ours 후보생성 + 리랭커 + v10 Judge
```

세 번째 행을 추가해야 Judge의 실용적 기여를 말할 수 있다. 저장소의 현재
Vpick/Ours summary와 인수인계 문서가 보고하는 표본 수가 서로 다르므로, 최종
보고 전에 동일 pair manifest로 다시 집계한다.

## 실행

94개 정규화 및 그룹 분할:

```bash
python src/prepare_judge_validation_94.py \
  --labels /private/vpick_goldlabel_final_PRIVATE.csv \
  --texts /private/vpick_short_subtitles_final_plain.csv \
  --output-dir data/private/judge_validation_94
```

실험 1 대조군:

```bash
python src/evaluate_judge_performance_consistency.py \
  --targets data/private/judge_validation_94/validation_targets_94_PRIVATE.csv \
  --output data/private/judge_validation_94/performance_controls_94.json
```

실험 2 입력 생성:

```bash
python src/build_within_video_judge_eval.py \
  --labels /private/vpick_goldlabel_final_PRIVATE.csv \
  --output-dir data/private/judge_within_video_v1
```

98개 직접 Pointwise 평가 컴파일:

```bash
python src/compile_codex_direct_v10.py \
  --candidates data/private/judge_within_video_v1/within_video_candidates_blind.jsonl \
  --dimensions data/private/judge_within_video_v1/codex_direct_v10_dimensions_98.csv \
  --config config/shortform_judge_v10_opus.json \
  --output-dir data/private/judge_within_video_v1 \
  --output-prefix codex_direct_v10_within_video_98 \
  --run-id codex_direct_shortform_judge_v10_within_video_98 \
  --source-salience-policy evaluated_from_compact_longform_overview \
  --execution-mode direct_single_session
```

구간 정합성 계산:

```bash
python src/evaluate_within_video_judge.py \
  --targets data/private/judge_within_video_v1/within_video_pool_targets_PRIVATE.csv \
  --scores data/private/judge_within_video_v1/codex_direct_v10_within_video_98_scores.csv \
  --output-dir data/private/judge_within_video_v1/codex_direct_v10_validation
```

## 다음 게이트

1. 저장된 Vpick 후보를 더 확보해 16개 Vpick-miss 편향 표본을 확대한다.
2. 후보 순서를 셔플하고 후보당 독립 요청으로 프로덕션 동등 locked test를
   새로 만든다.
3. dev 풀에서 Top1 실패 사례와 동점 원인을 분석하되 현재 16개 결과에는
   프롬프트를 사후 맞추지 않는다.
4. Top3 안의 최종 선택은 동점 후보만 pairwise로 비교하는 사전 고정 규칙을
   별도 dev에서 시험한다.
5. 텍스트 baseline 이후 같은 실험 2에서만 컷 빈도·오디오 특징 ablation을
   추가한다.

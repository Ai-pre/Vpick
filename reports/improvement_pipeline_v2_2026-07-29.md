# 하이라이트 선택 개선 파이프라인 v2

## 결론

최종 개발 방식은 Adaptive Coverage가 만든 Top5 중 상위 4개를 보존하고,
구조 기반 후보 점수 75%와 완결성 중심 Listwise Judge 점수 25%로 선정한
비중복 후보 1개를 추가한다.

Vpick baseline과 동일하게 비교할 수 있는 17개 롱폼, 18개 Gold pair에서
Adaptive Coverage의 Core@5는 50.0%였고, 최종 결합 방식은 55.6%였다.
Best IoU@5는 0.361에서 0.405로 상승했다.

이 결과는 일부 Gold 타임스탬프를 이미 확인한 상태에서 수행한 비블라인드
개발 결과다. 신규 롱폼 외부 Holdout 성능으로 표현하지 않는다.

## 입력

- Vpick scene timestamp와 description
- scene 내부 transcript와 speech timestamp
- 후보 직전·직후 문맥
- 익명 longform/candidate ID

채널명, 조회수, 좋아요, 성과 라벨, 실제 Shorts URL과 Gold 타임스탬프는
후보 생성 및 LLM Judge 입력에 사용하지 않는다.

## 후보 생성

```text
Vpick scene
  -> 단일 scene seed
  -> scene_{i-1} + scene_i bridge
  -> 30·45·60·75초 trim window
  -> scene·speech boundary 보정
  -> 동일 시간 구간 제거
  -> 유사 사건 대표 후보 압축
```

### 유사 후보 압축

1. 시작·종료 시각이 1ms 단위로 같은 후보를 제거한다.
2. 짧은 후보 기준 시간 겹침이 약 55% 이상이거나 같은 Vpick scene에서 생성된
   후보를 같은 사건 변형으로 묶는다.
3. 규칙 기반 점수, 45초 내외 길이 적합성, speech/scene 경계 자연스러움을
   기준으로 대표 후보 하나를 남긴다.

이 단계는 같은 사건의 30초·45초·60초 버전이 후보 수를 부풀리는 것을 막는다.

## Adaptive Coverage

롱폼을 5개 시간 구간으로 나누고 각 구간에서 규칙 기반 이벤트 점수가 가장
높은 대표 후보를 하나씩 선택한다. 따라서 초반 설명 장면이나 한 사건에
Top5가 몰리지 않고 원본 전반을 탐색한다.

Adaptive Coverage 단독 결과를 구조 기반 baseline으로 사용하며 Top5 중
상위 4개를 최종 방식의 anchor로 보존한다.

## Listwise Judge

프롬프트:
`prompts/hierarchical_multislate_listwise_v2_ko.md`

후보별 평가축:

| 평가축 | 가중치 |
|---|---:|
| 초반 명확성·흡인력 | 0.15 |
| 사건·반응·변화 | 0.25 |
| 전개·회수 | 0.20 |
| 독립적 이해 | 0.15 |
| 시작·종료 경계 | 0.15 |
| 제목화 가능성 | 0.10 |

```text
raw_selection_score
= (0.15×opening_clarity_pull
   +0.25×event_reaction_change
   +0.20×progression_payoff
   +0.15×self_contained
   +0.15×boundary_integrity
   +0.10×titleability) / 4
```

### 완결성 게이트

`progression_payoff`, `self_contained`, `boundary_integrity`에 적용한다.

| 조건 | 배율 |
|---|---:|
| 세 축 모두 기준 충족 | 1.00 |
| 1점 이하 축 1개 | 0.80 |
| 1점 이하 축 2개 이상 | 0.65 |
| 한 축이라도 0점 | 0.50 |

흥미로운 대사가 있더라도 결말 전에 끊기거나 후보 밖 문맥 없이는 이해할 수
없는 구간이 상위에 오르는 것을 막는 하드 제약에 가까운 감점이다.

## 최종 Top5 알고리즘

```text
1. Adaptive Coverage Top5 생성
2. AC rank 1~4 고정
3. 전체 후보를 5개 시간 구간으로 나눔
4. 혼합 점수 = 구조 기반 후보 점수 75% + 완결성 게이트 Judge 점수 25%
5. 각 구간의 혼합 점수 상위 후보를 전체 순위화
6. AC Top4와 짧은 구간 기준 58%를 초과해 겹치는 후보 제외
7. 남은 최고 후보 1개를 rank 5로 추가
8. 적격 후보가 없으면 기존 AC 후보로 보충
```

58%는 이론적으로 고정된 상수가 아니라 현재 개발 데이터에서 같은 사건의
중복과 인접 사건의 다양성 사이를 조정한 휴리스틱이다. 발표와 사용자 설명에서는
"약 60% 초과 중복 제거"로 해석한다.

## common18 결과

| 방식 | Core@1 | Core@3 | Core@5 | Tight@5 | Best IoU@5 |
|---|---:|---:|---:|---:|---:|
| Vpick baseline | 0.000 | 0.056 | 0.056 | 0.056 | 0.060 |
| Judge-only + MMR | 0.111 | 0.111 | 0.222 | 0.222 | 0.181 |
| 75:25 혼합 랭킹 Top5 | 0.222 | 0.278 | 0.444 | 0.444 | 0.307 |
| Adaptive Coverage | 0.167 | 0.444 | 0.500 | 0.500 | 0.361 |
| **AC Top4 + 혼합 후보 1개** | **0.167** | **0.444** | **0.556** | **0.556** | **0.405** |

Judge가 Top5 전체를 선택하지 않는 이유는 Judge-only Core@5 22.2%가
Adaptive Coverage의 50.0%보다 낮았기 때문이다. Judge는 구조 기반 선택기를
대체하지 않고 놓친 후보를 제한적으로 보강한다.

## 해석 제한

- 비교 범위는 Vpick baseline을 확보한 common18이다.
- 최종 방식은 기존 9개 hit을 유지하면서 `G016` 한 건을 추가했다.
- 25% Judge 가중치와 58% 중복 임계값은 일반적으로 최적이라고 주장할 수 없다.
- 실제 영상·음성·썸네일이 없는 선택 전 후보이므로 시각·음성 효과를 직접
  평가하지 않는다.
- 신규 롱폼과 신규 숏폼으로 구조·가중치를 고정한 외부 Holdout이 필요하다.

## 주요 파일

- 설정: `config/best_improvement_pipeline.json`
- 후보 생성: `src/build_longform_slate.py`
- Listwise 적용: `src/apply_hierarchical_listwise_results_v1.py`
- 5구간 혼합 랭킹: `src/select_intrinsic_v2_coverage.py`
- Top4 보존·후보 보강: `src/augment_b2_with_intrinsic.py`
- 프롬프트: `prompts/hierarchical_multislate_listwise_v2_ko.md`
- 공개 지표: `results/final/improvement_metrics.csv`

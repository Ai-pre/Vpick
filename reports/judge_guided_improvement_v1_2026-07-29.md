# Judge-Guided Highlight Improvement v1

## 결론

개선 파이프라인에는 평가 프롬프트만 쓰는 것이 아니라 역할이 다른 두 구성요소를
함께 사용한다.

1. **Shortform Judge v10**은 편집·구간 선택 품질과 텍스트에서 관찰되는 콘텐츠
   흡인력을 절대 기준으로 진단한다.
2. **Performance Calibrator v14**는 익명 설명·자막·구조 정보로 후보의 상대 성과
   잠재력을 보정한다.
3. 최종 선택은 두 점수의 영상 내 순위 백분위를 고정 `50:50`으로 결합한다.

다만 이번에 구현한 첫 신규 OOTB 골드 1건에서는 어느 방식도 Top5 정답 일치를
만들지 못했다. 따라서 **파이프라인 구현은 완료됐지만 Judge가 개선에 유효하다는
검증은 아직 통과하지 못했다.** 이 결과를 보고 가중치나 후보군을 다시 맞추지
않았으며, 다음 신규 홀드아웃에도 같은 설정을 적용해야 한다.

## 알고리즘

```text
Vpick scenes
  -> 시간대 분할 Stage 1 후보
  -> 이전 장면 bridge + 30/45/60/75초 경계 확장
  -> 결정론 점수·중복 제거·시간대 커버리지로 후보군 고정
  -> 후보별 익명 Vpick 설명·자막·전후 문맥 생성
     ├─ Pointwise Judge v10
     │    editorial 4축 + engagement 4축 -> 0~100
     └─ v14 performance calibrator
          익명 텍스트·구조 -> raw relative score
  -> 롱폼 내부 tie-aware 평균 순위 백분위로 각각 정규화
  -> 0.5 * Pointwise percentile + 0.5 * v14 percentile
  -> TopK
  -> 모든 채점 종료 뒤 외부 gold timestamp 결합
  -> Core@K / Tight@K / IoU@K 평가
```

후보 생성기는 모든 비교 방식에 동일하다. 채널명, 조회수, 좋아요, 성과 라벨,
숏폼 URL, 골드 시작·종료 시각은 후보 생성과 채점 입력에서 제외한다.

## 비교군

| 방식 | 역할 |
|---|---|
| Deterministic baseline | 기존 휴리스틱 리랭킹 순위 |
| Pointwise-only | v10 평가 기준만 사용 |
| v14-only | 성과 보정기만 사용 |
| Hybrid 50:50 | 두 점수의 영상 내 순위 백분위를 고정 결합 |

Judge의 점수 자체를 정답으로 사용하지 않는다. 다음 외부 비교로만 기여도를
판단한다.

- `Pointwise-only > deterministic`: LLM 평가 기준의 선택 기여
- `Hybrid > v14-only`: v14에 추가된 Pointwise Judge의 순증 기여
- `Hybrid > deterministic`: 전체 Judge 결합 파이프라인의 개선 기여

## 첫 신규 OOTB 파일럿

- 신규 롱폼: 1개
- 성과가 확인된 연속 편집 숏폼 gold: 1개
- gold를 주입하지 않고 만든 후보: 14개
- 근거 입력: 후보마다 동일한 Vpick scene description·transcript
- 후보군 oracle: gold와 겹치는 후보가 존재하며 최대 IoU `0.581`

| 방식 | gold 후보 순위 | Core@5 | Tight@5 | Best IoU@5 |
|---|---:|---:|---:|---:|
| Deterministic baseline | 7 | 0 | 0 | 0.000 |
| Pointwise-only | 8 | 0 | 0 | 0.000 |
| v14-only | 14 | 0 | 0 | 0.000 |
| Hybrid 50:50 | 12 | 0 | 0 | 0.000 |

Pointwise Judge는 실시간 주식 손익의 갈등·반전이 큰 후반 구간을 높게 평가했다.
실제 성과 gold인 앞쪽 종목 질의응답 구간은 독립 이해와 정보 가치는 있었지만
텍스트만 보면 변화와 경계 완결성이 상대적으로 약해 8위였다. v14는 그 구간을
14위로 두어 현재 artifact가 공개 숏폼 간 성과 순서를 학습한 결과를 임의의
롱폼 후보 리랭킹으로 바로 전이하지 못한다는 문제가 드러났다.

## 해석 원칙

- 이 실험은 `n=1`이므로 성공률 추정이나 모델 기각의 최종 근거가 아니다.
- 기존 16개 풀에서 v14가 높게 나온 결과는 v14 개발 데이터와 실제 채택 후보가
  겹쳐 재노출 가능성이 있으므로 통합 smoke test일 뿐 검증 수치로 사용하지 않는다.
- 이번 신규 후보 입력은 블라인드였지만 직접 채점자가 대화 이력에서 gold 정보를
  접한 상태이므로 완전한 독립 외부 평가라고 과장하지 않는다.
- 결과를 본 뒤 50:50 가중치, 후보 수, 평가 기준을 바꾸지 않는다.

## 다음 검증

신규 롱폼과 신규 연속 편집 숏폼을 추가해 동일 설정으로 최소 4개 gold를 먼저
확보한다. 성과가 낮은 숏폼은 후보 선택 정답이 아니므로 주 구간 일치도에는 넣지
않고 성과 보정기 순위 진단에만 사용한다. 최소 보고 단위는 롱폼별 결과와
macro 평균이며, 샘플이 늘면 longform bootstrap 신뢰구간을 추가한다.

Judge 채택 조건은 다음과 같다.

1. Pointwise-only가 deterministic baseline보다 Core@5 또는 Tight@5를 높인다.
2. Hybrid가 v14-only보다 높아 Pointwise Judge의 순증 기여가 확인된다.
3. Hybrid가 deterministic baseline보다 높고, 특정 한 영상에만 의존하지 않는다.
4. 조건을 만족하지 않으면 프로덕션 선택기는 기존 deterministic 방식을 유지하고
   Judge는 진단·후보 설명 용도로만 사용한다.

## 재현

```bash
POINTWISE_SCORES=/path/to/pointwise_scores.csv \
bash scripts/run_judge_guided_improvement.sh \
  /path/to/gold.csv \
  /path/to/frozen_slate.csv \
  /path/to/vpick_scenes \
  /path/to/output
```

- 프로토콜: `config/judge_guided_improvement_v1.json`
- Pointwise 기준: `prompts/shortform_judge_v10_ko.md`
- 공개 파일럿 결과:
  `results/judge_guided_improvement_v1/ootb_fresh_pilot_PUBLIC.json`

# Vpick 평가 데이터 정의 v1

- 작성일: 2026-07-23
- 기준 데이터: `data/processed/gold_reference_relabelled_2026-07-23.csv`
- 목적: 모델을 고르기 전에 정답 단위, 라벨 의미, 평가별 사용 범위를 고정한다.

## 1. 데이터 한 건의 정의

데이터 한 건은 다음 네 요소로 구성된 `long-short reference pair`다.

1. 실제 공개된 원본 롱폼
2. 해당 롱폼에서 파생된 실제 공개 숏폼
3. 숏폼에 대응하는 원본 롱폼의 시작·종료 구간
4. Vpick이 제공한 장면 설명, 대사, 인물 및 타임스탬프

같은 롱폼에서 여러 숏폼이 만들어졌다면 각 숏폼을 별도 reference pair로 기록한다. 따라서 한 롱폼은 여러 gold interval을 가질 수 있다.

## 2. 성과 라벨의 정의

성과 라벨은 콘텐츠 품질의 절대 정답이 아니라 같은 채널 안에서의 상대적 성과다.

| `performance_label` | 정의 | 의미 |
|---|---|---|
| `pos` | 채널 내 조회수 백분위 75 이상 | 성과가 검증된 숏폼 |
| `neg` | 채널 내 조회수 백분위 25 이하 | 저성과 대조 사례 |
| `unlabeled` | 백분위 25 초과 75 미만 | 성과 신호가 애매한 사례 |

`pos` 숏폼이 원본에서 가져간 구간을 하이라이트 선택의 정답 신호로 사용한다. `neg`는 나쁜 콘텐츠라는 정답이 아니라 Judge의 성과 구분력을 확인하는 control이다.

## 3. 매핑 라벨의 정의

성과 라벨과 원본 구간 매핑 가능성은 별도 축으로 관리한다.

| `alignment_status` | 의미 | 단일 구간 평가 사용 |
|---|---|---|
| `continuous` | 숏폼이 원본의 한 연속 구간과 대응 | 사용 |
| `light_edit` | 소수의 짧은 생략이 있으나 대표 시작·종료 구간을 정할 수 있음 | 완화 평가에 사용 |
| `heavy_edit` | 다수 구간 재배열·결합 또는 긴 생략 | 제외 |
| `missing_subtitle` | 자막 근거 부족 | 제외 |
| `insufficient_alignment` | 대응 구간 신뢰 부족 | 제외 |

엄격 평가에서는 `continuous`만 사용한다. 완화 평가에서는 `continuous + light_edit`를 사용하고 두 결과를 함께 보고한다.

## 4. 평가 목적별 데이터셋

하나의 CSV를 모든 평가에 그대로 사용하지 않는다.

### A. 성과 라벨 데이터셋

- 대상: 전체 60개
- 구성: pos 30, neg 30
- 조건: 채널 내 백분위 스냅샷 검증 완료
- 용도: Judge 점수와 실제 성과의 외적 정합성 분석
- 지표: pos-neg AUC, 백분위 Spearman, 좋아요율 Spearman

### B. 엄격한 시간 구간 gold

- 대상: `pos AND continuous`
- 현재 수량: 17개
- 용도: Vpick baseline과 Ours의 가장 신뢰도 높은 구간 선택 성능 비교

### C. 완화된 시간 구간 gold

- 대상: `pos AND alignment_status IN {continuous, light_edit}`
- 현재 수량: 20개
- 용도: 짧은 생략 편집까지 허용한 구간 선택 성능 비교

### D. 매핑 가능 성과 control

- 대상: `neg AND alignment_status IN {continuous, light_edit}`
- 현재 수량: 28개
- 용도: 인간 쌍대평가 및 Judge의 성과 구분력 검증

### E. Judge 입력 가능 데이터셋

- 대상: Vpick 장면 설명 또는 복원 가능한 transcript가 있는 후보
- 조건: 입력 근거가 없으면 `abstain`
- 용도: LLM-as-a-Judge 평가

시간 구간 gold에 포함된다고 자동으로 Judge 입력에도 포함되는 것은 아니다. 매핑은 정확하지만 대사·장면 근거가 부족할 수 있기 때문이다.

## 5. 현재 데이터 현황

| 구분 | pos | neg | 합계 |
|---|---:|---:|---:|
| 전체 성과 라벨 | 30 | 30 | 60 |
| continuous | 17 | 21 | 38 |
| light_edit | 3 | 7 | 10 |
| heavy_edit | 4 | 2 | 6 |
| missing_subtitle | 3 | 0 | 3 |
| insufficient_alignment | 3 | 0 | 3 |
| 완화된 매핑 사용 가능 | 20 | 28 | 48 |

따라서 발표에서 단순히 “gold 30개”라고 말하면 안 된다.

- 성과 기반 gold 후보: 30개
- 엄격한 시간 구간 gold: 17개
- 완화된 시간 구간 gold: 20개

## 6. 인간 평가의 역할

인간 평가는 성과 라벨을 대체하지 않고, 성과가 콘텐츠 자체에서 발생했는지 확인하는 보강 근거다.

완화된 시간 구간 gold 20개와 채널·포맷·길이가 비슷한 control 20개를 1:1로 매칭한다. 최대 20쌍을 3명이 독립적으로 평가한다.

평가자는 조회수, 좋아요 수, pos/neg를 보지 않고 다음만 판단한다.

- 어느 구간이 독립 숏폼 하이라이트로 더 적합한가
- 차이가 없거나 판단 근거가 부족한가
- 판단 확신도는 1~5 중 얼마인가

성과와 인간 판단의 관계는 다음처럼 기록한다.

| 상태 | 조건 |
|---|---|
| `strong_gold` | pos 구간을 인간 다수도 선호 |
| `strong_control` | neg 구간을 인간 다수가 비선호 |
| `conflicted` | 성과 라벨과 인간 선호가 반대 |
| `unreviewed` | 인간 평가 미실시 |

## 7. LLM-as-a-Judge의 위치

LLM Judge는 정답을 만드는 모델이 아니다. Vpick 장면 분석과 transcript만 보고 고정된 기준으로 후보 품질을 채점하는 평가 도구다.

Judge 검증은 세 질문으로 나눈다.

1. 같은 입력을 반복 평가해도 결과가 안정적인가
2. 인간 쌍대선호와 Judge 순위가 일치하는가
3. Judge 점수가 성과 pos/neg 및 백분위와도 관련되는가

사전학습 비디오 engagement·highlight 모델은 로컬 영상 파일을 요구하고 현재 입력 조건과 맞지 않으므로 본 평가체계 범위에서 제외한다.

## 8. 구간 선택 평가 지표

파이프라인은 롱폼별 Top K 후보와 해당 롱폼의 모든 gold interval을 비교한다.

- Best IoU@K
- Recall@K at IoU 0.3
- Recall@K at IoU 0.5
- Center Hit@K
- 시작·종료 Boundary MAE

한 롱폼에 여러 gold가 있으면 각 gold가 Top K 중 하나와 매칭됐는지 계산한다. 동일 후보가 여러 gold를 중복 회수한 것으로 세지 않도록 일대일 최대 매칭을 사용한다.

## 9. 고정 순서

1. 성과 라벨과 매핑 라벨을 분리한다.
2. 엄격 gold 17개와 완화 gold 20개를 확정한다.
3. gold 20개와 control 20개를 매칭한다.
4. 인간 3인의 블라인드 쌍대평가를 완료한다.
5. 개발·테스트를 `long_video_id` 기준으로 분리한다.
6. 그 뒤에만 Judge 프롬프트를 개발셋에서 조정한다.
7. 고정 테스트셋에서 Judge 신뢰도와 인간·성과 정합성을 최종 계산한다.
8. 검증된 Judge로 Vpick baseline과 Ours를 평가한다.

## 10. 한 문장 정의

성과 상위 숏폼이 원본에서 선택한 매핑 가능한 구간을 gold reference로 정의하고, 저성과 구간은 control로 사용하며, 인간 평가와 LLM Judge는 이 약한 정답 신호의 신뢰도를 보강하고 검증한다.

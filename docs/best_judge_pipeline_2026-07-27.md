# Vpick Best Judge Pipeline

## 1. 현재 결론

현재 채택한 평가체계는 **원본 문맥을 포함한 후보 단독 Pointwise
LLM-as-a-Judge**다. 후보의 편집·구간 선택 품질과 콘텐츠 자체의 흡인 잠재력을
분리해 설명 가능한 점수를 만든다.

이 Judge는 실제 조회수나 좋아요를 예측하는 모델로 검증되지 않았다. 채널 내
성과 백분위와 POS/NEG는 채점 입력이 아니라, 채점 완료 후 외부 타당도를 점검하는
정답 신호로만 사용한다.

## 2. 확정 데이터

- 파일:
  `data/processed/goldlabel_60_replaced_v6_channel_normalized_2026-07-23.csv`
- 공개 롱폼-숏폼 페어: 60개
- 채널: BDNS 5, OOTB 12, 숏박스 14, 안원잘부 11, 워크맨 14, 피식대학 4
- 중복 숏폼: 0개
- POS/NEG: 각 30개
- 성과 신호: 같은 채널 내 조회수 백분위

60건 모두 롱폼 ID와 타임스탬프가 있고 Judge용 설명·자막 입력이 완성됐다.
다만 CSV의 `usable_for_gold`는 Judge 사용 가능 여부가 아니라, 이전 단계의 엄격한
원본 연속 구간 정렬 검증 필드다. 이 값은 10건만 `yes`이며 나머지 공란을 임의로
수정하지 않았다. 따라서 현재 60건은 Judge 평가·성과 타당도 진단에는 사용하지만,
향후 구간 IoU 정답셋으로 사용할 때는 정렬 검증을 별도로 다시 적용해야 한다.

POS/NEG는 절대적인 재미의 정답이 아니다. 채널 내 상위·하위 성과 구간을
검증하기 위한 파생 라벨이다. 메인 평가에서는 연속형 백분위를 우선하고,
POS/NEG AUC는 보조 지표로만 본다.

## 3. 증거 입력

모델에는 후보 하나씩 다음 정보만 전달한다.

1. 익명 후보 ID와 시작·종료 시각
2. 해당 숏폼 구간을 설명하는 1~2문장 후보 설명
3. 후보 구간 자막
4. 시작·종료 경계를 판단할 직전·직후 문맥
5. 원본에서 후보의 중요도를 판단할 롱폼 장면 개요
6. 시각 장면 설명 제공 여부

채널명, URL, 제목, 조회수, 좋아요, 성과 백분위, POS/NEG, 후보 생성 시스템은
모델 입력에서 제외한다. Vpick 분석이 없는 롱폼은 `yt-dlp` 타임스탬프 자막으로
대체하고, 시각 증거가 없다는 사실을 명시한다. 롱폼 전체 설명을 후보 설명으로
재사용하지 않는다.

## 4. Fable 5 피드백과 반영

| 문제 | 최종 반영 |
|---|---|
| 총점 집계식이 실행 전에 고정되지 않음 | 8개 항목과 50:50 집계식을 프롬프트·설정에 고정 |
| 독립 채점과 전체 점수 분포 제약이 모순 | 점수 분포·상위 비율 지시 삭제 |
| 불확실할 때 중간값을 주도록 유도 | 관찰 근거가 기우는 쪽을 채점하고 불확실성은 confidence로 분리 |
| 자막·경계 품질이 콘텐츠 점수에 섞임 | evidence는 품질 관리에만 사용하고 총점에서 제외 |
| abstain 처리 불명확 | 순위·상관·AUC에서 제외하고 abstain율 별도 보고 |
| 여러 후보를 한 요청에 넣어 암묵 비교 발생 | 후보 1개당 요청 1회 |
| 점수 뒤에 이유를 쓰는 사후 정당화 | 최상위 reason과 차원별 reason을 점수보다 먼저 출력 |
| saliency와 세부 항목의 중복 계산 | 편집 4축과 콘텐츠 4축으로 역할 분리 |
| 롱폼 메타 설명이 후보 설명으로 오인됨 | 후보 구간 자막·장면만으로 short-specific description 생성 |

## 5. 평가 기준과 고정식

편집·구간 선택 품질:

- 원본 내 중요도
- 독립적 이해 가능성
- 전개와 회수
- 시작·종료 경계

콘텐츠 흡인 잠재력:

- 초반 주목도
- 변화·발견·반전
- 감정 또는 정보 이득
- 기억에 남는 구체성

각 항목은 0~4점이다.

```text
editorial_score_100 = 25 * mean(편집 4항목)
engagement_score_100 = 25 * mean(콘텐츠 4항목)
judge_score_100 = 0.5 * editorial_score_100
                + 0.5 * engagement_score_100
```

근거 충분성, confidence, failure flags는 총점에 포함하지 않는다. 의미 복원이
불가능할 때만 abstain한다.

## 6. 검증 결과

Codex 블라인드 v10 평가:

- 평가: 60/60
- abstain: 0
- 평균: 75.9375
- 고유 점수: 18개
- 최대 동점 집단: 7개
- 채널 중심화 성과 상관: 0.0717
- 6채널 macro Spearman: 0.1089
- 표본이 안정적인 4채널 macro Spearman: 0.1384

따라서 평가 규칙과 출력 스키마는 구현했지만, 성과 예측 타당도는 확보하지
못했다.

POS/NEG를 유지한 V1~V5 프롬프트 실험에서도 채택 가능한 성과 Judge는 나오지
않았다. mR3의 가장 높은 pooled AUC는 V3의 0.5578이었지만 안정 채널 macro
AUC는 0.4785, 셀 내부 순서 정확도는 0.5371이었다. Codex 보조 재채점도 셀 내부
순서 정확도 최대 0.5457에 그쳤고, mR3-Codex 순위 상관도 최대 0.3656이었다.

이 결과는 POS/NEG를 없애서 실패한 것이 아니라, 현재 텍스트 중심 증거만으로
게시 성과를 안정적으로 복원하기 어렵다는 뜻이다.

## 7. 사용 원칙

현재 사용 가능:

- 후보 하나의 편집 품질 진단
- 후보 하나의 콘텐츠 흡인 잠재력 진단
- 동일 증거 정책 아래 후보 간 설명 가능한 점수 비교
- 실패 원인과 경계 문제 분석

현재 사용 금지:

- 조회수·좋아요 예측
- 채널 내 성과 순위 예측이 검증됐다는 주장
- POS/NEG 또는 성과 백분위를 모델 입력으로 사용
- Judge만으로 Ours가 Vpick보다 우수하다고 결론

## 8. 실행

패키지 무결성과 누출을 먼저 검사한다.

```bash
python src/audit_best_judge_pipeline.py
```

모델 호출 없이 요청 구성을 확인한다.

```bash
python src/run_shortform_judge_v9.py \
  --input results/judge_evaluation_v2_2026-07-27/short_candidate_descriptions_codex/candidates_blind_short_description_60.jsonl \
  --config config/shortform_judge_v10_opus.json \
  --dry-run
```

Amazon 서버에서 실제 반복 평가를 실행한다.

```bash
BEST_JUDGE_REPEAT_COUNT=2 bash scripts/run_best_judge_pipeline.sh
```

`run_shortform_judge_v9.py`와 출력 파일명의 `v9`는 기존 실행기 호환을 위해
유지했다. 실제 적용 프롬프트와 설정은 v10이다.

## 9. 후속 검증 게이트

1. 같은 모델·프롬프트로 2회 독립 실행해 반복 Spearman과 MAE를 확인한다.
2. 평가자 2인의 기준 점수와 편집·흡인 축 정합성을 확인한다.
3. POS/NEG가 아닌 채널 내 연속 성과 백분위와의 상관을 별도 검증한다.
4. 제목·썸네일·시각·음성 정보를 추가한 실험은 텍스트 기준과 분리 비교한다.
5. 위 검증을 통과한 뒤에만 Ours와 Vpick의 후보 선택 개선 실험에 Judge를
   사용한다.

## 10. 핵심 산출물

- 프롬프트: `prompts/shortform_judge_v10_ko.md`
- 설정: `config/shortform_judge_v10_opus.json`
- 패키지 명세: `config/best_judge_pipeline.json`
- 최종 데이터: `data/processed/goldlabel_60_replaced_v6_channel_normalized_2026-07-23.csv`
- 블라인드 입력:
  `results/judge_evaluation_v2_2026-07-27/short_candidate_descriptions_codex/candidates_blind_short_description_60.jsonl`
- v10 결과:
  `results/judge_evaluation_v2_2026-07-27/codex_judge_v10_blind/`
- POS/NEG 프롬프트 진단:
  `reports/prompt_ablation_posneg_mr3_codex_2026-07-27.md`

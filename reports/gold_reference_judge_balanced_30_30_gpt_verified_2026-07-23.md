# 30:30 검증 데이터셋 GPT Judge 1차 진단 보고서

- 평가일: 2026-07-23
- 데이터셋: `data/processed/gold_reference_relabelled_2026-07-23.csv`
- 평가 기준: `prompts/shortform_reference_judge_v6_ko.md`
- 입력: Short transcript + 롱폼 앞뒤 문맥
- 결과: `results/gold_reference_judge_balanced_30_30_gpt/codex_single_verified/`

## 실행 상태

OpenAI API의 `gpt-5.6-terra` 자동 실행은 `429 insufficient_quota`로 시작하지 못했다. 동일한 블라인드 입력을 현재 Codex GPT 세션에서 1회 직접 채점한 결과이므로, 아래 결과는 API 반복실험을 대체하지 않는 예비 진단이다.

## 성과 라벨 검증

신규 neg 21개에 대해 채널별 공개 숏폼 48개와 대상 영상을 합친 48~52개 코호트의 조회수를 다시 수집했다. 21개의 채널 내 백분위는 1.0~16.3이었으며 모두 하위 25%에 속했다. 라벨 변경은 없고 최종 분포는 pos 30, neg 30이다.

## Judge 결과

| 지표 | 결과 |
|---|---:|
| 정상 채점 | 57/60 |
| abstain | 3/60 |
| pos 평균 Reference 점수 | 61.4224 |
| neg 평균 Reference 점수 | 69.1964 |
| pos 대 neg ROC-AUC | 0.4292 |
| 채널 백분위 Spearman | -0.0743 |
| 좋아요율 Spearman | 0.1069 |
| 인간 2인 공통 12개 Reference Spearman | 0.5708 |
| 인간 suitable 일치율 | 0.6667 |

## 해석

성과 라벨은 검증되었지만 Judge 점수는 성과와 거의 연관되지 않았다. 따라서 현재 Judge는 기업이 정한 성과 기반 정답 신호를 구분하지 못했다. 다만 `pos/neg`는 약한 정답이므로, 성과와 인간 판단이 합의한 고신뢰 subset을 만든 뒤 현재 프롬프트의 실패인지 제목·썸네일·노출 교란의 영향인지 분리해야 한다.

Judge의 채택 여부는 성과 정합성, 인간 블라인드 쌍대평가와의 정합성, 동일 모델 반복 신뢰도를 함께 보고 결정해야 한다. 현재 인간 비교는 2인, 12개 공통 후보뿐이고 모델 실행도 1회이므로 아직 검증 완료 상태가 아니다.

## 다음 단계

1. pos-neg를 채널·포맷·길이별로 30쌍 매칭하고 인간 3인의 블라인드 쌍대평가를 완료한다.
2. GPT API 할당량 복구 후 동일 입력을 2회 이상 실행한다.
3. 반복 Spearman 0.80, suitable 반복 일치율 0.80, 인간 쌍대선호 일치율 0.70, 고신뢰 subset AUC 0.70을 기준으로 통과 여부를 결정한다.
4. 통과한 평가 프롬프트만 Claude와 Gemini 교차 평가에 사용한다.
5. 전체 성과 AUC와 백분위 상관도 별도 결과로 반드시 보고한다.

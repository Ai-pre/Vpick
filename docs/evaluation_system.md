# Vpick 하이라이트 선택 평가체계

## 1. 이번 파일럿에서 확인한 기준점

파일럿 pair `P001`에서 Vpick 자동 숏폼은 총 5개가 생성되었고, 그중 1개가 실제 쇼츠 정답 구간을 포함했다.

- Gold 구간: `22:19 - 22:36`, 17.0초
- Vpick auto best: `22:18.95 - 23:18.95`, 60.0초
- Gold Coverage: `1.0000`
- Temporal IoU: `0.2833`
- Start Error: `0.05초`
- End Error: `42.95초`

이 결과가 중요한 이유는 하나다. 모델이 "좋은 장면의 시작점"은 거의 정확히 잡아도, 숏폼 길이를 길게 뽑으면 IoU는 낮아진다. 따라서 평가체계는 "핵심 장면을 포함했는가"와 "편집 경계를 정확히 맞췄는가"를 분리해야 한다.

## 2. 평가 단위

평가 단위는 long-short pair 1개다.

각 pair에는 다음 정보가 있어야 한다.

- long_video_url: 원본 롱폼 영상 URL
- short_video_url: 실제 업로드된 쇼츠 URL
- gold_start_sec / gold_end_sec: 쇼츠가 원본 영상에서 가져온 실제 구간
- short_views / short_likes: 쇼츠 성과 지표
- label_confidence: gold 구간 라벨 신뢰도
- vpick_project_id / vpick_asset_id: Vpick 장면 분석 API 연결 정보

모델 또는 Vpick의 출력은 다음 형태로 저장한다.

- pair_id
- run_id
- selector_type: `vpick_auto`, `llm_prompt`, `heuristic_baseline` 등
- prompt_id: LLM 프롬프트 버전
- model_name
- rank
- pred_start_sec / pred_end_sec
- selected_scene_ids
- confidence
- notes

## 3. 핵심 지표

### Gold Coverage

정답 쇼츠 구간 중 예측 구간이 포함한 비율이다.

`overlap(pred, gold) / gold_duration`

의미:

- `1.0`: 실제 쇼츠 핵심 구간을 전부 포함
- `0.0`: 실제 쇼츠 구간과 전혀 겹치지 않음

이 지표는 "바이럴 핵심을 놓치지 않았는가"를 본다.

### Temporal IoU

예측 구간과 정답 구간의 시간적 겹침 정확도다.

`overlap(pred, gold) / union(pred, gold)`

의미:

- 구간이 길게 늘어나면 낮아진다.
- 편집 경계가 정확할수록 높아진다.

이 지표는 "실제 쇼츠 길이와 경계를 얼마나 잘 맞췄는가"를 본다.

### Start Error

예측 시작점과 gold 시작점의 차이다.

`abs(pred_start_sec - gold_start_sec)`

파일럿에서는 Vpick auto best의 Start Error가 `0.05초`로 매우 좋았다.

### End Error

예측 종료점과 gold 종료점의 차이다.

`abs(pred_end_sec - gold_end_sec)`

파일럿에서는 Vpick auto best의 End Error가 `42.95초`로 컸다. 즉 시작점은 맞았지만 자동 길이가 길었다.

### Core Hit

핵심 장면을 맞췄는지 보는 이진 지표다.

기준:

- Gold Coverage >= `0.70`
- Overlap sec >= `min(5초, gold_duration * 0.5)`

Core Hit은 "일단 맞는 장면을 골랐는가"를 보는 지표다.

### Tight Hit

경계까지 잘 맞췄는지 보는 이진 지표다.

기준:

- Core Hit = true
- Temporal IoU >= `0.30`

Tight Hit은 "그 장면을 숏폼으로 잘라낼 만큼 정확히 골랐는가"를 보는 지표다.

## 4. 최종 점수

초기 실험용 최종 점수는 100점 만점으로 둔다.

`Final Score = 100 * (0.45 * Coverage + 0.30 * IoU + 0.15 * StartScore + 0.10 * LengthScore)`

구성:

- Coverage: 핵심 장면 포함 여부
- IoU: 경계 정확도
- StartScore: 시작점 정확도, `max(0, 1 - start_error / 10)`
- LengthScore: 길이 유사도, `min(pred_duration, gold_duration) / max(pred_duration, gold_duration)`

이 점수는 초반에는 Vpick/LLM 선택 결과를 빠르게 비교하기 위한 실험 점수다. 최종 발표에서는 Coverage, IoU, Start Error, End Error를 반드시 같이 보여주는 것이 좋다.

## 5. 비교군

최소 비교군은 4개로 둔다.

1. Vpick scene baseline
   - Vpick 장면 분석 결과 중 gold를 포함하는 장면
   - 모델이 아니라 Vpick 장면 분할 자체의 상한/한계 확인용

2. Vpick auto shortform
   - Vpick이 자동 생성한 숏폼
   - 기업 제공 baseline

3. LLM prompt baseline
   - Vpick scene JSON을 LLM에 넣고 직접 작성한 프롬프트로 top-k 후보 선택
   - 학생 파이프라인의 기본 모델

4. Improved LLM prompt
   - 후킹 강도, 갈등/반전, 대사 완결성, 길이 제약 등을 반영한 개선 프롬프트
   - 성능 개선 실험용

## 6. 정답 데이터셋 구축 방향

정답 데이터셋은 "조회수 높은 쇼츠가 원본 롱폼의 어느 구간에서 왔는가"를 사람이 라벨링해서 만든다.

추천 절차:

1. 같은 채널 안에서 long-short pair를 수집한다.
2. 쇼츠 제목/내용을 원본 롱폼에서 찾아 gold_start/end를 기록한다.
3. 라벨링 근거를 notes에 남긴다.
4. 쇼츠 조회수/좋아요 수를 기록한다.
5. 가능하면 같은 채널 안에서 조회수 분위수 또는 상대 순위를 계산한다.

조회수와 좋아요 수는 정답 구간을 찾는 보조 근거이지, 모델 채점의 직접 정답은 아니다. 모델 채점의 정답은 gold_start/end다.

## 7. 발표용 한 줄 정의

이 프로젝트의 평가체계는 실제 유튜브 롱폼-쇼츠 pair에서 추출한 gold 구간을 기준으로, Vpick/LLM이 선택한 후보 구간의 핵심 장면 포함도와 시간 경계 정확도를 분리 측정하는 체계다.

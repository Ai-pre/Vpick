# 재현 절차

## 1. 공개 범위

저장소만으로 다음을 확인할 수 있다.

- 최종 설정과 점수 공식
- Pointwise·Listwise 프롬프트
- 후보 생성·중복 제거·Adaptive Coverage 코드
- 성과 보정과 평가 코드
- 공개 가능한 최종 지표와 단위 테스트

과거 60개 개발 CSV는 실험 이력으로 남기되, 다음 자료는 공개 저장소에
포함하지 않는다.

- 원본 영상 파일
- Vpick 계정·API 토큰
- Vpick raw scene dump
- 최종 94개 평가 패키지의 후보별 Gold 타임스탬프와 성과 라벨
- 원시 LLM API 응답과 대규모 중간 후보 파일

## 2. 환경

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

API를 사용할 때만 환경변수를 설정한다.

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."
export VPICK_EMAIL="..."
export VPICK_PASSWORD="..."
```

## 3. 공개 릴리스 검사

```bash
python src/validate_final_release.py
python -m pytest -q
```

`validate_final_release.py`는 다음을 검사한다.

- 최종 manifest의 참조 경로
- Pointwise 및 Listwise 가중치 합
- 데이터 수량과 HIGH/MID/LOW 합
- 공개 지표와 설정 값 일치
- 금지된 비공개 경로가 최종 manifest에 포함되지 않았는지 여부

기본 테스트는 공개 저장소만으로 실행된다. 비공개 Gold와 생성된 실험 결과를
보유한 내부 환경에서는 다음 명령으로 통합 테스트를 별도 실행한다.

```bash
python -m pytest -q -m private_artifacts
```

## 4. 전체 평가체계 재현

비공개 94개 패키지와 독립 채널 성과 파일이 필요하다.

```bash
python src/build_codex_package_judge_v1.py
python src/collect_short_publish_dates.py
python src/evaluate_package_and_context_v1.py
python src/summarize_package_improvements_v1.py
python src/evaluate_salience_augmented_judge_v1.py
python src/tune_salience_augmented_weights_v1.py
```

최종 Pointwise 식:

```text
0.40 × change_or_surprise
+ 0.15 × title_packaging
+ 0.45 × thumbnail_packaging
```

성과 보정 입력:

```text
pointwise_score
+ log1p(independent_channel_historical_median_views)
+ log1p(upload_age_days)
```

## 5. 전체 선택 개선 재현

필요 입력:

```text
data/raw/vpick/{long_video_id}_scenes.json
비공개 Gold pair CSV
Vpick baseline 자동 숏폼 결과
```

주요 단계:

```bash
python src/expand_trim_windows.py --help
python src/build_hierarchical_multislate_v1.py --help
python src/build_hierarchical_listwise_batches_v1.py --help
python src/apply_hierarchical_listwise_results_v1.py --help
python src/select_intrinsic_v2_coverage.py --help
python src/augment_b2_with_intrinsic.py --help
python src/evaluate_predictions.py --help
```

실제 API 호출 없이 구조 선택기만 확인하려면 기존
`scripts/run_best_no_api_pipeline.sh`를 사용할 수 있다. 최종 Listwise Judge
실행은 provider별 runner를 사용한다.

```text
src/run_openai_listwise_v2.py
src/run_anthropic_listwise_v2.py
src/run_gemini_listwise_v2.py
```

## 6. 결과 검증

최종 공개 지표:

```text
results/final/judge_metrics.json
results/final/improvement_metrics.csv
```

개발 결과와 외부 Holdout 결과를 같은 이름으로 덮어쓰지 않는다. 신규
Holdout은 별도 release ID와 결과 디렉터리를 사용한다.

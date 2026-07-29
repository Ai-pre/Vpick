# 60개 Judge 입력 자막 출처 감사 및 통일

- 감사 대상: `data/processed/gold_reference_relabelled_2026-07-23.csv`
- 후보 수: 60개
- 고유 롱폼 수: 52개
- 감사일: 2026-07-23

## 1. 기존 입력의 실제 출처

기존 `results/gold_reference_judge_balanced_30_30_gpt/input_verified/candidates_blind.csv`는
Vpick API 장면 데이터가 아니라 모두 `yt-dlp`로 수집한 YouTube 자막으로 만들어졌다.

| 기존 입력 방식 | 개수 |
|---|---:|
| 게시 쇼츠 자막 + 롱폼 전후 문맥 | 57 |
| 롱폼 gold 구간 자막 + 롱폼 전후 문맥 | 3 |
| Vpick scenes | 0 |

모든 후보의 `description`이 비어 있으므로 기존 60개 Judge 실행에서 Vpick 장면 설명은
사용되지 않았다.

쇼츠 자막이 없어 롱폼 자막으로 대체됐던 3개는 다음과 같다.

| pair_id | label | channel | short_video_id |
|---|---|---|---|
| P010 | pos | 안녕하세요원이입니다잘부탁드립니다 | Djc2pShwxpI |
| G009 | pos | 피식대학 | 8QmzdjiuBPo |
| G028 | pos | 워크맨 | vkjmsyrvRDI |

## 2. Vpick 데이터 가용성

Amazon 서버의 `data/raw/vpick/*_scenes.json`과 현재 60개를 대조한 결과는 다음과 같다.

| 구분 | Vpick scenes 있음 | Vpick scenes 없음 |
|---|---:|---:|
| pos | 30 | 0 |
| neg | 9 | 21 |
| 합계 | 39 | 21 |

고유 롱폼 기준으로는 52개 중 33개가 Vpick 분석 완료이고 19개가 미완료다.
따라서 현재 상태에서 `Vpick 우선 + yt-dlp 폴백`을 사용하면 입력 출처가 성과 라벨을
노출한다. 이 혼합 입력은 Judge 검증에 사용하지 않는다.

## 3. 즉시 적용한 통일 기준

60개 성과 라벨 검증용 canonical 입력은 다음으로 고정한다.

> `yt-dlp`로 수집한 롱폼 자막에서 gold 시작·종료 구간과 같은 롱폼의 전후 문맥을 추출한다.

게시 쇼츠 자막은 사용하지 않는다. 이 기준은 쇼츠 자막 유무에 따른 57:3 혼합을 없애고,
Judge가 최종 게시본의 재편집 자막이 아니라 원본에서 선택된 동일한 구간을 보도록 한다.

새 입력:

`results/gold_reference_judge_balanced_30_30_gpt/input_longform_ytdlp/candidates_blind.csv`

검증 결과:

| 항목 | 결과 |
|---|---:|
| 후보 | 60 |
| `evidence_provider=yt_dlp` | 60 |
| `evidence_source=long_subtitle_interval` | 60 |
| `transcript_scope=longform_gold_interval` | 60 |
| 후보 transcript 누락 | 0 |
| before context 누락 | 0 |
| after context 누락 | 0 |

## 4. 최종 Vpick용 검증본

현재 통일본은 자막 출처 혼합을 제거한 성과 라벨 검증용 입력이다. 실제 Ours와 Vpick
후보를 평가할 Judge는 Vpick 장면 분석을 입력받으므로, 다음 조건이 충족되면 별도의
Vpick-only 검증본을 만든다.

1. 미분석 롱폼 19개를 Vpick에 추가한다.
2. 60개 모든 후보의 `evidence_source`가 `vpick_scenes`인지 확인한다.
3. pos와 neg 모두 동일한 Vpick API 버전과 동일한 scene 추출 코드로 생성한다.
4. Vpick-only 결과와 현재 yt-dlp longform-only 결과를 별도로 보고한다.
5. 두 출처의 점수를 한 표본 안에서 섞어 하나의 AUC나 평균 차이로 계산하지 않는다.

## 5. 재현 명령

```bash
scripts/build_gold_reference_balanced_input.sh
```

후보 생성기는 `--require-evidence-source long_subtitle_interval`과
`--require-uniform-provider`를 사용한다. 새 후보 한 건이라도 다른 출처로 폴백하면
입력 생성을 실패시킨다.

Judge 결과도 기존 쇼츠 자막 기반 결과와 섞이지 않도록
`results/gold_reference_judge_balanced_30_30_gpt/longform_ytdlp/`에 별도로 저장한다.

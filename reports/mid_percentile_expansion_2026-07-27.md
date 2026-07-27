# 중간 백분위 골드 확장 — 매핑 파이프라인 실행 결과 (2026-07-27)

## 최종 산출

`results/mid_percentile_mapping_2026-07-27/goldlabel/`

| 파일 | 내용 |
|---|---|
| `goldlabel_master_mid_percentile_PRIVATE.csv` | 신규 30건, 기존 60건과 **27컬럼 완전 동일** |
| `goldlabel_master_90_PRIVATE.csv` | 기존 60 + 신규 30 = 90건 병합본 |
| `candidates_mid_percentile_blind.csv` | 판정용 블라인드 입력 6컬럼 |
| `blind_input_audit.json` | 검수 결과 (PASS) |
| `rejected.csv` | 제외 12건과 사유 |

검수: 스키마 일치, 금지 키 누출 0, description/transcript 공백 0, candidate_id 중복 0,
description 공백제외 52~81자 (기존 60건 59~83자와 동일 분포).

## 백분위 공백이 메워졌습니다

이번 확장의 핵심 성과입니다. 기존 60건은 중간 구간이 완전히 비어 있었습니다.

| 구간 | 기존 60 | 신규 30 | 합계 90 |
|---|---|---|---|
| p0_20 | 30 | 0 | 30 |
| p20_40 | **0** | 13 | 13 |
| p40_60 | **0** | 8 | 8 |
| p60_80 | **0** | 9 | 9 |
| p80_100 | 30 | 0 | 30 |

`performance_label_PRIVATE`에 `mid` 30건이 추가되어 pos 30 / mid 30 / neg 30이 됩니다.
연속형 `channel_performance_percentile`을 주지표로 쓰겠다는 계획의 선행 조건이 충족됐습니다.
표본도 n=90이 되어 rho=0.3을 80% power로 검출하는 데 필요한 n≈85를 넘습니다.

## 입력별 처리 결과

| 입력 | 건수 | 승인 | 제외 |
|---|---|---|---|
| `mid_percentile_origin_linked_18` (핀 댓글 자동 링크) | 18 | 9 | 9 |
| `mid_percentile_manual_mapping_22` (사용자 수동 매핑) | 22 | 20 | 2 |
| 사용자가 18건 시트에 직접 붙인 URL | 3 | 1 | 2(언어 제외) |
| **합계(중복 제외)** | | **30** | **12** |

**수동 매핑이 압도적으로 정확합니다** — 자동(핀 댓글) 50% vs 수동 91%.
자동 링크 실패의 원인은 아래 §링크 품질에 정리했습니다.

`timestamp_method`: `yt_dlp_subtitle_alignment` 26건 + `gemini_short_transcript_and_subtitle_alignment` 4건
`evidence_provider`: 전 건 `yt_dlp_transcript_fallback`
`mapping_confidence`: high 24 / medium 6

## 제외 12건

| 사유 | 건수 |
|---|---|
| `origin_mismatch_suspect` (링크된 롱폼과 자막 공통 부분문자열 0자) | 5 |
| `alignment_heavy_edit` (정렬은 되나 비연속 조립) | 3 |
| `gemini_empty_transcript` (발화 없는 음악/노래) | 2 |
| `needs_manual_review` (Gemini 후에도 span 과대) | 2 |

사용자 판단으로 제외한 언어 사유 2건은 위 집계와 별도입니다 —
피식대학 `02Y7z_3_tX0`(영어 발화), 안원잘부 `P5hyRkrFUSY`(일본어).

## 링크 품질: 핀 댓글 기반 origin 탐색의 한계

자동 링크 5건이 완전히 무관했고, 원인이 랜덤이 아니라 체계적입니다.
**핀/업로더 댓글의 링크가 원본 롱폼이 아니라 BGM·챌린지·참고 영상인 경우가 많습니다.**

| short | 잘못 링크된 origin | 실제 성격 |
|---|---|---|
| 안원잘부 `P5hyRkrFUSY` | 【こずえ】ルカルカ★ナイトフィーバー | 원곡/참고 영상 |
| 안원잘부 `m-MBLSS70x0` | 리센느 - Love Attack (유로비트) | BGM |
| 워크맨 `Ab2ZSmjbebg` | 언니 저 맘에 안들져 #눈네모챌린지 (44초) | 챌린지 숏폼 |
| BDNS `M4tdH9WM9jA` | 상훈이 엄마 장경자 매드무비 (60초) | 60초 클립 |

→ 댓글 링크는 반드시 자막 정렬 검증을 통과해야 씁니다. 검증 없이 쓰면 오염됩니다.

## 파이프라인 버그 2건 수정

**1. 자막 언어 불일치.** 기존 `audit_short_long_alignment.py`의 `_preferred_track`이 숏폼과
롱폼의 자막 언어를 독립적으로 골라 교차 언어 fuzzy 매칭이 일어났습니다.
`bxToIBbK4uk`는 38초 숏폼에 6.5초 구간이 `continuous`로 승인될 뻔했습니다(span ratio 0.17).
`src/align_shorts_langlocked.py`가 공통 언어를 먼저 정하고 span/숏폼 길이 비율로 게이트합니다.
기존 스크립트는 과거 실행 재현성을 위해 수정하지 않았습니다.

**2. 엑셀이 깨뜨린 video id.** `-`로 시작하는 id를 엑셀이 `#NAME?`로 바꿔
워크맨 한 건의 백분위 조회가 조용히 실패했습니다. 이제 short URL에서 id를 함께
파싱해 두 키로 인덱싱합니다.

## Vpick scene API

18건 배치: FAILED 12 / SUBMIT_FAILED 2 / READY 3 — 사유 미반환, 길이 무관.
`YwwjT_-xxR0`는 5시간 몰아보기(17,225초)로 `YOUTUBE_VIDEO_TOO_LONG`.

22건 배치는 실행 시점 기준 업로드 진행 중입니다. READY가 확보되면
`evidence_provider`를 `vpick_scene_api`로 승격하고 scene description을 붙일 수 있습니다.
현재 30건은 전부 `yt_dlp_transcript_fallback`입니다.

**주의: 기존 60건은 47건이 `vpick_scene_api`인데 신규 30건은 전부 fallback입니다.**
90건을 합쳐 평가할 때 provider 층화가 필수입니다 — 기존 60건에서
provider만으로 pos/neg AUC가 0.617이었습니다.

## Gemini

- `gemini-3.6-flash`: 2건 후 일일 쿼터 소진(`RESOURCE_EXHAUSTED`)
- `gemini-3.1-flash-lite`: 정상, 이후 전량 처리
- 영상 입력 1건당 약 6,600~7,300 토큰
- 복구 효과: span ratio 21.96→0.874, 43.83→0.979, 5.63→0.968, 언어불일치→0.961

## description

30건 전부 자막을 직접 읽고 작성했습니다. 규격은
`generate_short_candidate_descriptions.py`의 `SYSTEM_PROMPT`를 따랐습니다 —
관찰된 중심 상황·발화·반응만, 품질/성과 판단 배제, 공백제외 45~150자.

## 신규 코드

| 파일 | 역할 |
|---|---|
| `src/align_shorts_langlocked.py` | 언어 고정 자막 정렬 + span 타당성 게이트 |
| `src/fill_alignment_gaps_with_gemini.py` | Gemini 숏폼 전사 후 재정렬, 실패 시 origin 오류로 분류 |
| `src/build_mid_percentile_goldlabel.py` | 기존 27컬럼 골드라벨 스키마로 출력 |
| `src/audit_mid_percentile_blind_input.py` | 블라인드 입력 스키마·누출·공백 검수 |
| `src/build_longform_fill_sheet_18.py` | 수동 매핑용 기입 시트 생성 |
| `src/harvest_channel_catalogs.py` | 채널 롱폼 카탈로그 수집(자동 탐색용, 이번엔 미사용) |

## 남은 일

1. Vpick 22건 배치 결과 확인 → READY분은 `vpick_scene_api`로 승격
2. `alignment_heavy_edit` 3건 — 비연속 조립 구간을 골드로 인정할지 정책 결정
3. `origin_mismatch_suspect` 5건 + `needs_manual_review` 2건 — origin 재매핑 필요
4. 90건 병합 후 평가 시 **provider 층화 + permutation p-value** 필수

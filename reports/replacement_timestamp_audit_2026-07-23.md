# 교체 영상 10개 YouTube 메타데이터·타임스탬프 검증

## 결과 요약

- 입력: `goldlabel_60_all10_replaced_v3_draft.csv`
- 메타데이터 출처: yt-dlp가 수집한 YouTube 공개 메타데이터
- 타임스탬프 산출: 원본 자막 정렬을 우선하고, 자막이 없거나 언어가 깨진 경우 저화질 프레임 정렬로 보완
- 최종 판정: 10개 중 8개 사용 가능, 2개 heavy edit로 재교체 필요

| pair_id | 쇼츠 ID | 최종 롱폼 ID | 원본 구간 | 판정 | 사용 여부 |
|---|---|---|---|---|---|
| G002 | 447MOrMHk40 | d8W9nAukh4U | 03:00.480–03:45.200 | continuous | 사용 |
| G006 | 2hS55U-AMAQ | AeVP4dzq174 | 01:12.240–01:50.320 | continuous | 사용 |
| G007 | X5pBumtoQEo | d2QR-XPnPrw | 00:42.000–01:43.000 | continuous | 사용 |
| G009 | VuzTI2xk3-I | Nzvzpg0i78Y | 17:23.000–18:15.000 | light_edit | 사용 |
| G012 | vxfz-37y0qY | Fd32iyF-l8Q | 16:06.600–17:00.839 | light_edit | 사용 |
| G014 | MYmi2EX3Z_c | GQSSLlBKeSE | 03:08.360–04:06.959 | continuous | 사용 |
| G018 | ST3IiaHTbHI | -ZZfsK32sJQ | 14:29.120–22:44.960에 3개 조각 | heavy_edit | 재교체 |
| G024 | 1Sp31pKVVqc | 9aPAIgj6g8E | 07:56.483–08:51.484 | light_edit | 사용 |
| P001 | ORksT1QDf3k | 5JZ5biQ_hMI | 25:21–27:19에 다수 조각 | heavy_edit | 재교체 |
| P011 | gRESmZMqlFA | JJBf0jCcfrQ | 05:38.919–06:35.280 | light_edit | 사용 |

## 주요 수정

- G006은 draft에 롱폼이 없었으나 채널 롱폼 자막 탐색으로 `AeVP4dzq174(병문안)`을 찾았다.
- G007은 draft의 `AeVP4dzq174(병문안)` 매핑이 잘못됐다. 쇼츠 화면과 롱폼을 직접 대조해 `d2QR-XPnPrw(퇴실 30분 전)`으로 수정했다.
- G009는 쇼츠와 롱폼 자막 언어가 달라 텍스트 정렬이 실패했다. 프레임 정렬에서 17:23–18:15 구간이 순서대로 반복 일치했다.
- P001은 롱폼 자막이 없었지만 프레임 정렬로 원본 위치를 확인했다. 다만 35초 쇼츠가 약 118초 범위의 여러 장면을 조합해 연속 gold 구간으로 사용할 수 없다.
- G018도 3개 원본 조각 사이에 약 414초의 점프가 있어 교체본으로 채택하지 않았다.

## 산출물

- 전체 60개 반영본: `data/processed/goldlabel_60_all10_replaced_v3_timestamped_2026-07-23.csv`
- 교체 10개 상세본: `data/processed/goldlabel_replacements_v3_timestamped_2026-07-23.csv`
- YouTube 메타데이터 원본: `data/processed/goldlabel_replacements_v3_youtube_metadata_2026-07-23.csv`
- 최종 판정 원본: `outputs/replacement_alignment_2026-07-23/final_timestamp_decisions.csv`

`G018`과 `P001`의 공식 start/end는 의도적으로 비워 두고 `usable_for_gold=no`, `next_action=replace_pair`로 기록했다.

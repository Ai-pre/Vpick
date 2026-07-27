# 인수인계 — 숏폼 Judge 타당도 검증 및 중간 백분위 골드 확장 (2026-07-27)

## 한 줄 요약

**텍스트-only LLM Judge는 채널 내 조회수 백분위를 예측하지 못합니다.** 네 개의 독립적인
실험이 같은 결론에 도달했고, 측정 도구의 결함이나 표본 부족으로 설명되지 않습니다.
다음 단계는 프롬프트 개선이 아니라 멀티모달 증거 추가입니다.

---

## 1. 결론과 그 근거

### 1.1 핵심 수치

| 실험 | 평가자 | n | judge_score vs 백분위 rho | p | 유의 |
|---|---|---|---|---|---|
| mR3 prompt ablation v1~v5 | mR3-Qwen3-8B | 60 | −0.16 ~ +0.19 | — | 없음 |
| codex_rescore v1~v5 | 프론티어 모델 | 60 | +0.11 ~ +0.14 | — | 없음 |
| v10 judge | 프론티어 모델 | 60 | **+0.121** | 0.354 | 없음 |
| **v11 judge** | **Opus 5 직접 채점** | **94** | **+0.086** | 0.409 | 없음 |
| **v11 judge** | **Gemini 3.1 flash-lite** | **94** | **+0.089** | 0.402 | 없음 |

마지막 두 행이 가장 중요합니다. 서로 독립적인 두 평가자가 같은 프롬프트로 같은 94건을
채점해 **0.086 / 0.089라는 사실상 동일한 값**에 도달했습니다.

### 1.2 "측정 노이즈 때문"이 아닌 이유

평가자 간 일치도(Opus 5 vs Gemini, n=94):

| 축 | Spearman | 해석 |
|---|---|---|
| body_strength | **+0.655** | 재현 가능 (07-26 파일럿 게이트 0.60 통과) |
| opening_pull | **+0.585** | 재현 가능에 근접 |
| judge_score_100 | +0.459 | 중간 |
| boundary_integrity | **+0.056** | **재현 불가 — 축 설계 문제** |

즉 **내용 축은 재현 가능한 판정을 만드는데, 그 판정이 조회수와 연결되지 않습니다.**
도구 고장이 아니라 텍스트 증거의 한계입니다.

### 1.3 검정력

`judge_score` rho 0.086을 80% power로 검출하려면 **n≈1,000**이 필요합니다.
현재 94건이므로 표본을 두 배로 늘려도 유의해지지 않습니다. 표본 확대는 해법이 아닙니다.

---

## 2. 가장 실행 가능한 다음 실험

### 2.1 `opening_pull` 멀티모달 검증 (우선순위 1)

이 축이 가장 좁고 검정력이 유리한 가설입니다.

```
현재 상태:  평가자 간 일치 0.585  ·  백분위 상관 0.004~0.020
```

**텍스트로는 일관되게 판정되는데 조회수와 완전히 무관합니다.** 두 해석 중 하나입니다.

1. 도입 흡인력이 실제로 조회수와 무관하다 → 업계 상식과 배치
2. 텍스트로는 도입 흡인력의 실체를 못 본다 → 유력

첫 3초의 흡인력은 화면 자막, 컷 속도, 음향, 첫 프레임 구도에 있고 대사에는 없습니다.
**첫 3초의 프레임 + 오디오 + 화면 자막을 추가해 이 축만 재측정하십시오.**
그래도 rho가 0으로 남으면 1번 해석이 맞고, 그때 도입 흡인력 가설을 버릴 수 있습니다.

### 2.2 `boundary_integrity` 축 재설계 (우선순위 2)

평가자 간 0.056은 루브릭이 판정 불가능한 것을 요구한다는 뜻입니다.
before/after_context 대사만으로는 "경계를 1초도 옮길 이유가 없다"를 판정할 수 없습니다.
장면 전환 타임스탬프와 컷 정보를 증거에 넣지 않으면 이 축은 계속 노이즈입니다.

### 2.3 성과 예측과 구간 선택의 분리 (설계 원칙)

제목·썸네일을 넣으면 조회수 예측력은 오릅니다. 그러나 **둘 다 구간 선택으로 통제할 수
없는 변수**입니다. 성과 예측 Judge와 구간 선택 Judge를 하나로 만들려 하면 목표가 섞입니다.
처음부터 지표를 분리하십시오.

---

## 3. 골드라벨 데이터셋 (94건)

`results/mid_percentile_mapping_2026-07-27/final/`

| 파일 | 내용 |
|---|---|
| `vpick_goldlabel_final_PRIVATE.csv` | 94행 × 22컬럼. 롱폼/숏폼 주소, 구간, 채널, 라벨 |
| `vpick_short_subtitles_final_plain.csv` | 94행 × 6컬럼. **판정용** 대사 (형식 누출 제거) |
| `vpick_short_subtitles_final.csv` | 화자 라벨 유지판 (분석용, 판정에 쓰지 말 것) |
| `dataset_audit.json` | 11개 항목 검수 결과 (전부 통과) |

두 파일은 `candidate_id`로 1:1 조인됩니다.

```
라벨    pos 30 / mid 34 / neg 30
백분위  p0_20 30 · p20_40 12 · p40_60 12 · p60_80 10 · p80_100 30
출처    vpick_scene_api 47 / yt_dlp_transcript_fallback 47
```

### 3.1 이번에 새로 만든 34건

기존 60건은 극단만 있었습니다(p0_20 30, p80_100 30, 중간 **0건**).
중간 백분위 34건을 추가해 연속형 지표 사용의 선행 조건을 충족시켰습니다.

주소를 받은 44건 중 34건 편입(77%). 탈락 10건의 사유는
`goldlabel/rejected.csv`와 `data/longform_remap_sheet_2026-07-27.csv`에 있습니다.

### 3.2 판정용 파일을 반드시 `_plain` 쪽으로 쓸 것

`vpick_short_subtitles_final.csv`(화자 라벨 유지판)를 Judge에 주면 **라벨이 새어나갑니다.**
vpick 출처 행만 `S1:`/`S2:` 화자 구분이 있고 나머지는 `S?`이며, 그 차이만으로
mid 라벨을 **AUC 0.908**로 맞힐 수 있습니다. `_plain` 파일은 화자 토큰과 줄 구분을
제거해 이 누출을 없앴습니다(모든 형식 지표 AUC 0.500, 최악 0.583, 유의 0개).

---

## 4. 반드시 알아야 할 함정 7가지

이번에 실제로 발생했고 조용히 결과를 망칠 수 있는 것들입니다.

### 4.1 출력 예시값 앵커링

프롬프트 `[출력 형식]`의 예시 JSON에 박아둔 숫자를 모델이 그대로 복사합니다.

| 프롬프트 | 예시값 | 그 값을 그대로 출력한 비율 |
|---|---|---|
| ablation v1 | `3` | **93%** |
| ablation v4 | `63` | **88%** |
| ablation v2 | `63` | **72%** |
| v10 전체 | `confidence: 4` | **73%** |

**예시값은 placeholder(`<0-100 정수>`)로 쓰거나 저/중/고 3개를 함께 주십시오.**
v11은 예시를 63/71/55로 둬서 이 함정을 피했고, 실제 앵커값 사용률이 9~12%였습니다.

아이러니: `"10단위로 반올림하지 마십시오"`라는 지시가 63을 오히려 정당해 보이게 만들어
앵커를 더 고착시켰습니다.

### 4.2 유의성 검정 없이 축 순위를 보고하지 말 것

v10에서 `memorable_specificity` rho 0.162가 "가장 나은 축"으로 보고돼 있었는데
**p=0.216, 필요 n=296**입니다. n=60에서 그 크기는 노이즈와 구분되지 않습니다.

제가 v11 채점 중 46건 시점에 `boundary_integrity` rho **0.286**을 보고 유망하다고 판단했는데,
94건 완주 시 **0.094**로 떨어졌습니다. 절반 표본의 노이즈였습니다.

`src/evaluate_judge_validity.py`를 쓰면 permutation p-value, Holm 보정,
provider 층화, 검정력 부족 여부가 자동으로 붙습니다.

### 4.3 자막은 반드시 json3, VTT는 dedupe 필수

VTT 자동 자막은 같은 구절을 연속 큐에 반복합니다. dedupe 없이 파싱하면:

```
같은 롱폼:  json3 파싱 241큐  vs  VTT 파싱 11큐   (95% 유실)
어절 3-gram 중복률:  정상 0.007  vs  VTT 미dedupe 0.42
```

그러면 `alignment_score`가 무의미해집니다. 3큐 대 3큐를 맞추면 0.97이 나옵니다.
`--sub-format json3`을 쓰십시오. `src/audit_goldlabel_dataset.py`가 이 중복을 감지합니다.

### 4.4 숏폼/롱폼 자막 언어를 반드시 일치시킬 것

기존 `audit_short_long_alignment.py`는 각 영상의 자막 언어를 독립적으로 고릅니다.
숏폼 `ja-orig` × 롱폼 `ko-orig` 같은 조합이 생기고, 교차 언어 fuzzy 매칭이
**거짓 양성**을 만듭니다 — 38초 숏폼에 6.5초 구간이 `continuous`로 승인될 뻔했습니다.

`src/align_shorts_langlocked.py`가 공통 언어를 먼저 정하고 span/숏폼 길이 비율로 게이트합니다.

### 4.5 창 탐색은 coverage만 쓰면 안 됨

구간을 좁힐 때 coverage(숏폼 발화가 창에 얼마나 포함되나)만 최대화하면
**창이 넓을수록 무조건 점수가 오릅니다.** 항상 상한에 붙습니다.
precision과의 F1을 쓰면 내부 최적점이 생깁니다.

```
coverage only:  ratio 1.586, 1.559  (상한 1.6에 붙음)
F1:             ratio 0.936        (상한 1.0/1.3/1.6에서 동일 = 진짜 최적점)
```

### 4.6 자막 유사도 비교 시 화자 토큰을 제거할 것

Gemini 재전사와 기존 자막을 대조할 때 `S1:`/`S2:` 토큰을 텍스트로 세면 점수가 깎입니다.
8건이 "신뢰불가"로 떴는데 토큰 제거 후 **7건이 거짓 양성**이었습니다.
그대로 진행하면 더 좋은 자막을 더 나쁜 것으로 교체했을 것입니다.

### 4.7 엑셀이 video id를 파괴함

`-`로 시작하는 YouTube id를 엑셀이 `#NAME?`로 바꿉니다(`-KwnmGBZz-g`).
`short_url`에서 id를 함께 파싱해 두 키로 인덱싱하십시오. 안 하면 조회가 조용히 실패합니다.

---

## 5. 자동 origin 링크(핀 댓글)는 신뢰도 50%

핀/업로더 댓글의 링크로 롱폼을 찾는 방식은 **정확도 50%**였습니다(수동 매핑은 91%).
실패가 랜덤이 아니라 체계적입니다 — **댓글 링크가 원본 롱폼이 아니라 BGM·챌린지·참고 영상**인 경우.

| short | 잘못 링크된 origin | 실제 성격 |
|---|---|---|
| 안원잘부 `P5hyRkrFUSY` | 【こずえ】ルカルカ★ナイトフィーバー | 원곡 참고 영상 |
| 안원잘부 `m-MBLSS70x0` | 리센느 - Love Attack | BGM |
| 워크맨 `Ab2ZSmjbebg` | 챌린지 숏폼 (44초) | 롱폼 아님 |
| BDNS `M4tdH9WM9jA` | 매드무비 (60초) | 롱폼 아님 |

**댓글 링크는 반드시 자막 정렬 검증을 통과시키십시오.** 검증 없이 쓰면 오염됩니다.

---

## 6. 재현 방법

### 6.1 환경

```bash
python3 -m pip install rapidfuzz     # 정렬에 필요
# yt-dlp 2026.07.04 이상
# 자막 요청은 6초 이상 간격 (그보다 빠르면 HTTP 429)
```

자격증명은 환경변수로만 전달하십시오. 공유 서버에서는 `ps`에 노출되지 않도록
0600 권한 env 파일을 리포 밖에 두고 `set -a; . envfile; set +a` 로 로드하십시오.

```
GEMINI_API_KEY          Gemini 자막 전사·판정
VPICK_EMAIL/PASSWORD    Vpick scene API
```

### 6.2 파이프라인 순서

```bash
# 1) 자막 수집 (언어 고정) + 구간 정렬
python3 src/align_shorts_langlocked.py \
  --input <short/long 쌍 CSV> --out-dir <out> --sleep-seconds 7

# 2) 정렬 실패분 Gemini 전사 후 재정렬
python3 src/fill_alignment_gaps_with_gemini.py \
  --alignment <out>/alignment_langlocked.csv \
  --subtitle-dir <out>/subtitles --out-dir <gem> --model gemini-3.1-flash-lite

# 3) 구간이 과대한 건은 창 탐색으로 좁힘 (F1 기준)
python3 src/locate_span_windowed.py \
  --input <origin 확정 CSV> --subtitle-dir <out>/subtitles \
  --gemini-transcripts <gem>/gemini_short_transcripts.jsonl \
  --out-dir <win> --min-ratio 0.6 --max-ratio 1.6

# 4) 기존 27컬럼 골드라벨 스키마로 조립
python3 src/build_mid_percentile_goldlabel.py --span-budget-sec 60 ...

# 5) 라벨/자막 2파일로 분리 (판정용 plain 포함)
python3 src/build_final_goldlabel_split.py --master ... --descriptions ...

# 6) 검수 — 통과할 때까지 다음 단계로 넘어가지 말 것
python3 src/audit_goldlabel_dataset.py --labels ... --subtitles ...
python3 src/audit_mid_percentile_blind_input.py --evidence ... --manifest ...

# 7) 판정 실행
python3 src/run_shortform_judge_v11_gemini.py \
  --evidence <plain 자막> --prompt prompts/shortform_judge_v11_ko.md --out-dir ...

# 8) 타당도 검증 (p-value·Holm·층화 자동)
python3 src/evaluate_judge_validity.py --labels ... --scores ... --iterations 20000
```

### 6.3 신규 코드 목록

| 파일 | 역할 |
|---|---|
| `src/align_shorts_langlocked.py` | 언어 고정 자막 정렬 + span 타당성 게이트 |
| `src/fill_alignment_gaps_with_gemini.py` | Gemini 숏폼 전사 후 재정렬, 실패 시 origin 오류 분류 |
| `src/locate_span_windowed.py` | F1 기반 창 탐색으로 과대 구간 축소 |
| `src/transcribe_spans_with_gemini.py` | 구간 지정 Gemini 재전사 + 기존 자막 신뢰도 대조 |
| `src/build_mid_percentile_goldlabel.py` | 27컬럼 골드라벨 스키마 출력 |
| `src/build_final_goldlabel_split.py` | 라벨/자막 2파일 분리 + 누출 제거판 생성 |
| `src/audit_goldlabel_dataset.py` | 데이터셋 11개 항목 검수 |
| `src/audit_mid_percentile_blind_input.py` | 블라인드 입력 스키마·누출 검수 |
| `src/significance.py` | permutation p-value, Holm 보정, 검정력 계산 |
| `src/evaluate_judge_validity.py` | 타당도 검증 (rho/AUC + p + 층화) |
| `src/run_shortform_judge_v11_gemini.py` | v11 판정 실행 (Gemini) |
| `src/run_shortform_judge_v10_gemini.py` | v10 판정 실행 (Gemini) |
| `src/harvest_channel_catalogs.py` | 채널 롱폼 카탈로그 수집 |
| `src/fetch_goldlabel_longform_captions.py` | 골드 롱폼 자막 일괄 캐시 |
| `src/build_longform_remap_sheet.py` | 재매핑 기입 시트 생성 |
| `src/build_longform_fill_sheet_18.py` | 롱폼 URL 기입 시트 생성 |
| `src/build_unlinked_fill_sheet.py` | 미링크 후보 시트 + 후보 롱폼 자동 추천 |

---

## 7. 미해결 과제

| 항목 | 상태 |
|---|---|
| 롱폼 재매핑 7건 | `data/longform_remap_sheet_2026-07-27.csv` — origin 링크 오류 등 |
| 주소 미확보 22건 | `data/longform_fill_unlinked_2026-07-27.csv` — 후보 롱폼 3개씩 자동 추천 부착 |
| Vpick 승격 | 22건 배치에서 READY 6건 확보했으나 골드에 미반영. Vpick 성공률 약 25% |
| `boundary_integrity` 재설계 | 평가자 간 0.056 — 축 자체를 다시 정의해야 함 |
| Opus 5 자기 신뢰도 | **측정 실패** — 같은 세션 재채점은 회상이라 무의미(§8) |

주소 미확보 22건은 **중간대 검정에는 부족합니다.** 전량 편입해도 중간대 53건이고
rho 0.35 검출에 62건이 필요합니다. 우선순위를 낮게 두십시오.

---

## 8. 실패로 기록하는 것

**자기 반복 신뢰도 측정을 시도했다가 실패했습니다.** 같은 세션에서 순서를 셔플해
94건을 재채점하려 했으나, 10건 시점에 중단했습니다.

```
전체 축 평균 절대차 1.47점 | 최대 2점 | 2점 이내 비율 100%
종합 Spearman(1회차, 2회차) = 0.988
```

1회차 점수가 컨텍스트에 남아 있어 재판정이 아니라 **회상**이 됐습니다.
순서 셔플과 description 가림으로는 막을 수 없었습니다. 설계 오류입니다.

**자기 신뢰도를 재려면 1회차 결과가 없는 새 세션에서 채점해야 합니다.**
`prompts/shortform_judge_v11_ko.md`와 `vpick_short_subtitles_final_plain.csv`만 있으면 재현됩니다.

대신 **Gemini를 독립 평가자로 써서 평가자 간 일치도를 측정**했고, 결과적으로 그게
훨씬 강한 증거였습니다(§1.2). 오염된 자기 반복 대신 다른 모델을 쓰는 것을 권합니다.

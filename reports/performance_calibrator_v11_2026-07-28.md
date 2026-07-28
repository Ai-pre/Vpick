# Vpick 성과 예측 Judge v11 검증 보고서

## 1. 목적

Vpick 과제 PDF 10쪽의 `정답 일치도` 정의에 맞춰, 블라인드 Judge 점수가 실제
채널 내 Shorts 성과 백분위 순서를 복원하는지 검증했다. 기존 v10에서 변별력이
있는 7개 루브릭 축을 고정된 품질 특징 추출기로 사용하고, 성과 백분위는 모델
입력에서 제외했다.

## 2. 데이터 감사

- 후보: 94개
- 원본 롱폼: 85개
- canonical Vpick 장면 파일이 있는 롱폼: 43개
- yt-dlp 대체 장면 파일이 있는 롱폼: 14개
- 장면 파일이 전혀 없는 롱폼: 31개
- Vpick READY inventory와 연결된 롱폼: 33개
- Vpick FAILED inventory와 연결된 롱폼: 11개

수집 방식이 예측 편법으로 작동하는지 확인하기 위해 `transcript_source`는
배포 후보 입력에서 제외하고, 출처 존재 여부만 쓰는 점수를 사후 대조군으로
두었다.

## 3. 검증 설계

- 목표값: 채널 내 연속 성과 백분위
- 외부 검증: 5-fold GroupKFold를 3개 seed로 반복
- 그룹 키: `longform_id` (동일 원본의 후보가 학습·검증에 동시에 들어가지 않음)
- 하이퍼파라미터: 각 외부 학습 폴드 안의 4-fold grouped CV에서만 선택
- 주 지표: 채널 중심화 Spearman
- 편법 방지 지표: 채널별 macro, 자막 출처 제거 상관,
  같은 채널 쌍 정확도, 연속 백분위 차이 10~40인 근접 쌍 정확도
- 불확실성: 롱폼 단위 bootstrap 95% CI

기존 locked test는 이미 여러 차례 열람했으므로, 이번 수치는 `exploratory nested
OOF`다. 최종 상용 주장에는 새로 수집한 미공개 holdout이 필요하다.

## 4. 모델 비교

| 모델 | 배포 후보 | Pooled rho | 채널 중심 rho | 채널 Macro rho | 출처 제거 rho | 쌍 정확도 | 강건 점수 |
|---|---|---|---|---|---|---|---|
| pairwise_char_tfidf_numeric | True | 0.2424 | 0.2577 | 0.2544 | 0.2099 | 0.5918 | 0.2249 |
| char_tfidf_numeric | True | 0.2049 | 0.2092 | 0.2088 | 0.1215 | 0.5706 | 0.1765 |
| rank_ensemble_text_structure | True | 0.2033 | 0.2085 | 0.2027 | 0.1760 | 0.5706 | 0.1748 |
| pairwise_quality_structure | True | 0.1354 | 0.1429 | 0.1295 | 0.0659 | 0.5524 | 0.1429 |
| stacked_text_structure | True | 0.1217 | 0.1060 | 0.2337 | 0.0836 | 0.5417 | 0.1188 |
| ridge_quality_structure | True | 0.1011 | 0.1068 | 0.1261 | 0.0112 | 0.5341 | 0.1144 |
| nested_selected_pipeline | True | 0.1125 | 0.1048 | 0.1345 | 0.0579 | 0.5463 | 0.1126 |
| char_tfidf | True | 0.1821 | 0.1604 | 0.1657 | 0.1658 | 0.5569 | 0.0922 |
| pairwise_char_tfidf | True | 0.1391 | 0.1577 | 0.1429 | 0.1843 | 0.5448 | 0.0916 |
| extra_trees_quality_structure | True | 0.0214 | 0.0060 | 0.0161 | -0.0397 | 0.5326 | 0.0799 |
| fixed_v10 | True | 0.0131 | 0.0080 | -0.0269 | 0.0419 | 0.5038 | -0.0360 |
| fixed_equal_quality | True | 0.0131 | 0.0080 | -0.0269 | 0.0419 | 0.5038 | -0.0360 |
| ridge_quality | True | -0.2397 | -0.2273 | -0.2310 | -0.2408 | 0.4188 | -0.1848 |
| source_presence_fixed | False | 0.1572 | 0.1780 | 0.1771 | 0.0573 | 0.5683 | 0.1383 |
| source_only | False | -0.0903 | -0.0069 | -0.0839 | -0.3195 | 0.4818 | -0.0484 |
| random | False | -0.0859 | -0.1180 | -0.0756 | -0.1413 | 0.4659 | -0.0643 |
| constant | False | -0.3242 | -0.3174 | -0.2771 | -0.3667 | 0.3665 | -0.2474 |

## 5. 최종 게이트

- 주 검증 파이프라인: `nested_selected_pipeline`
- 판정: **기각**
- 채널 중심화 Spearman: 0.1048
- 채널 중심화 Spearman 95% bootstrap CI:
  [-0.0961, 0.3212]
- 가장 강한 출처 대조군: `source_presence_fixed`
- 출처 대조군 채널 중심화 Spearman: 0.1780
- 최상위 후보 강건 점수: 0.1126

| 게이트 | 관측값 | 최소 기준 | 통과 |
|---|---|---|---|
| channel_centered_spearman | 0.1048 | 0.3000 | False |
| channel_macro_spearman | 0.1345 | 0.2000 | False |
| source_residual_spearman | 0.0579 | 0.1000 | False |
| same_channel_pairwise_accuracy | 0.5463 | 0.5800 | False |
| same_channel_local_pairwise_accuracy | 0.5592 | 0.5500 | True |
| channel_centered_gain_over_source_control | -0.0732 | 0.0500 | False |
| bootstrap_primary_ci_lower | -0.0961 | 0.0000 | False |

## 6. 결론

현재 입력만으로는 PDF 10쪽에서 요구하는 높은 정답 일치도를 확보하지 못했다. 개발 단계의 개별 최고 모델보다 완전 중첩 모델 선택 파이프라인의 성능이 크게 낮아 안정적인 일반화 신호를 확인하지 못했다. v10은 편집·내용 품질 진단기로 유지하고, 성과 예측 Judge라는 명칭은 사용하지 않는다. 다음 데이터 수집에서는 모든 후보에 동일한 Vpick 장면·자막 근거를 확보하고, 연속 성과 분포의 신규 미공개 holdout을 추가해야 한다.

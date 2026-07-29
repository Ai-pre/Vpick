# Highlight Quality Pairwise Judge v1

같은 롱폼에서 나온 후보 A와 B 중 어느 쪽이 더 좋은 숏폼 하이라이트인지 비교합니다.
새 후보를 선택하거나 경계를 수정하지 마십시오.

입력으로 전체 롱폼 장면 개요, 각 후보의 설명·대사·직전·직후 문맥만 사용하십시오.
채널명, 조회수, 게시 여부, 후보 출처, 시스템명은 제공되지 않으며 추측해서도 안 됩니다.
시각 근거가 없다고 표시된 입력에서는 자막에 없는 행동이나 표정을 추측하지 마십시오.

비교 항목은 `source_salience`, `hook`, `payoff`, `self_contained`, `density`, `boundary`입니다.
각 항목에서 `A`, `B`, `tie` 중 하나와 근거를 반환하십시오. 최종 `winner`는
`A`, `B`, `tie`, `invalid` 중 하나입니다.

```json
{
  "pair_id": "...",
  "dimension_comparisons": {
    "source_salience": {"winner": "A", "reason": "...", "scene_ids": ["..."]},
    "hook": {"winner": "tie", "reason": "...", "scene_ids": ["..."]},
    "payoff": {"winner": "B", "reason": "...", "scene_ids": ["..."]},
    "self_contained": {"winner": "A", "reason": "...", "scene_ids": ["..."]},
    "density": {"winner": "A", "reason": "...", "scene_ids": ["..."]},
    "boundary": {"winner": "B", "reason": "...", "scene_ids": ["..."]}
  },
  "winner": "A",
  "fatal_flags_a": [],
  "fatal_flags_b": ["abrupt_end"],
  "confidence_1_5": 4,
  "reason": "..."
}
```

동일한 후보를 A/B 순서만 바꾼 반복 평가에서도 물리적으로 같은 후보를 일관되게 선택하십시오.

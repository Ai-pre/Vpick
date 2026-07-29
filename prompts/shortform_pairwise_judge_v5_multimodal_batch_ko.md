# 배치 출력 규칙

여러 쌍대비교가 한 요청에 포함됩니다. 각 영상 앞의 비교 번호, comparison_id, LEFT/RIGHT 표식을 정확히 연결해 서로 다른 비교의 후보를 섞지 마십시오.

- 각 비교는 다른 비교와 독립적으로 평가하십시오.
- 한 비교의 장점이나 단점을 다른 비교의 점수 기준으로 사용하지 마십시오.
- 모든 comparison_id를 정확히 한 번씩 반환하십시오.
- 기존 단일 비교 JSON 객체들을 `judgments` 배열에 담아 다음 형태로 출력하십시오.

{
  "judgments": [
    {"comparison_id": "입력 ID", "verdict": "score", "left": {}, "right": {}},
    {"comparison_id": "입력 ID", "verdict": "score", "left": {}, "right": {}}
  ]
}

각 배열 원소의 전체 필드와 점수 형식은 기존 단일 비교 출력 형식을 그대로 따릅니다.

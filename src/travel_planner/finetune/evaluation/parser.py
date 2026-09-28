"""모델 출력 문자열 → 예측 라벨.

1. JSON 파싱: 코드펜스나 앞뒤 설명이 있어도 첫 '{'부터 JSON 객체 하나를 읽는다.
2. 형식 검사: labels.label_issues로 형식(schema) · 허용값(value) · 근거(evidence) 문제를 모은다.
3. 채점용 정리: 형식이 깨진 aspect는 버리고, 허용값을 벗어난 aspect는 남긴다 (틀린 예측으로 채점된다).
   JSON 파싱이 실패하거나 최상위 모양이 틀리면 빈 예측(aspect 0개)으로 채점한다.
"""

import json
from dataclasses import dataclass, field

from ..labels import ASPECT_FIELDS, Issue, label_issues


@dataclass
class ParsedPrediction:
    raw: str
    parsed: bool  # JSON 객체를 읽었는가
    data: dict | None
    parse_error: str | None = None
    issues: list[Issue] = field(default_factory=list)

    @property
    def schema_valid(self) -> bool:
        """파싱 성공 + 필수 필드 · 타입이 맞음."""
        return self.parsed and not any(issue.kind == "schema" for issue in self.issues)

    @property
    def strict_valid(self) -> bool:
        """schema_valid + 허용되지 않은 값이 없음. 비교표의 'JSON Valid'."""
        return self.schema_valid and not any(issue.kind == "value" for issue in self.issues)

    def label_for_scoring(self) -> dict:
        empty = {"traveler_context": [], "aspects": []}
        if not self.parsed or not isinstance(self.data, dict):
            return empty
        contexts = self.data.get("traveler_context")
        aspects = self.data.get("aspects")
        return {
            "traveler_context": [c for c in contexts if isinstance(c, str)] if isinstance(contexts, list) else [],
            "aspects": [
                {f: a[f] for f in ASPECT_FIELDS}
                for a in (aspects if isinstance(aspects, list) else [])
                if isinstance(a, dict) and all(isinstance(a.get(f), str) for f in ASPECT_FIELDS)
            ],
        }


def extract_json_object(text: str) -> tuple[dict | None, str | None]:
    decoder = json.JSONDecoder()
    error = "출력에 JSON 객체가 없음"
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError as e:
            error = f"JSON 파싱 실패: {e.msg}"
        else:
            if isinstance(value, dict):
                return value, None
            error = "JSON 객체가 아님"
        start = text.find("{", start + 1)
    return None, error


def parse_prediction(raw: str, category: str, review: str) -> ParsedPrediction:
    data, error = extract_json_object(raw)
    if data is None:
        return ParsedPrediction(raw=raw, parsed=False, data=None, parse_error=error)
    return ParsedPrediction(raw=raw, parsed=True, data=data, issues=label_issues(data, category, review))

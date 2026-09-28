"""정답 레코드 + 모델 출력 문자열 → 지표. 모델이 무엇이든(Base · LoRA · QLoRA · 가짜 출력) 같은 함수로 채점한다."""

from .metrics import aggregate, safe_div, score_review
from .parser import parse_prediction


def format_metrics(parsed: list) -> dict:
    n = len(parsed)
    pred_aspects = sum(len(p.data.get("aspects", [])) for p in parsed if p.parsed and isinstance(p.data.get("aspects"), list))
    value_issues = sum(1 for p in parsed for issue in p.issues if issue.kind == "value")
    return {
        "json_parse_rate": safe_div(sum(p.parsed for p in parsed), n),
        "schema_valid_rate": safe_div(sum(p.schema_valid for p in parsed), n),
        "json_valid_rate": safe_div(sum(p.strict_valid for p in parsed), n),
        "invalid_value_count": value_issues,
        "invalid_value_rate": safe_div(value_issues, pred_aspects),  # 예측 aspect 하나당 허용값 위반 수
    }


def evaluate(records: list[dict], outputs: list[str]) -> dict:
    """records[i]의 정답과 outputs[i]를 비교한다. {"overall", "by_category", "samples"}를 돌려준다."""
    if len(records) != len(outputs):
        raise ValueError(f"레코드 {len(records)}건과 출력 {len(outputs)}건의 수가 다릅니다")

    parsed = [parse_prediction(out, r["category"], r["review"]) for r, out in zip(records, outputs)]
    scores = [
        score_review(p.label_for_scoring(), r["label"], r["review"], r["category"]) for r, p in zip(records, parsed)
    ]

    def summary(indices: list[int]) -> dict:
        return {**format_metrics([parsed[i] for i in indices]), **aggregate([scores[i] for i in indices])}

    categories = sorted({r["category"] for r in records})
    samples = [
        {
            "review_id": r["review_id"],
            "category": r["category"],
            "raw_output": p.raw,
            "parsed": p.parsed,
            "parse_error": p.parse_error,
            "issues": [f"{issue.kind}: {issue.message}" for issue in p.issues],
            "prediction": p.label_for_scoring(),
            "gold": r["label"],
            "exact_match": s.exact_match,
        }
        for r, p, s in zip(records, parsed, scores)
    ]
    return {
        "overall": summary(list(range(len(records)))),
        "by_category": {c: summary([i for i, r in enumerate(records) if r["category"] == c]) for c in categories},
        "samples": samples,
    }

"""정답 레코드 + 모델 출력 문자열 → 지표. 모델이 무엇이든(Base · LoRA · QLoRA · 가짜 출력) 같은 함수로 채점한다."""

from dataclasses import dataclass

from .metrics import ReviewScore, aggregate, safe_div, score_review, task_checks
from .parser import ParsedPrediction, parse_prediction


@dataclass
class ScoredReview:
    record: dict
    parsed: ParsedPrediction
    score: ReviewScore
    checks: dict[str, bool]  # 과업 성공 조건별 통과 여부 (metrics.task_checks)

    @property
    def success(self) -> bool:
        return all(self.checks.values())


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


def score_outputs(records: list[dict], outputs: list[str]) -> list[ScoredReview]:
    """records[i]의 정답과 outputs[i]를 리뷰 단위로 채점한다."""
    if len(records) != len(outputs):
        raise ValueError(f"레코드 {len(records)}건과 출력 {len(outputs)}건의 수가 다릅니다")
    scored = []
    for r, out in zip(records, outputs):
        parsed = parse_prediction(out, r["category"], r["review"])
        score = score_review(parsed.label_for_scoring(), r["label"], r["review"], r["category"])
        scored.append(ScoredReview(r, parsed, score, task_checks(score, parsed.strict_valid)))
    return scored


def summarize(scored: list[ScoredReview]) -> dict:
    return {
        **format_metrics([s.parsed for s in scored]),
        **aggregate([s.score for s in scored]),
        "task_success_rate": safe_div(sum(s.success for s in scored), len(scored)),
    }


def evaluate(records: list[dict], outputs: list[str]) -> dict:
    """records[i]의 정답과 outputs[i]를 비교한다. {"overall", "by_category", "samples"}를 돌려준다."""
    scored = score_outputs(records, outputs)
    categories = sorted({r["category"] for r in records})
    samples = [
        {
            "review_id": s.record["review_id"],
            "category": s.record["category"],
            "raw_output": s.parsed.raw,
            "parsed": s.parsed.parsed,
            "parse_error": s.parsed.parse_error,
            "issues": [f"{issue.kind}: {issue.message}" for issue in s.parsed.issues],
            "prediction": s.parsed.label_for_scoring(),
            "gold": s.record["label"],
            "exact_match": s.score.exact_match,
            "task_success": s.success,
            "task_checks": s.checks,
        }
        for s in scored
    ]
    return {
        "overall": summarize(scored),
        "by_category": {c: summarize([s for s in scored if s.record["category"] == c]) for c in categories},
        "samples": samples,
    }

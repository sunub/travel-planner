"""같은 split에서 두 run(A · B)의 예측을 리뷰 단위로 짝지어 비교한다.

모델을 불러오지 않는다. 저장된 예측(predictions/<split>.jsonl의 raw_output)을 지금 채점 코드로 다시 채점한다.
  - transitions: 리뷰마다 과업 성공 여부가 A → B로 어떻게 바뀌었는가 (개선 · 퇴행 · 둘 다 성공 · 둘 다 실패)
  - paired_bootstrap: 같은 리뷰 묶음을 복원추출해 지표 차이(B − A)를 반복 계산한 95% 신뢰구간
"""

import random
from pathlib import Path

from ..data.dataset import read_jsonl
from .evaluator import ScoredReview, score_outputs, summarize

BOOTSTRAP_METRICS = (
    "task_success_rate", "exact_review_match", "aspect_f1", "aspect_precision", "aspect_recall",
    "attribute_accuracy", "sentiment_accuracy", "evidence_exact_match", "evidence_overlap_f1", "json_valid_rate",
)  # fmt: skip
TRANSITIONS = ("improved", "regressed", "both_success", "both_fail")


def load_outputs(path: Path) -> dict[str, str]:
    """predictions/<split>.jsonl → {review_id: raw_output}."""
    if not path.exists():
        raise FileNotFoundError(f"예측 파일이 없습니다: {path}")
    return {row["review_id"]: row["raw_output"] for row in read_jsonl(path)}


def align_outputs(records: list[dict], outputs: dict[str, str], name: str) -> list[str]:
    """records 순서대로 출력을 늘어놓는다. 예측과 레코드의 review_id가 정확히 같아야 한다."""
    expected, got = {r["review_id"] for r in records}, set(outputs)
    if expected != got:
        raise ValueError(
            f"{name}: 예측과 데이터의 review_id가 다릅니다 (예측에만 {len(got - expected)}건, 데이터에만 {len(expected - got)}건). "
            "같은 split · 같은 데이터로 만든 예측인지 확인하세요."
        )
    return [outputs[r["review_id"]] for r in records]


def transition(a: ScoredReview, b: ScoredReview) -> str:
    if a.success == b.success:
        return "both_success" if a.success else "both_fail"
    return "improved" if b.success else "regressed"


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def paired_bootstrap(
    a: list[ScoredReview], b: list[ScoredReview], *, n_resamples: int = 2000, seed: int = 0, metrics=BOOTSTRAP_METRICS
) -> dict[str, dict]:
    """리뷰 index를 복원추출해 두 모델에 같은 묶음을 쓰고, 지표 차이(B − A)의 95% 신뢰구간을 낸다."""
    rng = random.Random(seed)
    n = len(a)
    diffs: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        sa, sb = summarize([a[i] for i in idx]), summarize([b[i] for i in idx])
        for m in metrics:
            if sa[m] is not None and sb[m] is not None:
                diffs[m].append(sb[m] - sa[m])
    full_a, full_b = summarize(a), summarize(b)
    result = {}
    for m in metrics:
        observed = None if full_a[m] is None or full_b[m] is None else full_b[m] - full_a[m]
        if not diffs[m]:
            result[m] = {"diff": observed, "ci_low": None, "ci_high": None, "excludes_zero": None}
            continue
        low, high = _percentile(diffs[m], 0.025), _percentile(diffs[m], 0.975)
        result[m] = {"diff": observed, "ci_low": low, "ci_high": high, "excludes_zero": low > 0 or high < 0}
    return result


def compare(records: list[dict], outputs_a: list[str], outputs_b: list[str], *, n_resamples: int = 2000, seed: int = 0) -> dict:
    """두 run 비교 결과와 리뷰별 사례를 돌려준다. {"report": 리뷰 원문 없는 요약, "cases": 리뷰별 원문 · 출력 · 판정}."""
    a, b = score_outputs(records, outputs_a), score_outputs(records, outputs_b)
    kinds = [transition(x, y) for x, y in zip(a, b)]
    categories = sorted({r["category"] for r in records})
    report = {
        "num_samples": len(records),
        "overall": {"a": summarize(a), "b": summarize(b)},
        "by_category": {
            c: {side: summarize([s for s in scored if s.record["category"] == c]) for side, scored in (("a", a), ("b", b))}
            for c in categories
        },
        "transitions": {
            kind: {"count": kinds.count(kind), "review_ids": [r["review_id"] for r, k in zip(records, kinds) if k == kind]}
            for kind in TRANSITIONS
        },
        "bootstrap": paired_bootstrap(a, b, n_resamples=n_resamples, seed=seed),
    }
    cases = [
        {
            "review_id": r["review_id"],
            "category": r["category"],
            "transition": kind,
            "review": r["review"],
            "gold": r["label"],
            **{
                side: {"success": s.success, "checks": s.checks, "prediction": s.parsed.label_for_scoring(), "raw_output": s.parsed.raw}
                for side, s in (("a", x), ("b", y))
            },
        }
        for r, x, y, kind in zip(records, a, b, kinds)
    ]
    return {"report": report, "cases": cases}

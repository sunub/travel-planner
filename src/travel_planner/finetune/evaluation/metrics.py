"""채점 지표. 순수 함수만 있고 모델 · 추론 코드와 무관하다. 정의는 docs/evaluation-plan.md 2절을 따른다.

짝짓기 (리뷰 하나 안에서):
  - aspect category가 같은 예측과 정답끼리만 짝을 짓는다.
  - 같은 category가 여러 개면 evidence 겹침 F1이 큰 쌍부터 짝짓는다 (동점이면 앞 순서).
  - 짝이 없는 예측은 FP(없는 걸 만들어냄), 짝이 없는 정답은 FN(놓침).

지표:
  - aspect_{precision,recall,f1}: 짝지어진 쌍 수(TP)를 전체 예측 · 정답 aspect 수로 나눈 micro 값.
  - aspect_macro_f1: aspect 종류("카테고리/aspect")마다 F1을 낸 뒤 평균. 정답이나 예측에 한 번이라도 나온 종류만.
  - aspect_set_f1: 리뷰마다 중복을 없앤 aspect category 집합을 비교한 멀티라벨 micro F1 (짝짓기 없이).
  - attribute_accuracy / sentiment_accuracy: 짝지어진 쌍 중 값이 같은 비율. 놓친 aspect와 값 오류를 분리한다.
  - sentiment_macro_f1: 짝지어진 쌍에서 positive · negative · neutral F1의 평균 (정답이나 예측에 나온 값만).
  - evidence_in_source_rate: 모든 예측 aspect 중 evidence가 리뷰 원문의 부분 문자열인 비율.
  - evidence_exact_match: 짝지어진 쌍 중 evidence가 정답과 글자까지 같은 비율.
  - evidence_overlap_f1: 짝지어진 쌍의 글자 위치 겹침 F1 평균. 두 구절을 원문에서 찾아 [시작, 끝) 구간을 비교하고,
    구절이 원문에 여러 번 나오면 가장 많이 겹치는 위치를 쓴다. 예측 구절이 원문에 없으면 0.
  - traveler_context_f1: 리뷰마다 traveler_context 집합을 비교한 micro F1.
  - exact_review_match: aspect 목록(순서 무시)과 traveler_context가 모두 정답과 같은 리뷰 비율.
  - empty_review_accuracy: 정답 aspect가 없는 리뷰에서 예측도 비어 있는 비율 ("없을 때 조용한가").
  - task_success_rate: 아래 네 조건(task_checks)을 모두 만족한 리뷰 비율. 평가 강의자료의 "성공 = 기준 모두 충족"에 해당한다.
      format   JSON이 스키마와 허용값을 지킨다 (parser의 strict_valid)
      aspects  예측 aspect와 정답 aspect가 모두 짝지어진다 (FP · FN 없음)
      values   짝지어진 모든 aspect의 attribute · sentiment가 정답과 같다
      evidence 모든 예측 evidence가 원문 구절이고, 짝지어진 쌍의 겹침 F1이 TASK_SUCCESS_MIN_OVERLAP 이상이다
    traveler_context는 조건에 넣지 않는다 (traveler_context_f1로 따로 본다).
분모가 0이면 값은 None이다.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Final

SENTIMENT_VALUES = ("positive", "negative", "neutral")
TASK_SUCCESS_MIN_OVERLAP: Final = 0.5  # 경계 차이는 허용하고 다른 문장을 근거로 쓴 경우만 실패로 본다 (2026-09-29 결정)


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def f1_score(tp: float, n_pred: float, n_gold: float) -> dict[str, float | None]:
    precision, recall = safe_div(tp, n_pred), safe_div(tp, n_gold)
    if precision is None or recall is None:
        f1 = None if not n_pred and not n_gold else 0.0
    else:
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------- evidence ----------


def _occurrences(text: str, span: str) -> list[int]:
    starts, start = [], text.find(span)
    while span and start != -1:
        starts.append(start)
        start = text.find(span, start + 1)
    return starts


def evidence_overlap_f1(pred: str, gold: str, review: str) -> float:
    pred_starts, gold_starts = _occurrences(review, pred), _occurrences(review, gold)
    if not pred_starts or not gold_starts:
        return 1.0 if pred and pred == gold else 0.0
    best = 0
    for p in pred_starts:
        for g in gold_starts:
            best = max(best, min(p + len(pred), g + len(gold)) - max(p, g))
    return 2 * best / (len(pred) + len(gold))


# ---------- 리뷰 하나 채점 ----------


def match_aspects(pred: list[dict], gold: list[dict], review: str) -> list[tuple[int, int]]:
    """(예측 index, 정답 index) 쌍 목록."""
    candidates = [
        (-evidence_overlap_f1(p["evidence"], g["evidence"], review), i, j)
        for i, p in enumerate(pred)
        for j, g in enumerate(gold)
        if p["category"] == g["category"]
    ]
    used_pred, used_gold, pairs = set(), set(), []
    for _, i, j in sorted(candidates):
        if i not in used_pred and j not in used_gold:
            used_pred.add(i)
            used_gold.add(j)
            pairs.append((i, j))
    return sorted(pairs)


def _aspect_key(aspect: dict) -> tuple[str, str, str, str]:
    return aspect["category"], aspect["attribute"], aspect["sentiment"], aspect["evidence"]


@dataclass
class ReviewScore:
    category: str
    n_pred: int
    n_gold: int
    pairs: list[dict]  # 짝지어진 쌍: {aspect, attribute_ok, sentiment_ok, gold_sentiment, pred_sentiment, exact, overlap}
    pred_in_source: int
    set_tp: int
    set_pred: int
    set_gold: int
    context_tp: int
    context_pred: int
    context_gold: int
    exact_match: bool
    gold_empty: bool
    pred_empty: bool
    per_aspect: Counter = field(default_factory=Counter)  # (aspect종류, "tp"|"pred"|"gold") → 개수


def score_review(pred_label: dict, gold_label: dict, review: str, category: str) -> ReviewScore:
    pred, gold = pred_label["aspects"], gold_label["aspects"]
    pairs = []
    for i, j in match_aspects(pred, gold, review):
        p, g = pred[i], gold[j]
        pairs.append({
            "aspect": g["category"],
            "attribute_ok": p["attribute"] == g["attribute"],
            "sentiment_ok": p["sentiment"] == g["sentiment"],
            "gold_sentiment": g["sentiment"],
            "pred_sentiment": p["sentiment"],
            "exact": p["evidence"] == g["evidence"],
            "overlap": evidence_overlap_f1(p["evidence"], g["evidence"], review),
        })  # fmt: skip

    per_aspect: Counter = Counter()
    for a in pred:
        per_aspect[f"{category}/{a['category']}", "pred"] += 1
    for a in gold:
        per_aspect[f"{category}/{a['category']}", "gold"] += 1
    for pair in pairs:
        per_aspect[f"{category}/{pair['aspect']}", "tp"] += 1

    pred_set, gold_set = {a["category"] for a in pred}, {a["category"] for a in gold}
    pred_ctx, gold_ctx = set(pred_label["traveler_context"]), set(gold_label["traveler_context"])
    return ReviewScore(
        category=category,
        n_pred=len(pred),
        n_gold=len(gold),
        pairs=pairs,
        pred_in_source=sum(1 for a in pred if a["evidence"] and a["evidence"] in review),
        set_tp=len(pred_set & gold_set),
        set_pred=len(pred_set),
        set_gold=len(gold_set),
        context_tp=len(pred_ctx & gold_ctx),
        context_pred=len(pred_ctx),
        context_gold=len(gold_ctx),
        exact_match=Counter(map(_aspect_key, pred)) == Counter(map(_aspect_key, gold)) and pred_ctx == gold_ctx,
        gold_empty=not gold,
        pred_empty=not pred,
        per_aspect=per_aspect,
    )


def task_checks(score: ReviewScore, format_ok: bool, min_overlap: float = TASK_SUCCESS_MIN_OVERLAP) -> dict[str, bool]:
    """과업 성공 조건별 통과 여부. 모두 True면 성공이다."""
    return {
        "format": format_ok,
        "aspects": score.n_pred == score.n_gold == len(score.pairs),
        "values": all(p["attribute_ok"] and p["sentiment_ok"] for p in score.pairs),
        "evidence": score.pred_in_source == score.n_pred and all(p["overlap"] >= min_overlap for p in score.pairs),
    }


# ---------- 모아서 지표 ----------


def _macro_f1(tp: Counter, n_pred: Counter, n_gold: Counter) -> float | None:
    classes = sorted(set(n_pred) | set(n_gold))
    scores = [f1_score(tp[c], n_pred[c], n_gold[c])["f1"] or 0.0 for c in classes]
    return safe_div(sum(scores), len(scores))


def aggregate(scores: list[ReviewScore]) -> dict:
    pairs = [pair for s in scores for pair in s.pairs]
    tp, n_pred, n_gold = len(pairs), sum(s.n_pred for s in scores), sum(s.n_gold for s in scores)
    aspect = f1_score(tp, n_pred, n_gold)

    per_aspect: Counter = sum((s.per_aspect for s in scores), Counter())
    kinds = {key for key, _ in per_aspect}
    aspect_macro = _macro_f1(
        Counter({k: per_aspect[k, "tp"] for k in kinds}),
        Counter({k: per_aspect[k, "pred"] for k in kinds if per_aspect[k, "pred"]}),
        Counter({k: per_aspect[k, "gold"] for k in kinds if per_aspect[k, "gold"]}),
    )
    sentiment_macro = _macro_f1(
        Counter(p["gold_sentiment"] for p in pairs if p["sentiment_ok"]),
        Counter(p["pred_sentiment"] for p in pairs),
        Counter(p["gold_sentiment"] for p in pairs),
    )
    gold_empty = [s for s in scores if s.gold_empty]
    return {
        "reviews": len(scores),
        "gold_aspects": n_gold,
        "pred_aspects": n_pred,
        "matched_aspects": tp,
        "aspect_precision": aspect["precision"],
        "aspect_recall": aspect["recall"],
        "aspect_f1": aspect["f1"],
        "aspect_macro_f1": aspect_macro,
        "aspect_set_f1": f1_score(sum(s.set_tp for s in scores), sum(s.set_pred for s in scores), sum(s.set_gold for s in scores))["f1"],
        "attribute_accuracy": safe_div(sum(p["attribute_ok"] for p in pairs), tp),
        "sentiment_accuracy": safe_div(sum(p["sentiment_ok"] for p in pairs), tp),
        "sentiment_macro_f1": sentiment_macro,
        "evidence_in_source_rate": safe_div(sum(s.pred_in_source for s in scores), n_pred),
        "evidence_exact_match": safe_div(sum(p["exact"] for p in pairs), tp),
        "evidence_overlap_f1": safe_div(sum(p["overlap"] for p in pairs), tp),
        "traveler_context_f1": f1_score(
            sum(s.context_tp for s in scores), sum(s.context_pred for s in scores), sum(s.context_gold for s in scores)
        )["f1"],
        "exact_review_match": safe_div(sum(s.exact_match for s in scores), len(scores)),
        "empty_review_accuracy": safe_div(sum(s.pred_empty for s in gold_empty), len(gold_empty)),
    }  # fmt: skip

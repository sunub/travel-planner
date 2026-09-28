import json

import pytest

from travel_planner.finetune.evaluation.evaluator import evaluate
from travel_planner.finetune.evaluation.metrics import aggregate, evidence_overlap_f1, match_aspects, score_review
from travel_planner.finetune.evaluation.parser import parse_prediction

REVIEW = "방은 깨끗했지만 조금 시끄러웠어요. 직원은 친절했어요."
GOLD = {
    "traveler_context": ["couple"],
    "aspects": [
        {"category": "cleanliness", "attribute": "clean", "sentiment": "positive", "evidence": "방은 깨끗했지만"},
        {"category": "noise_level", "attribute": "noisy", "sentiment": "negative", "evidence": "조금 시끄러웠어요"},
        {"category": "staff_service", "attribute": "friendly", "sentiment": "positive", "evidence": "직원은 친절했어요"},
    ],
}
RECORD = {"review_id": "r1", "place_id": "p1", "category": "hotel", "review": REVIEW, "label": GOLD, "tier": "gold", "synthetic": False}


def score(pred: dict) -> dict:
    return aggregate([score_review(pred, GOLD, REVIEW, "hotel")])


def test_perfect_prediction():
    m = score(GOLD)
    for key in ("aspect_f1", "attribute_accuracy", "sentiment_accuracy", "evidence_in_source_rate", "evidence_exact_match", "traveler_context_f1", "exact_review_match"):
        assert m[key] == 1.0, key


def test_missing_extra_and_wrong_values():
    pred = {
        "traveler_context": [],
        "aspects": [
            {"category": "cleanliness", "attribute": "dirty", "sentiment": "positive", "evidence": "방은 깨끗했지만"},
            {"category": "noise_level", "attribute": "noisy", "sentiment": "neutral", "evidence": "시끄러웠어요"},
            {"category": "view_quality", "attribute": "good", "sentiment": "positive", "evidence": "바다 전망"},
        ],
    }
    m = score(pred)
    assert m["matched_aspects"] == 2
    assert m["aspect_precision"] == pytest.approx(2 / 3)
    assert m["aspect_recall"] == pytest.approx(2 / 3)
    assert m["attribute_accuracy"] == pytest.approx(1 / 2)
    assert m["sentiment_accuracy"] == pytest.approx(1 / 2)
    assert m["evidence_in_source_rate"] == pytest.approx(2 / 3)  # "바다 전망"은 원문에 없다
    assert m["evidence_exact_match"] == pytest.approx(1 / 2)
    assert m["traveler_context_f1"] == 0.0
    assert m["exact_review_match"] == 0.0


def test_same_category_matched_by_evidence_overlap():
    review = "맛은 좋은데 양념은 별로였어요"
    gold = [
        {"category": "food_quality", "attribute": "good", "sentiment": "positive", "evidence": "맛은 좋은데"},
        {"category": "food_quality", "attribute": "poor", "sentiment": "negative", "evidence": "양념은 별로였어요"},
    ]
    pred = list(reversed(gold))
    assert match_aspects(pred, gold, review) == [(0, 1), (1, 0)]


def test_evidence_overlap_f1():
    review = "주차장이 넓고 편했어요"
    assert evidence_overlap_f1("주차장이 넓고", "주차장이 넓고", review) == 1.0
    assert evidence_overlap_f1("넓고", "주차장이 넓고", review) == pytest.approx(2 * 2 / (2 + 7))
    assert evidence_overlap_f1("없는 말", "주차장이 넓고", review) == 0.0


def test_empty_gold_review():
    empty = {"traveler_context": [], "aspects": []}
    silent = aggregate([score_review(empty, empty, "그냥 그랬어요", "hotel")])
    assert silent["empty_review_accuracy"] == 1.0 and silent["aspect_f1"] is None
    noisy = aggregate([score_review(GOLD, empty, REVIEW, "hotel")])
    assert noisy["empty_review_accuracy"] == 0.0 and noisy["aspect_precision"] == 0.0


def test_parser_handles_fences_and_reports_invalid_values():
    fenced = "```json\n" + json.dumps(GOLD, ensure_ascii=False) + "\n```"
    p = parse_prediction(fenced, "hotel", REVIEW)
    assert p.parsed and p.strict_valid

    bad = dict(GOLD, aspects=[dict(GOLD["aspects"][0], attribute="sparkling"), {"category": "noise_level"}])
    p = parse_prediction(json.dumps(bad, ensure_ascii=False), "hotel", REVIEW)
    assert p.parsed and not p.schema_valid and not p.strict_valid
    assert len(p.label_for_scoring()["aspects"]) == 1  # 필드가 빠진 aspect는 버리고 허용값 위반은 남긴다

    garbage = parse_prediction("모르겠습니다", "hotel", REVIEW)
    assert not garbage.parsed and garbage.label_for_scoring()["aspects"] == []


def test_evaluate_outputs():
    outputs = [json.dumps(GOLD, ensure_ascii=False), "not json"]
    result = evaluate([RECORD, dict(RECORD, review_id="r2")], outputs)
    overall = result["overall"]
    assert overall["json_parse_rate"] == 0.5 and overall["json_valid_rate"] == 0.5
    assert overall["aspect_precision"] == 1.0 and overall["aspect_recall"] == 0.5
    assert set(result["by_category"]) == {"hotel"}
    assert len(result["samples"]) == 2

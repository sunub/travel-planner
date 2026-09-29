"""두 run 예측의 리뷰 단위 비교 (evaluation/paired.py). 모델 없이 문자열 출력만 쓴다."""

import json

import pytest

from travel_planner.finetune.data.sft import target_text
from travel_planner.finetune.evaluation.paired import align_outputs, compare

from conftest import make_record


def records_and_outputs():
    records = [make_record(i, ("hotel", "restaurant", "attraction")[i % 3], f"p{i}") for i in range(6)]
    correct = [target_text(r["label"]) for r in records]
    # A는 앞 3건만 맞히고, B는 1건(0번)을 틀리는 대신 나머지를 맞힌다
    a = correct[:3] + ["{}"] * 3
    b = ["not json"] + correct[1:]
    return records, a, b


def test_transitions_and_rates():
    records, a, b = records_and_outputs()
    report = compare(records, a, b, n_resamples=50)["report"]
    t = report["transitions"]
    assert (t["improved"]["count"], t["regressed"]["count"], t["both_success"]["count"], t["both_fail"]["count"]) == (3, 1, 2, 0)
    assert t["regressed"]["review_ids"] == [records[0]["review_id"]]
    assert report["overall"]["a"]["task_success_rate"] == pytest.approx(3 / 6)
    assert report["overall"]["b"]["task_success_rate"] == pytest.approx(5 / 6)
    assert set(report["by_category"]) == {"hotel", "restaurant", "attraction"}


def test_bootstrap_is_reproducible_and_zero_for_identical_runs():
    records, a, b = records_and_outputs()
    first = compare(records, a, b, n_resamples=100, seed=3)["report"]["bootstrap"]
    again = compare(records, a, b, n_resamples=100, seed=3)["report"]["bootstrap"]
    assert first == again
    assert first["task_success_rate"]["diff"] == pytest.approx(2 / 6)

    same = compare(records, a, a, n_resamples=100)["report"]["bootstrap"]["task_success_rate"]
    assert same["diff"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0 and same["excludes_zero"] is False


def test_cases_keep_review_and_checks():
    records, a, b = records_and_outputs()
    cases = compare(records, a, b, n_resamples=10)["cases"]
    first = cases[0]
    assert first["transition"] == "regressed" and first["review"] == records[0]["review"]
    assert first["a"]["success"] and not first["b"]["success"] and first["b"]["checks"]["format"] is False
    json.dumps(cases, ensure_ascii=False)  # JSONL로 저장할 수 있어야 한다


def test_align_outputs_requires_same_review_ids():
    records, a, _ = records_and_outputs()
    outputs = {r["review_id"]: out for r, out in zip(records, a)}
    assert align_outputs(list(reversed(records)), outputs, "a")[0] == outputs[records[-1]["review_id"]]
    del outputs[records[0]["review_id"]]
    with pytest.raises(ValueError, match="review_id"):
        align_outputs(records, outputs, "a")

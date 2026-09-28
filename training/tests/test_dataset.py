import copy

import pytest

from travel_planner.finetune.data.dataset import DatasetError, load_records, record_issues, summarize

from conftest import GOLD_PATH, requires_gold


@requires_gold
def test_gold_file_loads_and_matches_schema():
    records = load_records(GOLD_PATH)  # 모든 레코드 검사를 통과해야 한다
    summary = summarize(records)
    assert summary["records"] == 1147
    assert summary["by_tier"] == {"gold": 1147}
    assert summary["synthetic"] == 0
    assert set(summary["by_category"]) == {"hotel", "restaurant", "attraction"}
    for record in records:
        assert set(record) == {"review_id", "place_id", "category", "synthetic", "review", "label", "tier"}
        assert set(record["label"]) == {"traveler_context", "aspects"}
        assert all(a["evidence"] in record["review"] for a in record["label"]["aspects"])


def test_record_checks(small_records):
    good = small_records[1]
    assert record_issues(good) == []

    bad_attribute = copy.deepcopy(good)
    bad_attribute["label"]["aspects"][0]["attribute"] = "amazing"
    assert any("attribute" in p for p in record_issues(bad_attribute))

    bad_evidence = copy.deepcopy(good)
    bad_evidence["label"]["aspects"][0]["evidence"] = "리뷰에 없는 말"
    assert any("evidence" in p for p in record_issues(bad_evidence))

    missing = {k: v for k, v in good.items() if k != "tier"}
    assert any("tier" in p for p in record_issues(missing))


def test_load_records_stops_on_problems(small_records, write_jsonl):
    rows = copy.deepcopy(small_records[:3])
    rows[2]["review_id"] = rows[1]["review_id"]
    with pytest.raises(DatasetError, match="중복"):
        load_records(write_jsonl(rows))

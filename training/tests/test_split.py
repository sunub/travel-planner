from collections import Counter

import pytest

from travel_planner.finetune.data.dataset import file_sha256, load_records
from travel_planner.finetune.data.split import (
    SPLITS,
    SplitError,
    apply_split,
    assign_places,
    build_manifest,
    check_no_leakage,
    filter_extra_train,
    load_manifest,
)

from conftest import GOLD_PATH, MANIFEST_PATH, make_record, requires_gold

RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def split_records(records, assignment):
    return {s: [r for r in records if assignment[r["place_id"]] == s] for s in SPLITS}


def test_group_split_has_no_place_leakage(small_records):
    assignment = assign_places(small_records, seed=42, ratios=RATIOS, large_place_min_reviews=10)
    splits = split_records(small_records, assignment)
    check_no_leakage(splits)
    assert all(splits[s] for s in SPLITS)


def test_split_is_reproducible_and_seed_dependent(small_records):
    a = assign_places(small_records, seed=42, ratios=RATIOS, large_place_min_reviews=10)
    b = assign_places(list(reversed(small_records)), seed=42, ratios=RATIOS, large_place_min_reviews=10)
    c = assign_places(small_records, seed=7, ratios=RATIOS, large_place_min_reviews=10)
    assert a == b
    assert a != c


def test_large_place_forced_to_train_and_targets_not_exceeded(small_records):
    assignment = assign_places(small_records, seed=42, ratios=RATIOS, large_place_min_reviews=10)
    assert assignment["hotel:big"] == "train"
    counts = Counter((r["category"], assignment[r["place_id"]]) for r in small_records)
    for category in ("hotel", "restaurant", "attraction"):
        total = sum(1 for r in small_records if r["category"] == category)
        assert counts[category, "test"] <= round(total * 0.1)
        assert counts[category, "validation"] <= round(total * 0.1)


def test_leakage_detected():
    splits = {"train": [make_record(0, "hotel", "p1")], "validation": [], "test": [make_record(1, "hotel", "p1")]}
    with pytest.raises(SplitError, match="p1"):
        check_no_leakage(splits)


def test_manifest_roundtrip_and_extra_train_filter(small_records):
    assignment = assign_places(small_records, seed=42, ratios=RATIOS, large_place_min_reviews=10)
    manifest = build_manifest(
        small_records, assignment, version="t", seed=42, ratios=RATIOS, large_place_min_reviews=10, source={}, git={}
    )
    splits = apply_split(small_records, manifest)
    assert {s: len(v) for s, v in splits.items()} == {s: manifest["counts"][s]["total"] for s in SPLITS}

    test_place = next(p for p, s in assignment.items() if s == "test")
    silver = [make_record(900, "hotel", test_place, tier="silver"), make_record(901, "hotel", "new:place", tier="silver")]
    kept, removed = filter_extra_train(silver, manifest)
    assert removed == 1 and [r["place_id"] for r in kept] == ["new:place"]

    with pytest.raises(SplitError, match="manifest"):
        apply_split(small_records + [make_record(999, "hotel", "unknown")], manifest)


@requires_gold
def test_saved_gold_manifest_is_consistent():
    manifest = load_manifest(MANIFEST_PATH)
    records = load_records(GOLD_PATH)
    splits = apply_split(records, manifest)  # leakage 검사 포함
    assert manifest["seed"] == 42 and manifest["group_key"] == "place_id"
    assert manifest["source"]["sha256"] == file_sha256(GOLD_PATH)
    assert sum(len(v) for v in splits.values()) == len(records)
    assert all(r["tier"] == "gold" for r in splits["test"])
    for category in ("hotel", "restaurant", "attraction"):
        n_test = sum(1 for r in splits["test"] if r["category"] == category)
        n_total = sum(1 for r in records if r["category"] == category)
        assert 0.08 <= n_test / n_total <= 0.12

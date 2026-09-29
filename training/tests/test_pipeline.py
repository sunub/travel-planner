"""data.split_files: 이미 나뉜 train / validation / test 파일을 그대로 읽는 경로."""

import pytest

from travel_planner.finetune.data.dataset import DatasetError
from travel_planner.finetune.data.pipeline import DataPolicyError, load_splits
from travel_planner.finetune.data.split import SplitError

from conftest import make_record


@pytest.fixture
def split_config(tmp_config, write_jsonl):
    def _config(train, validation, test):
        files = {name: write_jsonl(rows, f"{name}.jsonl") for name, rows in (("train", train), ("validation", validation), ("test", test))}
        return tmp_config(
            "qlora_busan_v2.yaml", **{f"split_files.{name}": path for name, path in files.items()}
        )

    return _config


def test_split_files_are_read_as_given(split_config):
    train = [make_record(i, "hotel", f"train:{i}", tier="silver") for i in range(3)]
    validation = [make_record(10, "restaurant", "val:0", tier="silver")]
    test = [make_record(20, "attraction", "test:0")]
    splits, info = load_splits(split_config(train, validation, test))
    assert info["counts"] == {"train": 3, "validation": 1, "test": 1}
    assert info["split_version"] == "busan_review_sft_v1" and info["dataset_version"] == "busan_review_sft_v1"
    assert set(info["split_files"]) == {"train", "validation", "test"} and all(len(f["sha256"]) == 64 for f in info["split_files"].values())
    assert info["tiers"]["test"] == ["gold"]
    assert [r["review_id"] for r in splits["train"]] == [r["review_id"] for r in train]


def test_split_files_reject_leakage_duplicates_and_silver_test(split_config):
    with pytest.raises(SplitError, match="same|같은 place_id"):
        load_splits(split_config([make_record(0, "hotel", "p1", tier="silver")], [], [make_record(1, "hotel", "p1")]))
    with pytest.raises(DatasetError, match="review_id"):
        load_splits(split_config([make_record(0, "hotel", "p1", tier="silver")], [], [make_record(0, "hotel", "p2")]))
    with pytest.raises(DataPolicyError, match="Gold"):
        load_splits(split_config([make_record(0, "hotel", "p1", tier="silver")], [], [make_record(1, "hotel", "p2", tier="silver")]))

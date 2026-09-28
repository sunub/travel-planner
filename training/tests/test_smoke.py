"""샘플 8건으로 GPU 없이 파이프라인 전체를 돌린다: 분할 → SFT 변환 → (가짜) 출력 → 채점 → metrics.json → 비교표.

실제 Gold가 있으면 앞 8건을, 없으면 합성 fixture를 쓴다. 모델은 불러오지 않는다.
"""

import json

from travel_planner.finetune.data.dataset import read_jsonl
from travel_planner.finetune.data.pipeline import load_splits
from travel_planner.finetune.data.split import assign_places, build_manifest, save_manifest
from travel_planner.finetune.data.sft import target_text
from travel_planner.finetune.evaluation.evaluator import evaluate
from travel_planner.finetune.training.trainer import build_examples, train
from travel_planner.finetune.utils.compare import find_metrics, render_table, rows_from
from travel_planner.finetune.utils.tracking import base_metrics, create_run, write_json, write_summary

from conftest import GOLD_PATH, make_record

N_SAMPLES = 8


def sample_records() -> list[dict]:
    if GOLD_PATH.exists():
        records = read_jsonl(GOLD_PATH)
        by_place = {}
        for r in records:  # 장소가 서로 다른 8건 (분할이 세 쪽으로 나뉘도록)
            by_place.setdefault(r["place_id"], r)
        return list(by_place.values())[:N_SAMPLES]
    return [make_record(i, ("hotel", "restaurant", "attraction")[i % 3], f"p{i}") for i in range(N_SAMPLES)]


def test_end_to_end_smoke(tmp_path, tmp_config, write_jsonl):
    records = sample_records()
    dataset = write_jsonl(records, "smoke.jsonl")
    ratios = {"train": 0.5, "validation": 0.25, "test": 0.25}
    assignment = assign_places(records, seed=42, ratios=ratios, large_place_min_reviews=None)
    manifest_path = tmp_path / "smoke_split.json"
    save_manifest(
        build_manifest(records, assignment, version="smoke", seed=42, ratios=ratios, large_place_min_reviews=None, source={}, git={}),
        manifest_path,
    )

    config = tmp_config("qlora_gold_v1.yaml", dataset_path=dataset, split_manifest=manifest_path)
    splits, data_info = load_splits(config)
    assert sum(data_info["counts"].values()) == N_SAMPLES and data_info["split_version"] == "smoke"
    assert all(data_info["counts"][s] > 0 for s in ("train", "validation", "test"))

    examples = build_examples(config, splits)
    assert len(examples["train"]) == data_info["counts"]["train"]
    first = examples["train"][0]
    assert json.loads(first["completion"][0]["content"]) == splits["train"][0]["label"]

    assert train(config, dry_run=True) is None  # dry-run은 모델 없이 끝난다

    # 모델 대신 정답을 그대로 낸 출력과 깨진 출력으로 채점 경로를 확인한다
    test = splits["test"]
    outputs = [target_text(r["label"]) for r in test[:-1]] + ["{broken"]
    result = evaluate(test, outputs)
    assert result["overall"]["json_parse_rate"] == (len(test) - 1) / len(test)

    run = create_run(dict(config, experiment_name="smoke"))
    metrics = base_metrics(config, run, data_info=data_info, git={"commit": None, "dirty": False})
    metrics["evaluation"] = {"split": "test", "num_samples": len(test), "overall": result["overall"], "by_category": result["by_category"]}
    write_json(run.experiment_dir / "metrics.json", metrics)
    write_summary(metrics, run.experiment_dir / "summary.md")
    assert "aspect_f1" in (run.experiment_dir / "summary.md").read_text(encoding="utf-8")

    table = render_table(rows_from(find_metrics([tmp_path / "experiments"])))
    assert "smoke_v001" in table

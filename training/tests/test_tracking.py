import csv

from travel_planner.finetune.utils.compare import find_metrics, render_table, rows_from, write_csv
from travel_planner.finetune.utils.tracking import adapter_size_mb, base_metrics, create_run, next_run_name, write_json


def test_next_run_name(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "lora_gold_v001").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "lora_gold_v003").mkdir()
    (tmp_path / "b" / "qlora_gold_v009").mkdir()
    assert next_run_name("lora_gold", [tmp_path / "a", tmp_path / "b"]) == "lora_gold_v004"
    assert next_run_name("base", [tmp_path / "a"]) == "base_v001"


def test_runs_are_versioned_and_metrics_have_common_shape(tmp_config, tmp_path):
    config = tmp_config("lora_gold_v1.yaml")
    first, second = create_run(config), create_run(config)
    assert (first.name, second.name) == ("lora_gold_v001", "lora_gold_v002")
    assert first.artifact_dir.parent == tmp_path / "artifacts"
    assert first.experiment_dir.parent == tmp_path / "experiments"

    data_info = {"counts": {"train": 1, "validation": 1, "test": 1}}
    metrics = base_metrics(config, first, data_info=data_info, git={"commit": "abc", "dirty": False})
    for key in ("experiment_name", "timestamp", "git", "model_id", "method", "seed", "data", "hyperparameters", "training", "evaluation"):
        assert key in metrics
    assert metrics["hyperparameters"]["lora"]["r"] == 16


def test_adapter_size_counts_weight_files_only(tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"0" * 2**20)
    (tmp_path / "tokenizer.json").write_bytes(b"0" * 2**20)
    assert adapter_size_mb(tmp_path) == 1.0


def test_compare_table_and_csv(tmp_path):
    for name, f1, vram in (("base_v001", 0.2, None), ("qlora_gold_v001", 0.7, 7.5)):
        write_json(tmp_path / name / "metrics.json", {
            "run_name": name, "method": name.split("_")[0], "data": {},
            "training": {"peak_vram_reserved_gb": vram, "duration_sec": 600} if vram else None,
            "evaluation": {"overall": {"aspect_f1": f1, "json_valid_rate": 1.0}},
        })  # fmt: skip
    rows = rows_from(find_metrics([tmp_path]))
    table = render_table(rows)
    assert "qlora_gold_v001" in table and "0.7000" in table and "10.0" in table
    write_csv(rows, tmp_path / "cmp.csv")
    with open(tmp_path / "cmp.csv", encoding="utf-8-sig") as f:
        assert [r["experiment"] for r in csv.DictReader(f)] == ["base_v001", "qlora_gold_v001"]

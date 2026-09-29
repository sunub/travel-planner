"""evaluate.py의 실행 경로(run_evaluation)를 모델 없이 확인한다. 모델 로딩과 생성만 가짜로 바꾼다."""

import yaml

import travel_planner.finetune.inference.run_eval as run_eval
import travel_planner.finetune.training.model as model_module
from travel_planner.finetune.data.sft import target_text
from travel_planner.finetune.data.split import assign_places, build_manifest, save_manifest
from travel_planner.finetune.utils.tracking import read_json

from conftest import make_record


def test_base_evaluation_saves_config_and_metrics(tmp_path, tmp_config, write_jsonl, monkeypatch):
    records = [make_record(i, ("hotel", "restaurant", "attraction")[i % 3], f"p{i}") for i in range(8)]
    ratios = {"train": 0.5, "validation": 0.25, "test": 0.25}
    manifest_path = tmp_path / "split.json"
    assignment = assign_places(records, seed=42, ratios=ratios, large_place_min_reviews=None)
    save_manifest(build_manifest(records, assignment, version="t", seed=42, ratios=ratios, large_place_min_reviews=None, source={}, git={}), manifest_path)

    monkeypatch.setattr(model_module, "load_tokenizer", lambda config: None)
    monkeypatch.setattr(model_module, "load_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_module, "resolve_precision", lambda requested: "bf16")
    monkeypatch.setattr(
        run_eval, "generate_with_stats", lambda model, tok, recs, templates, config: ([target_text(r["label"]) for r in recs], {"duration_sec": 1.0})
    )

    config = tmp_config("base_eval.yaml", dataset_path=write_jsonl(records), split_manifest=manifest_path)
    metrics_path = run_eval.run_evaluation(config)

    run_dir = metrics_path.parent
    assert run_dir.name == "base_v001"
    saved = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert saved["method"] == "base" and saved["model"]["name_or_path"] == "google/gemma-4-E4B-it"
    metrics = read_json(metrics_path)
    assert metrics["evaluation"]["overall"]["aspect_f1"] == 1.0
    assert (tmp_path / "artifacts" / "base_v001" / "predictions" / "test.jsonl").exists()


def test_4bit_base_evaluation_loads_quantized_and_records_it(tmp_path, tmp_config, write_jsonl, monkeypatch):
    records = [make_record(i, ("hotel", "restaurant", "attraction")[i % 3], f"p{i}") for i in range(8)]
    ratios = {"train": 0.5, "validation": 0.25, "test": 0.25}
    manifest_path = tmp_path / "split.json"
    assignment = assign_places(records, seed=42, ratios=ratios, large_place_min_reviews=None)
    save_manifest(build_manifest(records, assignment, version="t", seed=42, ratios=ratios, large_place_min_reviews=None, source={}, git={}), manifest_path)

    loaded = {}
    monkeypatch.setattr(model_module, "load_tokenizer", lambda config: None)
    monkeypatch.setattr(model_module, "load_model", lambda config, precision, **kwargs: loaded.update(kwargs))
    monkeypatch.setattr(model_module, "resolve_precision", lambda requested: "bf16")
    monkeypatch.setattr(
        run_eval, "generate_with_stats", lambda model, tok, recs, templates, config: ([target_text(r["label"]) for r in recs], {"duration_sec": 1.0})
    )

    config = tmp_config("base4bit_eval.yaml", dataset_path=write_jsonl(records), split_manifest=manifest_path)
    metrics = read_json(run_eval.run_evaluation(config))
    assert loaded["quantized"] is True and loaded["adapter_path"] is None
    assert metrics["run_name"] == "base4bit_v001" and metrics["method"] == "base"
    assert metrics["hyperparameters"]["quantization"]["bnb_4bit_quant_type"] == "nf4"
    assert metrics["hyperparameters"]["lora"] is None and metrics["training"] is None


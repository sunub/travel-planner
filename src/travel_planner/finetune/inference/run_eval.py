"""Base / LoRA / QLoRA 모델을 같은 split(기본 Gold Test)에서 생성 → 채점 → metrics.json에 기록.

어댑터 경로는 어디든 된다 (저장소 안 artifacts/가 아니어도). 어댑터 폴더의 tripfit_adapter.json에서
base 모델 id와 학습 방식(lora/qlora)을 읽으므로 다른 브랜치 · 다른 PC의 어댑터도 같은 방식으로 평가한다.
"""

import copy
import json
from pathlib import Path

from ..config import require_model_id, validate_experiment
from ..data.dataset import write_jsonl
from ..data.pipeline import load_splits
from ..data.sft import PromptTemplates, build_messages
from ..evaluation.evaluator import evaluate
from ..utils import gpu
from ..utils.tracking import (
    RunPaths,
    base_metrics,
    create_run,
    git_info,
    open_run,
    read_json,
    save_config,
    write_json,
    write_summary,
)

ADAPTER_METADATA = "tripfit_adapter.json"


def evaluation_key(split: str) -> str:
    return "evaluation" if split == "test" else f"evaluation_{split}"


def evaluate_loaded_model(model, tokenizer, records: list[dict], config: dict, run: RunPaths, *, split: str, adapter_path: Path | None) -> dict:
    """이미 불러온 모델로 생성 · 채점하고 예측을 artifacts에 남긴다. metrics.json의 evaluation 묶음을 돌려준다."""
    templates = PromptTemplates.from_config(config)
    inference = config["inference"]
    gpu.reset_peak_memory()
    outputs, stats = generate_with_stats(model, tokenizer, records, templates, config)
    result = evaluate(records, outputs)
    predictions_path = run.prediction_dir / f"{split}.jsonl"
    write_jsonl(predictions_path, result["samples"])
    return {
        "split": split,
        "num_samples": len(records),
        "adapter_path": str(adapter_path) if adapter_path else None,
        "decoding": {"do_sample": False, "max_new_tokens": inference["max_new_tokens"], "batch_size": inference["batch_size"]},
        "chat_template_kwargs": config["prompt"].get("chat_template_kwargs") or {},
        "overall": result["overall"],
        "by_category": result["by_category"],
        "inference": {**stats, **gpu.peak_memory_gb()},
        "predictions_path": str(predictions_path),
    }


def generate_with_stats(model, tokenizer, records, templates, config) -> tuple[list[str], dict]:
    from .generate import generate_outputs

    inference = config["inference"]
    return generate_outputs(
        model,
        tokenizer,
        [build_messages(r, templates) for r in records],
        max_new_tokens=inference["max_new_tokens"],
        batch_size=inference["batch_size"],
        chat_template_kwargs=config["prompt"].get("chat_template_kwargs"),
    )


def read_adapter_metadata(adapter_path: Path) -> dict:
    path = adapter_path / ADAPTER_METADATA
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_evaluation(config: dict, *, adapter_path: Path | None = None, run_name: str | None = None, split: str = "test") -> Path:
    """평가를 실행하고 metrics.json 경로를 돌려준다.

    - adapter 없음: Base 모델 평가 (method=base).
    - adapter 있음: 어댑터 평가. run_name을 주면 그 run의 metrics.json에 결과를 더하고, 없으면 새 run을 만든다.
    """
    from ..training.model import load_model, load_tokenizer, resolve_precision

    config = copy.deepcopy(config)
    validate_experiment(config)
    adapter_meta = read_adapter_metadata(adapter_path) if adapter_path else {}
    if adapter_path:
        if not config["model"].get("name_or_path") and adapter_meta.get("base_model_id"):
            config["model"]["name_or_path"] = adapter_meta["base_model_id"]
        config["method"] = adapter_meta.get("method", config["method"])
        if adapter_meta.get("quantization"):  # QLoRA 어댑터는 학습 때와 같은 4bit 설정으로 base를 불러온다
            config["quantization"] = adapter_meta["quantization"]
        if config["method"] == "base":
            raise ValueError("어댑터 평가인데 method가 base입니다. 어댑터를 만든 lora/qlora 설정을 --config로 주세요.")
        if not run_name:
            config["experiment_name"] = f"{config['experiment_name']}_eval"
    elif config["method"] != "base":
        raise ValueError("Base 평가에는 method: base 설정(base_eval.yaml)을, LoRA/QLoRA 평가에는 --adapter를 주세요.")
    require_model_id(config)

    splits, data_info = load_splits(config)
    records = splits[split]
    precision = resolve_precision(config.get("precision", "auto"))

    if run_name:
        run = open_run(config, run_name)
        metrics = read_json(run.experiment_dir / "metrics.json")
        # 학습 때의 config.yaml은 그대로 두고, 평가에 쓴 설정을 따로 남긴다
        save_config(config, run.experiment_dir / f"config_{evaluation_key(split)}.yaml")
    else:
        run = create_run(config)
        metrics = base_metrics(config, run, data_info=data_info, git=git_info())
        metrics["precision"] = precision
        save_config(config, run.experiment_dir / "config.yaml")
    metrics["device"] = gpu.device_info()

    print(f"[{run.name}] {split} {len(records)}건 평가 · 모델 {config['model']['name_or_path']} · {config['method']} · {precision}")
    tokenizer = load_tokenizer(config)
    model = load_model(config, precision, quantized=config["method"] == "qlora", adapter_path=adapter_path)
    metrics[evaluation_key(split)] = evaluate_loaded_model(model, tokenizer, records, config, run, split=split, adapter_path=adapter_path)

    metrics_path = run.experiment_dir / "metrics.json"
    write_json(metrics_path, metrics)
    write_summary(metrics, run.experiment_dir / "summary.md")
    return metrics_path

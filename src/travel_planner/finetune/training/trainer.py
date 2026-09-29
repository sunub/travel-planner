"""LoRA / QLoRA SFT 학습 한 번 (= run 하나).

순서: 데이터 준비 → run 디렉터리 → tokenizer · 길이 검사 → 모델 → SFTTrainer 학습 → 어댑터 저장
      → (설정 시) Gold Test 생성 · 채점 → metrics.json · summary.md
dry_run이면 데이터 준비와 예제 확인까지만 하고 모델은 불러오지 않는다.
"""

import copy
import gc
import shutil
import time
from pathlib import Path

from ..config import ConfigError, require_model_id, resolve_path, validate_experiment
from ..data.pipeline import load_splits
from ..data.sft import PromptTemplates, to_sft_example
from ..utils import gpu
from ..utils.tracking import (
    RunPaths,
    adapter_size_mb,
    base_metrics,
    create_run,
    git_info,
    now_iso,
    save_config,
    write_json,
    write_loss_history,
    write_summary,
)
from ..inference.run_eval import ADAPTER_METADATA, evaluate_loaded_model, evaluation_key


def build_examples(config: dict, splits: dict[str, list[dict]]) -> dict[str, list[dict]]:
    templates = PromptTemplates.from_config(config)
    kwargs = config["prompt"].get("chat_template_kwargs") or {}
    return {name: [to_sft_example(r, templates, kwargs) for r in splits[name]] for name in ("train", "validation")}


def sft_config(config: dict, run: RunPaths, precision: str, has_validation: bool):
    from trl import SFTConfig

    t = config["training"]
    return SFTConfig(
        output_dir=str(run.checkpoint_dir),
        seed=config["seed"],
        data_seed=config["seed"],
        learning_rate=t["learning_rate"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        max_length=t["max_length"],
        warmup_steps=t["warmup_steps"],
        weight_decay=t["weight_decay"],
        lr_scheduler_type=t["lr_scheduler_type"],
        optim=t["optim"],
        max_grad_norm=t.get("max_grad_norm", 1.0),
        gradient_checkpointing=t["gradient_checkpointing"],
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"] if has_validation else "no",
        save_strategy=t["save_strategy"],
        save_total_limit=t.get("save_total_limit"),
        report_to=t.get("report_to", "none"),
        completion_only_loss=True,  # prompt(system + user)는 loss에서 빼고 assistant 응답만 학습한다
        packing=False,
        **(t.get("extra_args") or {}),
    )


def save_adapter(trainer, config: dict, run: RunPaths, precision: str, data_info: dict, git: dict) -> None:
    """어댑터 + 이 어댑터를 쓰는 데 필요한 정보를 한 폴더에. 폴더만 복사하면 다른 브랜치에서도 쓸 수 있다."""
    trainer.save_model(str(run.adapter_dir))
    prompt_dir = run.adapter_dir / "prompt"
    prompt_dir.mkdir(exist_ok=True)
    shutil.copy(resolve_path(config["prompt"]["system_template"]), prompt_dir / "system.txt")
    shutil.copy(resolve_path(config["prompt"]["user_template"]), prompt_dir / "user.txt")
    write_json(run.adapter_dir / ADAPTER_METADATA, {
        "run_name": run.name,
        "created_at": now_iso(),
        "base_model_id": config["model"]["name_or_path"],
        "base_model_revision": config["model"].get("revision"),
        "method": config["method"],
        "quantization": config.get("quantization") if config["method"] == "qlora" else None,
        "precision": precision,
        "prompt_files": {"system": "prompt/system.txt", "user": "prompt/user.txt"},
        "chat_template_kwargs": config["prompt"].get("chat_template_kwargs") or {},
        "label_schema": "utils/common/schema.py",
        "dataset_version": data_info["dataset_version"],
        "split_version": data_info["split_version"],
        "git": git,
    })  # fmt: skip


def selected_checkpoint(state, args) -> dict:
    """저장 · 평가한 어댑터가 어느 체크포인트인가. load_best_model_at_end면 Validation 기준 best, 아니면 마지막 step."""
    if args.load_best_model_at_end and state.best_model_checkpoint:
        name = Path(state.best_model_checkpoint).name
        return {"selected_by": args.metric_for_best_model, "selected_checkpoint": name,
                "selected_step": int(name.rsplit("-", 1)[-1]), "selected_metric": state.best_metric}  # fmt: skip
    return {"selected_by": "last_step", "selected_checkpoint": None, "selected_step": state.global_step, "selected_metric": None}


def restore_adapter_dtypes(model, dtypes: dict) -> int:
    """학습 전의 어댑터 dtype으로 되돌린다. 바꾼 파라미터 수를 돌려준다.

    load_best_model_at_end는 끝에 best 체크포인트를 model.load_adapter()로 다시 불러오는데, PEFT가 기본값
    (autocast_adapter_dtype=True)으로 어댑터를 fp32로 올린다. QLoRA 어댑터는 TRL이 bf16으로 학습 · 저장하므로
    (체크포인트도 bf16), 되돌리지 않으면 저장 어댑터와 Test 평가만 fp32가 된다. bf16 → fp32 → bf16은 값이 그대로다.
    """
    changed = 0
    for name, param in model.named_parameters():
        dtype = dtypes.get(name)
        if dtype is not None and param.dtype != dtype:
            param.data = param.data.to(dtype)
            changed += 1
    return changed


def last_value(log_history: list[dict], key: str) -> float | None:
    return next((entry[key] for entry in reversed(log_history) if key in entry), None)


def train(config: dict, *, run_name: str | None = None, dry_run: bool = False) -> Path | None:
    config = copy.deepcopy(config)
    validate_experiment(config)
    if config["method"] == "base":
        raise ConfigError("method: base는 학습하지 않습니다. Base 평가는 evaluate.py를 쓰세요.")
    splits, data_info = load_splits(config)
    examples = build_examples(config, splits)
    print(f"데이터: {data_info['counts']} (split {data_info['split_version']})")
    if dry_run:
        sample = examples["train"][0]
        print("\n--- 첫 학습 예제 (prompt) ---")
        for message in sample["prompt"]:
            print(f"[{message['role']}]\n{message['content']}\n")
        print(f"--- completion ---\n{sample['completion'][0]['content']}")
        print("\ndry-run: 모델을 불러오지 않고 끝냅니다.")
        return None

    from datasets import Dataset
    from transformers import set_seed
    from trl import SFTTrainer

    from .model import check_lora_targets, count_parameters, load_model, load_tokenizer, lora_config, resolve_precision, token_lengths

    require_model_id(config)
    set_seed(config["seed"])
    precision = resolve_precision(config.get("precision", "auto"))
    git = git_info()
    run = create_run(config, run_name)
    save_config(config, run.experiment_dir / "config.yaml")
    metadata = {"run_name": run.name, "status": "running", "started_at": now_iso(), "git": git, "device": gpu.device_info(), "precision": precision}
    write_json(run.artifact_dir / "run_metadata.json", metadata)
    print(f"[{run.name}] {config['method']} · {config['model']['name_or_path']} · {precision}")
    print(f"  artifacts: {run.artifact_dir}\n  experiment: {run.experiment_dir}")

    tokenizer = load_tokenizer(config)
    max_length = config["training"]["max_length"]
    lengths = token_lengths(tokenizer, examples["train"] + examples["validation"])
    too_long = sum(1 for n in lengths if n > max_length)
    length_report = {"max_tokens": max(lengths), "mean_tokens": round(sum(lengths) / len(lengths), 1), "over_max_length": too_long}
    print(f"  토큰 길이: {length_report}")
    if too_long:
        raise ConfigError(f"max_length({max_length})보다 긴 예제가 {too_long}건 있습니다. 잘리면 정답 JSON이 잘리므로 멈춥니다.")

    model = load_model(config, precision, quantized=config["method"] == "qlora")
    if config["method"] == "qlora" and config["quantization"].get("prepare_for_kbit_training"):
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config["training"]["gradient_checkpointing"])

    has_validation = bool(examples["validation"])
    trainer = SFTTrainer(
        model=model,
        args=sft_config(config, run, precision, has_validation),
        train_dataset=Dataset.from_list(examples["train"]),
        eval_dataset=Dataset.from_list(examples["validation"]) if has_validation else None,
        processing_class=tokenizer,
        peft_config=lora_config(config),
    )
    wrapped = check_lora_targets(trainer.model)
    params = count_parameters(trainer.model)
    print(f"  LoRA 모듈 {len(wrapped)}개 · 학습 파라미터 {params['trainable_params']:,}")

    gpu.reset_peak_memory()
    start = time.perf_counter()
    adapter_dtypes = {name: p.dtype for name, p in trainer.model.named_parameters() if p.requires_grad}
    train_output = trainer.train()
    duration = time.perf_counter() - start
    restore_adapter_dtypes(trainer.model, adapter_dtypes)
    peak = gpu.peak_memory_gb()

    selected = selected_checkpoint(trainer.state, trainer.args)
    print(f"  선택한 체크포인트: {selected['selected_checkpoint'] or '마지막 step'} (step {selected['selected_step']}, 기준 {selected['selected_by']})")
    save_adapter(trainer, config, run, precision, data_info, git)
    final_eval = trainer.evaluate() if has_validation else {}
    log_history = trainer.state.log_history
    write_json(run.log_dir / "log_history.json", log_history)
    write_loss_history(log_history, run.experiment_dir / "loss_history.csv")

    metrics = base_metrics(config, run, data_info=data_info, git=git)
    metrics["precision"] = precision
    metrics["device"] = metadata["device"]
    metrics["training"] = {
        "final_train_loss": last_value(log_history, "loss"),
        "mean_train_loss": train_output.training_loss,
        "eval_loss": final_eval.get("eval_loss"),
        "best_eval_loss": min((e["eval_loss"] for e in log_history if "eval_loss" in e), default=None),
        "duration_sec": round(duration, 1),
        "global_steps": trainer.state.global_step,
        **selected,
        **peak,
        "adapter_size_mb": adapter_size_mb(run.adapter_dir),
        "lora_modules": len(wrapped),
        **params,
        "token_lengths": length_report,
        "adapter_path": str(run.adapter_dir),
    }
    metrics_path = run.experiment_dir / "metrics.json"
    write_json(metrics_path, metrics)  # 평가 전에 한 번 저장: 평가가 실패해도 학습 기록은 남는다

    evaluation = config.get("evaluation") or {}
    if evaluation.get("run_after_train"):
        split = evaluation.get("split", "test")
        print(f"  {split} 평가 시작")
        trainer.optimizer = trainer.lr_scheduler = None  # 생성할 때 optimizer state가 VRAM을 차지하지 않게 한다
        gc.collect()
        metrics[evaluation_key(split)] = evaluate_loaded_model(
            trainer.model, tokenizer, splits[split], config, run, split=split, adapter_path=run.adapter_dir
        )
        write_json(metrics_path, metrics)
    write_summary(metrics, run.experiment_dir / "summary.md")
    write_json(run.artifact_dir / "run_metadata.json", {**metadata, "status": "finished", "finished_at": now_iso()})
    print(f"완료: {metrics_path}")
    return metrics_path

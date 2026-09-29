"""EXAONE-3.5-7.8B-Instruct QLoRA 학습.

이 스크립트를 명시적으로 실행하기 전까지는 모델을 내려받거나 학습을 시작하지 않는다 (--dry-run은
모델을 아예 불러오지 않는다). 데이터셋이 아직 확정되지 않았으므로 config.yaml의
data.{train_path,validation_path,test_path}가 비어 있으면 무엇을 채워야 하는지 알려주고 멈춘다.

인증은 저장소 루트의 .env(HF_TOKEN, HF_HOME)를 python-dotenv로 읽는다. 토큰 값은 로그에 출력하지 않는다.

사용법:
  uv run python src/travel_planner/model-yay/train.py --config src/travel_planner/model-yay/config.yaml --dry-run
  uv run python src/travel_planner/model-yay/train.py --config src/travel_planner/model-yay/config.yaml
  uv run python src/travel_planner/model-yay/train.py --config src/travel_planner/model-yay/config.yaml --max-samples 8
"""

import argparse
import json
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

from data import load_records, to_sft_example

MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODEL_DIR.parents[2]
IGNORE_KEYS_IN_LOG = frozenset({"HF_TOKEN"})  # .env 값은 절대 로그에 남기지 않는다


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def resolve_output_dir(config: dict) -> Path:
    path = Path(config["output"]["artifacts_root"])
    return path if path.is_absolute() else REPO_ROOT / path


def require_data_paths(config: dict) -> dict:
    """데이터셋 경로가 비어 있으면 무엇을 채워야 하는지 알려주고 멈춘다."""
    data = config["data"]
    missing = [key for key in ("train_path", "validation_path", "test_path") if not data.get(key)]
    if missing:
        names = ", ".join(f"data.{key}" for key in missing)
        raise SystemExit(f"데이터셋 경로가 아직 설정되지 않았습니다. config.yaml에서 다음 값을 채우세요: {names}")
    return data


def find_lora_target_modules(model) -> list[str]:
    """모델의 nn.Linear 층 이름(리프 이름)을 자동으로 찾는다.

    EXAONE은 Gemma와 층 이름이 다르므로 하드코딩하지 않고, 실제로 불러온 모델을 순회해서 찾는다.
    lm_head는 어휘 크기만큼 큰 출력층이라 LoRA 대상에서 뺀다.
    """
    import torch.nn as nn

    exclude = {"lm_head"}
    names = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] not in exclude
    }
    return sorted(names)


def load_tokenizer_and_model(config: dict, hf_token: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_cfg = config["model"]
    trust_remote_code = model_cfg.get("trust_remote_code", True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        revision=model_cfg.get("revision"),
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )
    if tokenizer.chat_template is None:
        raise RuntimeError(f"{model_cfg['name_or_path']}의 tokenizer에 chat template이 없습니다.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = config["quantization"]
    compute_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[quant.get("bnb_4bit_compute_dtype", "bf16")]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant["load_in_4bit"],
        bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        revision=model_cfg.get("revision"),
        trust_remote_code=trust_remote_code,
        quantization_config=bnb_config,
        attn_implementation=model_cfg.get("attn_implementation"),
        device_map={"": 0},
        token=hf_token,
    )
    return tokenizer, model


def build_peft_model(model, config: dict):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config["training"]["gradient_checkpointing"]
    )
    target_modules = find_lora_target_modules(model)
    print(f"LoRA target_modules (자동 탐색): {target_modules}")
    lora = config["lora"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=target_modules,
        bias=lora.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    return peft_model, target_modules


def build_sft_config(config: dict, checkpoint_dir: Path, has_validation: bool):
    from trl import SFTConfig

    t = config["training"]
    return SFTConfig(
        output_dir=str(checkpoint_dir),
        seed=config["seed"],
        data_seed=config["seed"],
        learning_rate=t["learning_rate"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", t["per_device_train_batch_size"]),
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        max_length=t["max_seq_length"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t.get("weight_decay", 0.0),
        lr_scheduler_type=t["lr_scheduler_type"],
        max_grad_norm=t.get("max_grad_norm", 1.0),
        optim=t["optim"],
        gradient_checkpointing=t["gradient_checkpointing"],
        bf16=True,
        logging_steps=t.get("logging_steps", 10),
        eval_strategy=t.get("eval_strategy", "epoch") if has_validation else "no",
        save_strategy=t.get("save_strategy", "epoch"),
        save_total_limit=t.get("save_total_limit", 2),
        report_to="none",
        completion_only_loss=True,  # prompt(system+user)는 loss에서 빼고 assistant 응답만 학습한다
        packing=False,
    )


def peak_vram_gb() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"peak_vram_allocated_gb": None, "peak_vram_reserved_gb": None}
    return {
        "peak_vram_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }


def last_value(log_history: list[dict], key: str) -> float | None:
    return next((entry[key] for entry in reversed(log_history) if key in entry), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="데이터와 첫 학습 예제만 보여주고 모델은 불러오지 않는다")
    parser.add_argument("--max-samples", type=int, help="train/validation 앞에서 N건만 (스모크 실행)")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    config = load_config(args.config)
    data_paths = require_data_paths(config)

    train_records = load_records(data_paths["train_path"])
    val_records = load_records(data_paths["validation_path"])
    if args.max_samples:
        train_records = train_records[: args.max_samples]
        val_records = val_records[: args.max_samples]
    print(f"데이터: train {len(train_records)}건 / validation {len(val_records)}건")

    train_examples = [to_sft_example(r) for r in train_records]
    val_examples = [to_sft_example(r) for r in val_records]

    if args.dry_run:
        sample = train_examples[0]
        print("\n--- 첫 학습 예제 (prompt) ---")
        for message in sample["prompt"]:
            print(f"[{message['role']}]\n{message['content']}\n")
        print(f"--- completion ---\n{sample['completion'][0]['content']}")
        print("\ndry-run: 모델을 불러오지 않고 끝냅니다.")
        return

    import os

    hf_token = os.environ.get("HF_TOKEN") or None  # 빈 문자열이면 None으로 (공개 모델이면 토큰 없이도 된다)

    from datasets import Dataset
    from transformers import set_seed
    from trl import SFTTrainer

    set_seed(config["seed"])
    output_dir = resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"모델: {config['model']['name_or_path']} (QLoRA, 4bit NF4) · artifacts: {output_dir}")
    tokenizer, base_model = load_tokenizer_and_model(config, hf_token)
    peft_model, target_modules = build_peft_model(base_model, config)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"학습 파라미터: {trainable_params:,} / 전체 {total_params:,}")

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    trainer = SFTTrainer(
        model=peft_model,
        args=build_sft_config(config, output_dir / "checkpoints", has_validation=bool(val_examples)),
        train_dataset=Dataset.from_list(train_examples),
        eval_dataset=Dataset.from_list(val_examples) if val_examples else None,
        processing_class=tokenizer,
    )

    start = time.perf_counter()
    train_output = trainer.train()
    duration = time.perf_counter() - start
    peak = peak_vram_gb()

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    log_history = trainer.state.log_history
    final_eval = trainer.evaluate() if val_examples else {}
    metrics = {
        "model_id": config["model"]["name_or_path"],
        "method": "qlora",
        "seed": config["seed"],
        "lora": config["lora"] | {"target_modules": target_modules},
        "trainable_params": trainable_params,
        "total_params": total_params,
        "final_train_loss": last_value(log_history, "loss"),
        "mean_train_loss": train_output.training_loss,
        "eval_loss": final_eval.get("eval_loss"),
        "duration_sec": round(duration, 1),
        "global_steps": trainer.state.global_step,
        **peak,
        "adapter_path": str(adapter_dir),
        "counts": {"train": len(train_records), "validation": len(val_records)},
        "dataset_version": config["data"].get("dataset_version"),
    }
    output_dir.joinpath("metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_dir.joinpath("log_history.json").write_text(
        json.dumps(log_history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"완료: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

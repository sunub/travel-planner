"""EXAONE-3.5-7.8B-Instruct QLoRA 학습.

이 스크립트를 명시적으로 실행하기 전까지는 모델을 내려받거나 학습을 시작하지 않는다 (--dry-run은
모델을 아예 불러오지 않는다). 데이터셋이 아직 확정되지 않았으므로 config.yaml의
data.{train_path,validation_path,test_path}가 비어 있으면 무엇을 채워야 하는지 알려주고 멈춘다.

인증은 저장소 루트의 .env(HF_TOKEN, HF_HOME)를 python-dotenv로 읽는다. 토큰 값은 로그에 출력하지 않는다.

사용법:
  uv run python src/travel_planner/model-yay/train.py --config src/travel_planner/model-yay/config.yaml --dry-run
  uv run python src/travel_planner/model-yay/train.py --config src/travel_planner/model-yay/config.yaml
  uv run python src/travel_planner/model-yay/train.py --config src/travel_planner/model-yay/config.yaml \\
      --limit 32 --max-steps 5 --output-dir src/travel_planner/model-yay/artifacts/smoke
"""

import argparse
import json
import math
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

from data import encode_example, format_duration, format_hm, load_records, to_sft_example

MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODEL_DIR.parents[2]
IGNORE_KEYS_IN_LOG = frozenset({"HF_TOKEN"})  # .env 값은 절대 로그에 남기지 않는다


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def resolve_output_dir(config: dict, override: str | None = None) -> Path:
    path = Path(override) if override else Path(config["output"]["artifacts_root"])
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


def build_sft_config(config: dict, checkpoint_dir: Path, has_validation: bool, max_steps: int | None):
    from trl import SFTConfig

    t = config["training"]
    # --max-steps로 시험 실행할 때는 epoch 기준 저장/평가를 끈다. 32건짜리 스모크 데이터에서는
    # 1 epoch가 몇 step 안 되고, max_steps가 epoch 중간에 끊기면 epoch 기준 strategy와 앞뒤가
    # 맞지 않는다. 어댑터는 어차피 학습이 끝나면 trainer.save_model()로 따로 저장한다.
    smoke = max_steps is not None
    return SFTConfig(
        output_dir=str(checkpoint_dir),
        seed=config["seed"],
        data_seed=config["seed"],
        learning_rate=t["learning_rate"],
        num_train_epochs=t["num_train_epochs"],
        max_steps=max_steps if smoke else -1,
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", t["per_device_train_batch_size"]),
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        max_length=t["max_seq_length"],
        warmup_steps=t["warmup_steps"],  # transformers 5: warmup_ratio가 제거됨. 1 미만 실수는 비율로 취급된다
        weight_decay=t.get("weight_decay", 0.0),
        lr_scheduler_type=t["lr_scheduler_type"],
        max_grad_norm=t.get("max_grad_norm", 1.0),
        optim=t["optim"],
        gradient_checkpointing=t["gradient_checkpointing"],
        bf16=True,
        logging_steps=t.get("logging_steps", 10),
        eval_strategy=(t.get("eval_strategy", "epoch") if has_validation else "no") if not smoke else "no",
        save_strategy=t.get("save_strategy", "epoch") if not smoke else "no",
        save_total_limit=t.get("save_total_limit", 2),
        report_to="none",
        # completion_only_loss는 안 쓴다. 데이터셋에 이미 -100으로 마스킹한 labels가 있으면(encode_example,
        # data.py) TRL은 그걸 그대로 쓰고 자체 prompt/completion 마스킹은 건너뛴다(trl SFTTrainer의
        # `is_processed`/"labels" 컬럼 존재 여부 분기).
        packing=False,
    )


def build_step_timer():
    """스텝별 소요 시간을 재는 TrainerCallback. transformers를 실제 학습할 때만 불러오려고 함수 안에서 만든다.

    - on_step_end마다 logging_steps 주기로 "현재 스텝/전체 스텝 · 경과 시간 · 예상 남은 시간"을 로그에 남긴다.
    - average_step_time()은 첫 스텝(워밍업: CUDA 커널 컴파일 등으로 느리다)을 빼고 평균을 낸다.
    """
    from transformers import TrainerCallback

    class StepTimer(TrainerCallback):
        def __init__(self):
            self.step_durations: list[float] = []
            self._step_start: float | None = None
            self.train_start: float | None = None

        def on_train_begin(self, targs, state, control, **kwargs):
            self.train_start = time.perf_counter()

        def on_step_begin(self, targs, state, control, **kwargs):
            self._step_start = time.perf_counter()

        def on_step_end(self, targs, state, control, **kwargs):
            if self._step_start is not None:
                self.step_durations.append(time.perf_counter() - self._step_start)
            if state.global_step % max(targs.logging_steps, 1) == 0 or state.global_step >= state.max_steps:
                elapsed = time.perf_counter() - self.train_start
                avg = self.average_step_time()
                remaining_steps = max(state.max_steps - state.global_step, 0)
                eta = avg * remaining_steps if avg is not None else None
                print(
                    f"  [진행] step {state.global_step}/{state.max_steps} · 경과 {format_duration(elapsed)} "
                    f"· 예상 남은 시간 {format_duration(eta)}",
                    flush=True,
                )

        def average_step_time(self) -> float | None:
            usable = self.step_durations[1:] or self.step_durations  # 첫 스텝(워밍업) 제외
            return sum(usable) / len(usable) if usable else None

    return StepTimer()


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
    parser.add_argument("--limit", type=int, help="train/validation 앞에서 N건만 (스모크 실행)")
    parser.add_argument("--max-steps", type=int, help="N 스텝만 학습하고 종료 (VRAM·속도 시험용)")
    parser.add_argument("--output-dir", help="결과 저장 위치. 기본값은 config.yaml의 output.artifacts_root")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    import os

    print(f"HF_TOKEN: {'설정됨' if os.environ.get('HF_TOKEN') else '없음'} · HF_HOME: {os.environ.get('HF_HOME') or '기본값'}")

    config = load_config(args.config)
    data_paths = require_data_paths(config)

    train_records = load_records(data_paths["train_path"])
    full_train_count = len(train_records)  # --limit와 상관없이 "전체 학습 스텝 수" 계산에 쓴다
    val_records = load_records(data_paths["validation_path"])
    if args.limit:
        train_records = train_records[: args.limit]
        val_records = val_records[: args.limit]
    print(f"데이터: train {len(train_records)}건 / validation {len(val_records)}건 (전체 train {full_train_count}건)")

    if args.dry_run:
        sample = to_sft_example(train_records[0])
        print("\n--- 첫 학습 예제 (prompt) ---")
        for message in sample["prompt"]:
            print(f"[{message['role']}]\n{message['content']}\n")
        print(f"--- completion ---\n{sample['completion'][0]['content']}")
        print("\ndry-run: 모델을 불러오지 않고 끝냅니다.")
        return

    hf_token = os.environ.get("HF_TOKEN") or None  # 빈 문자열이면 None으로 (공개 모델이면 토큰 없이도 된다)

    from datasets import Dataset
    from transformers import set_seed
    from trl import SFTTrainer

    set_seed(config["seed"])
    output_dir = resolve_output_dir(config, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"모델: {config['model']['name_or_path']} (QLoRA, 4bit NF4) · artifacts: {output_dir}")
    tokenizer, base_model = load_tokenizer_and_model(config, hf_token)

    # 토큰화 + assistant 응답만 남기는 labels 마스킹을 직접 한다 (encode_example, data.py 참고).
    # TRL의 prompt/completion 자동 마스킹에 맡기지 않는다 — EXAONE 채팅 템플릿에서 "Mismatch between
    # tokenized prompt and the start of tokenized prompt+completion" 경고가 나는데, 그 상태로 TRL에
    # 맡기면 엉뚱한 위치에서 마스킹 경계가 잘릴 수 있다.
    max_seq_length = config["training"]["max_seq_length"]
    train_examples = [encode_example(tokenizer, r, max_seq_length) for r in train_records]
    val_examples = [encode_example(tokenizer, r, max_seq_length) for r in val_records]
    print(f"토큰화 완료: train {len(train_examples)}건 / validation {len(val_examples)}건")

    peft_model, target_modules = build_peft_model(base_model, config)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"학습 파라미터: {trainable_params:,} / 전체 {total_params:,}")

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    step_timer = build_step_timer()
    trainer = SFTTrainer(
        model=peft_model,
        args=build_sft_config(
            config, output_dir / "checkpoints", has_validation=bool(val_examples), max_steps=args.max_steps
        ),
        train_dataset=Dataset.from_list(train_examples),
        eval_dataset=Dataset.from_list(val_examples) if val_examples else None,
        processing_class=tokenizer,
        callbacks=[step_timer],
    )

    start = time.perf_counter()
    train_output = trainer.train()
    duration = time.perf_counter() - start
    peak = peak_vram_gb()

    time_estimate = None
    if args.max_steps:
        # --limit와 상관없이 config의 전체 train 데이터(full_train_count) 기준으로 실제 학습에서 도는
        # 스텝 수를 추정한다. 1 epoch당 스텝 수는 HF Trainer와 같은 방식(올림)으로 계산한다.
        t = config["training"]
        effective_batch = t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]
        steps_per_epoch = math.ceil(full_train_count / effective_batch)
        total_train_steps = steps_per_epoch * t["num_train_epochs"]
        avg_step_sec = step_timer.average_step_time()
        estimated_total_sec = avg_step_sec * total_train_steps if avg_step_sec is not None else None
        time_estimate = {
            "avg_step_sec": round(avg_step_sec, 3) if avg_step_sec is not None else None,
            "full_train_count": full_train_count,
            "steps_per_epoch": steps_per_epoch,
            "num_train_epochs": t["num_train_epochs"],
            "total_train_steps": total_train_steps,
            "estimated_total_sec": round(estimated_total_sec, 1) if estimated_total_sec is not None else None,
            "estimated_total_hm": format_hm(estimated_total_sec),
        }
        print("\n--- 소요 시간 예측 (시험 실행 기준) ---")
        print(f"  스텝 1회 평균 시간(첫 스텝 제외): {avg_step_sec:.2f}초" if avg_step_sec is not None else "  스텝 1회 평균 시간: 계산 불가 (스텝이 1개뿐)")
        print(f"  전체 학습 스텝 수 (train {full_train_count}건, {t['num_train_epochs']}epoch 기준): {total_train_steps}")
        print(f"  예상 전체 학습 시간: {time_estimate['estimated_total_hm']}")
        print(f"  최대 VRAM: allocated {peak['peak_vram_allocated_gb']}GB / reserved {peak['peak_vram_reserved_gb']}GB")

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
        "smoke_test": {"limit": args.limit, "max_steps": args.max_steps} if (args.limit or args.max_steps) else None,
        "time_estimate": time_estimate,
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

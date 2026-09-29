"""모델 · tokenizer · LoRA 설정.

Gemma4 E2B/E4B 체크포인트는 멀티모달(Gemma4ForConditionalGeneration)이다. AutoModelForCausalLM도 이 클래스를 돌려준다.
텍스트 디코더는 model.language_model.layers.N.{self_attn,mlp}.* 아래에 있고, q/k/v/o_proj · gate/up/down_proj는
nn.Linear다. vision/audio 타워에도 같은 이름(q_proj 등)이 있지만 Gemma4ClippableLinear라서 PEFT가 감쌀 수 없다.
그래서 target_modules는 이름 목록이 아니라 language_model 아래만 고르는 정규식을 쓴다
(transformers 5.17 modeling_gemma4.py와 PEFT의 gemma4 기본값 `.*language_model\\..*\\.(q_proj|v_proj)`에서 확인).
KV를 공유하는 뒤쪽 레이어에는 k_proj/v_proj가 없으므로 정규식이 그 레이어에서는 q/o_proj와 MLP만 고른다.
8GB GPU용 메모리 옵션(model.text_only, model.offload_per_layer_embeddings)은 loading_options()를 본다.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Final

from ..config import ConfigError, require_model_id

PRECISIONS: Final = ("bf16", "fp16", "fp32")


def resolve_precision(requested: str = "auto") -> str:
    """auto: CUDA가 bf16을 지원하면 bf16, 아니면 fp16, GPU가 없으면 fp32."""
    import torch

    if requested not in (*PRECISIONS, "auto"):
        raise ConfigError(f"precision은 auto/{'/'.join(PRECISIONS)} 중 하나여야 합니다: {requested!r}")
    cuda = torch.cuda.is_available()
    if requested == "auto":
        return ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if cuda else "fp32"
    if requested == "bf16" and cuda and not torch.cuda.is_bf16_supported():
        raise ConfigError("이 GPU는 bf16을 지원하지 않습니다. precision: fp16 또는 auto를 쓰세요.")
    return requested


def torch_dtype(precision: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def load_tokenizer(config: dict):
    from transformers import AutoTokenizer

    model = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(require_model_id(config), revision=model.get("revision"))
    if tokenizer.chat_template is None:
        raise ConfigError(f"{model['name_or_path']}의 tokenizer에 chat template이 없습니다 (instruction-tuned 모델인지 확인)")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def quantization_config(config: dict, precision: str):
    """QLoRA: bitsandbytes 4bit NF4. compute dtype auto면 학습 precision을 따른다."""
    import torch
    from transformers import BitsAndBytesConfig

    q = config["quantization"]
    compute = q.get("bnb_4bit_compute_dtype", "auto")
    compute_dtype = torch_dtype(precision if precision != "fp32" else "fp16") if compute == "auto" else torch_dtype(compute)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )


PLE_MODULE: Final = "model.language_model.embed_tokens_per_layer"  # Gemma4ForConditionalGeneration 안의 PLE 표


def loading_options(config: dict) -> dict:
    """메모리를 줄이는 불러오기 옵션. 계산 결과는 바꾸지 않으므로 Base · QLoRA가 같은 값을 쓰면 비교에 영향이 없다.

    - text_only: vision/audio 타워를 만들지 않는다 (텍스트 과업에 쓰지 않는 가중치, E2B 약 0.9GB).
    - offload_per_layer_embeddings: PLE 표를 CPU RAM에 둔다 (training/offload.py, E2B 4.38GB).
    """
    model_cfg = config["model"]
    return {key: bool(model_cfg.get(key, False)) for key in ("text_only", "offload_per_layer_embeddings")}


def load_model(config: dict, precision: str, *, quantized: bool, adapter_path: Path | None = None):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    model_cfg = config["model"]
    options = loading_options(config)
    if quantized and not torch.cuda.is_available():
        raise ConfigError("4bit 양자화(QLoRA · 4bit Base)는 CUDA GPU가 필요합니다.")
    device_map = model_cfg.get("device_map") or ({"": 0} if torch.cuda.is_available() else None)
    extra = {}
    if options["text_only"]:
        hf_config = AutoConfig.from_pretrained(require_model_id(config), revision=model_cfg.get("revision"))
        hf_config.vision_config = hf_config.audio_config = None  # Gemma4Model은 이 값이 None이면 타워를 만들지 않는다
        extra["config"] = hf_config
    if options["offload_per_layer_embeddings"] and not quantized and device_map == {"": 0}:
        # 16bit는 PLE까지 GPU에 올리면 불러오는 중에 넘칠 수 있어 처음부터 CPU에 둔다.
        # 4bit는 전체가 GPU에 들어가므로 그대로 불러온 뒤 옮긴다 (bitsandbytes는 CPU 배치가 섞인 device_map을 학습에서 막는다).
        device_map = {"": 0, PLE_MODULE: "cpu"}
    model = AutoModelForCausalLM.from_pretrained(
        require_model_id(config),
        revision=model_cfg.get("revision"),
        dtype=torch_dtype(precision),
        attn_implementation=model_cfg.get("attn_implementation"),
        quantization_config=quantization_config(config, precision) if quantized else None,
        device_map=device_map,
        **extra,
    )
    if options["offload_per_layer_embeddings"]:
        from .offload import offload_per_layer_embeddings

        freed = offload_per_layer_embeddings(model)
        print(f"  PLE 표를 CPU RAM으로 옮김 (GPU {freed / 2**30:.2f}GB 절약)")
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    return model


def lora_config(config: dict):
    from peft import LoraConfig

    lora = config["lora"]
    return LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias=lora.get("bias", "none"),
        task_type="CAUSAL_LM",
    )


def check_lora_targets(model) -> list[str]:
    """LoRA가 붙은 모듈 이름. 하나도 없거나 텍스트 타워 밖에 붙었으면 멈춘다."""
    wrapped = [name for name, module in model.named_modules() if hasattr(module, "lora_A") and hasattr(module, "base_layer")]
    if not wrapped:
        raise ConfigError("LoRA가 붙은 모듈이 없습니다. lora.target_modules를 확인하세요.")
    outside = [name for name in wrapped if "language_model" not in name]
    if outside:
        raise ConfigError(f"텍스트 타워 밖에 LoRA가 붙었습니다: {outside[:5]}")
    return wrapped


def count_parameters(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())  # 4bit 가중치는 저장 단위로 세므로 실제보다 작게 나온다
    return {"trainable_params": trainable, "total_params": total}


def token_lengths(tokenizer, examples: list[dict]) -> list[int]:
    """prompt + completion을 chat template으로 토큰화한 길이."""
    lengths = []
    for example in examples:
        out = tokenizer.apply_chat_template(
            example["prompt"] + example["completion"], tokenize=True, **example.get("chat_template_kwargs", {})
        )
        ids = out["input_ids"] if isinstance(out, Mapping) else out
        lengths.append(len(ids))
    return lengths

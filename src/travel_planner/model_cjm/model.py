"""토크나이저·모델·adapter 로드. notebooks/exaone_lora_qlora_tripfit.ipynb의 로드 방식과 동일하게 맞춘다."""

from __future__ import annotations

import importlib
import os
from typing import Any

from .config import ModelConfig


def _dtype():
    import torch

    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_tokenizer(config: ModelConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id, revision=config.revision, token=os.environ.get("HF_TOKEN"), trust_remote_code=True,
    )
    tokenizer.clean_up_tokenization_spaces = False  # evidence의 원문 공백 보존
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _patch_exaone(model: Any) -> None:
    # EXAONE 3.5 원격 코드는 이전 create_causal_mask 인자명을 쓰므로 Transformers 5.x API로 변환한다.
    from transformers.masking_utils import create_causal_mask as hf_create_causal_mask

    module = importlib.import_module(model.__class__.__module__)
    if getattr(module, "_tripfit_mask_compat", False) or not hasattr(module, "create_causal_mask"):
        return

    def create_causal_mask(*, config, input_embeds=None, inputs_embeds=None, attention_mask=None,
                           past_key_values=None, position_ids=None, **_ignored):
        return hf_create_causal_mask(
            config=config,
            inputs_embeds=inputs_embeds if inputs_embeds is not None else input_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

    module.create_causal_mask = create_causal_mask
    module._tripfit_mask_compat = True


def load_model(config: ModelConfig):
    """base는 bf16/fp16, qlora는 NF4 4bit, lora는 bf16/fp16으로 GPU 0에 올린다."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    dtype = _dtype()
    kwargs: dict[str, Any] = {
        "token": os.environ.get("HF_TOKEN"), "revision": config.revision, "trust_remote_code": True,
        "dtype": dtype, "device_map": {"": 0}, "attn_implementation": "sdpa",
    }
    if config.mode == "qlora" or config.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
    _patch_exaone(model)
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        model.transformer._input_embed_layer = "wte"  # EXAONE: PEFT가 찾는 입력 임베딩 이름 연결
    return model


def load_adapter(model: Any, adapter_path: str):
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_path)

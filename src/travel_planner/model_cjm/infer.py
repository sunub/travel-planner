"""Gold JSONL 추론. notebooks/exaone_lora_qlora_tripfit.ipynb의 프롬프트·로드 방식과 동일하게 맞춘다."""

from __future__ import annotations

import gc
import importlib
import json
import os
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """당신은 부산 장소 리뷰 정보 추출기다.
주어진 카테고리와 리뷰 원문에서만 정보를 추출하라.
반드시 JSON 객체 하나만 출력하라. 정확한 형식은
{\"traveler_context\": [...], \"aspects\": [{\"category\": \"...\", \"attribute\": \"...\", \"sentiment\": \"...\", \"evidence\": \"원문 그대로의 연속 구절\"}]} 이다.
리뷰에 없는 사실을 만들지 말고, evidence는 반드시 리뷰 원문을 그대로 인용하라."""


def review_user_message(record: dict[str, Any]) -> str:
    return f"[카테고리]\n{record['category']}\n\n[리뷰]\n{record['review']}"


def _training_revision(model_id: str, adapter_path: str | None) -> str | None:
    """학습 때 고정한 base revision을 run_summary.json에서 읽는다 (EXAONE 원격 코드 호환)."""
    if not adapter_path:
        return None
    summary = Path(adapter_path).parent / "run_summary.json"
    if not summary.exists():
        return None
    data = json.loads(summary.read_text(encoding="utf-8"))
    return data.get("revision") if data.get("model_id") == model_id else None


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


def _load(model_id: str, mode: str, adapter_path: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    token = os.environ.get("HF_TOKEN")
    revision = _training_revision(model_id, adapter_path)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token, trust_remote_code=True)
    tokenizer.clean_up_tokenization_spaces = False  # evidence의 원문 공백 보존
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "token": token, "revision": revision, "trust_remote_code": True,
        "dtype": dtype, "device_map": {"": 0}, "attn_implementation": "sdpa",
    }
    if mode == "qlora":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    _patch_exaone(model)
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        model.transformer._input_embed_layer = "wte"  # EXAONE: PEFT가 찾는 입력 임베딩 이름 연결

    if mode in {"lora", "qlora"}:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def _parse_json(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _prompt(tokenizer: Any, record: dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": review_user_message(record)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def predict(*, model_id: str, mode: str, input_file: Path, output_file: Path,
            adapter_path: str | None = None, max_new_tokens: int = 384, batch_size: int | None = None) -> None:
    import torch

    batch_size = batch_size or int(os.environ.get("TRIPFIT_INFER_BATCH_SIZE", "8"))
    records = [json.loads(line) for line in Path(input_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    model, tokenizer = _load(model_id, mode, adapter_path)
    tokenizer.padding_side = "left"  # 배치 생성은 왼쪽 패딩이어야 프롬프트 끝이 정렬된다
    device = next(model.parameters()).device
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 비슷한 길이끼리 묶어 패딩 낭비를 줄이고, 저장은 원래 순서대로 한다.
    prompts = [_prompt(tokenizer, record) for record in records]
    order = sorted(range(len(records)), key=lambda i: len(prompts[i]))
    outputs: dict[int, str] = {}

    try:
        done = 0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            inputs = tokenizer([prompts[i] for i in batch], return_tensors="pt", padding=True,
                               add_special_tokens=False).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                )
            prompt_length = inputs["input_ids"].shape[1]
            for row, i in enumerate(batch):
                outputs[i] = tokenizer.decode(generated[row, prompt_length:], skip_special_tokens=True).strip()
            done += len(batch)
            print(f"[{done}/{len(records)}] batch={len(batch)}", flush=True)

        with output_file.open("w", encoding="utf-8") as sink:
            for i, record in enumerate(records):
                raw = outputs[i]
                label = _parse_json(raw)
                sink.write(json.dumps({
                    "review_id": record.get("review_id"),
                    "place_id": record.get("place_id"),
                    "category": record.get("category"),
                    "review": record.get("review", ""),
                    "raw_output": raw,
                    "label": label,
                    "json_valid": label is not None,
                }, ensure_ascii=False) + "\n")
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

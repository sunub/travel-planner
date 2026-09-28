"""prompt 메시지 → 모델 출력 문자열. 채점은 하지 않는다 (evaluation 패키지가 한다)."""

import time


def generate_outputs(
    model,
    tokenizer,
    message_lists: list[list[dict]],
    *,
    max_new_tokens: int,
    batch_size: int,
    chat_template_kwargs: dict | None = None,
) -> tuple[list[str], dict]:
    """greedy decoding (do_sample=False)이라 같은 입력에는 매번 같은 답이 나온다. (출력, 속도 통계)."""
    import torch

    model.eval()
    if getattr(model, "is_gradient_checkpointing", False):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    tokenizer.padding_side = "left"  # 생성은 오른쪽 끝에서 이어지므로 패딩을 왼쪽에 둔다

    prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **(chat_template_kwargs or {}))
        for m in message_lists
    ]
    outputs, generated_tokens = [], 0
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(prompts), batch_size):
            batch = tokenizer(prompts[i : i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
            batch = batch.to(model.device)
            generated = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = generated[:, batch["input_ids"].shape[1] :]
            generated_tokens += int((new_tokens != tokenizer.pad_token_id).sum())
            outputs += tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            print(f"  생성 {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    duration = time.perf_counter() - start
    return outputs, {
        "duration_sec": round(duration, 2),
        "sec_per_review": round(duration / max(len(prompts), 1), 3),
        "generated_tokens": generated_tokens,
        "tokens_per_sec": round(generated_tokens / duration, 2) if duration else None,
    }

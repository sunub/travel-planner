"""공정한 베이스라인: 베이스 모델 프롬프트에 카테고리별 허용 aspect·attribute 목록을 넣어 추론한다.

파인튜닝 프롬프트에는 라벨 이름 목록이 없어서, 베이스 모델은 이름을 몰라 점수가 0이 된다.
이 스크립트는 이름을 알려 줬을 때도 파인튜닝 모델과 차이가 나는지 보기 위한 조건이다.

  uv run python -m evaluation.schema_prompt_baseline \
    --out evaluation/runs/<실행>/extra/exaone_2_4b_base_schema.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.common.schema import ASPECTS, ATTRIBUTES, TRAVELER_CONTEXTS
from travel_planner.model_cjm.infer import SYSTEM_PROMPT, _load, _parse_json, review_user_message

from .pipeline import normalize_prediction_record, read_jsonl, write_jsonl


def system_prompt(category: str) -> str:
    allowed = {aspect: list(ATTRIBUTES[category][aspect]) for aspect in sorted(ASPECTS[category])}
    return (
        SYSTEM_PROMPT
        + "\n\ntraveler_context 허용 값: " + ", ".join(sorted(TRAVELER_CONTEXTS))
        + "\nsentiment 허용 값: positive, negative, neutral"
        + "\n이 카테고리에서 쓸 수 있는 category(aspect)와 attribute는 아래뿐이다. 다른 이름은 쓰지 마라.\n"
        + json.dumps(allowed, ensure_ascii=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="라벨 목록을 프롬프트에 넣은 베이스 모델 추론")
    parser.add_argument("--gold", type=Path, default=Path("datasets/test.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-id", default="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct")
    parser.add_argument("--name", default="exaone_2_4b_base_schema")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    import torch

    records = read_jsonl(args.gold)
    model, tokenizer = _load(args.model_id, "base", None, load_in_4bit=True)  # QLoRA 평가와 같은 NF4 4bit
    tokenizer.padding_side = "left"
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt(r["category"])}, {"role": "user", "content": review_user_message(r)}],
            tokenize=False, add_generation_prompt=True,
        )
        for r in records
    ]
    order = sorted(range(len(records)), key=lambda i: len(prompts[i]))
    outputs: dict[int, str] = {}
    for start in range(0, len(order), args.batch_size):
        batch = order[start:start + args.batch_size]
        inputs = tokenizer([prompts[i] for i in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(0)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                       eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        for row, i in enumerate(batch):
            outputs[i] = tokenizer.decode(generated[row, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"[{start + len(batch)}/{len(records)}]", flush=True)

    rows = []
    for i, record in enumerate(records):
        label = _parse_json(outputs[i])
        rows.append(normalize_prediction_record({
            "review_id": record["review_id"], "place_id": record.get("place_id"), "category": record["category"],
            "review": record["review"], "raw_output": outputs[i], "label": label, "json_valid": label is not None,
        }, args.name))
    write_jsonl(args.out, rows)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()

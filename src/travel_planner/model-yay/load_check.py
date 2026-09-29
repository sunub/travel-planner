"""EXAONE 로딩 확인 스크립트 (학습 없음).

train.py와 같은 load_tokenizer_and_model()로 토크나이저·모델만 불러오고, 로딩 전후 GPU 메모리
(torch.cuda.memory_allocated / max_memory_reserved)를 GB로 출력한 뒤 짧은 문장 하나를 generate해
본다. 데이터셋(config.yaml의 data.*)은 전혀 읽지 않으므로 경로가 비어 있어도 실행할 수 있다.

--method로 train.py와 같은 qlora(4bit NF4)/lora(bf16, 양자화 없음)를 골라, 두 방식의 로딩 메모리와
생성 결과를 비교해 볼 수 있다.

사용법:
  uv run python src/travel_planner/model-yay/load_check.py --config src/travel_planner/model-yay/config.yaml --method qlora
  uv run python src/travel_planner/model-yay/load_check.py --config src/travel_planner/model-yay/config.yaml --method lora
"""

import argparse

from dotenv import load_dotenv

from train import REPO_ROOT, load_config, load_tokenizer_and_model


def gb(num_bytes: int) -> float:
    return round(num_bytes / 2**30, 3)


def print_memory(label: str) -> None:
    import torch

    if not torch.cuda.is_available():
        print(f"[{label}] CUDA 사용 불가")
        return
    print(
        f"[{label}] allocated {gb(torch.cuda.memory_allocated())}GB "
        f"· max_reserved {gb(torch.cuda.max_memory_reserved())}GB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--method", required=True, choices=["qlora", "lora"], help="qlora=4bit NF4 / lora=bf16, 양자화 없음"
    )
    args = parser.parse_args()

    import os

    load_dotenv(REPO_ROOT / ".env")
    print(f"HF_TOKEN: {'설정됨' if os.environ.get('HF_TOKEN') else '없음'} · HF_HOME: {os.environ.get('HF_HOME') or '기본값'}")
    hf_token = os.environ.get("HF_TOKEN") or None

    config = load_config(args.config)
    method_desc = "4bit NF4" if args.method == "qlora" else "bf16, 양자화 없음"
    print(f"모델: {config['model']['name_or_path']} ({args.method.upper()}, {method_desc})")

    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print_memory("로딩 전")

    tokenizer, model = load_tokenizer_and_model(config, hf_token, args.method)
    print_memory("로딩 후")

    # embedding·lm_head가 4bit 대상에서 빠진 뒤(qlora) 어떤 dtype으로 올라갔는지 직접 확인한다.
    # lora는 애초에 양자화가 없으므로 전부 bf16이어야 한다.
    print(
        f"embedding dtype: {model.get_input_embeddings().weight.dtype} "
        f"· lm_head dtype: {model.get_output_embeddings().weight.dtype}"
    )

    messages = [
        {"role": "system", "content": "당신은 여행 도우미입니다."},
        {"role": "user", "content": "부산 여행을 한 문장으로 추천해줘."},
    ]
    # enable_thinking=False: data.py의 prompt_text()와 같은 이유로 사고 모드를 명시적으로 끈다.
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    print_memory("generate 후")

    print(f"\n--- prompt 끝부분 (사고 모드 꺼짐 확인) ---\n{prompt[-60:]!r}")
    print(f"\n--- generate 결과 ---\n{generated}")


if __name__ == "__main__":
    main()

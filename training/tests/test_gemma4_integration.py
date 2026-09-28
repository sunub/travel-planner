"""Gemma4 구조 · chat template 확인. 가중치를 받지 않는다.

- LoRA target: config만으로 meta device(메모리 0)에 Gemma4 멀티모달 모델을 만들고, LoRA가 텍스트 타워에만 붙는지 본다.
- chat template: TRIPFIT_TOKENIZER에 model_id나 로컬 경로를 주면 실제 tokenizer로 prompt/completion 경계를 확인한다.
  (tokenizer 파일을 받아야 하므로 기본은 건너뛴다.)
"""

import os
from collections import Counter

import pytest

from travel_planner.finetune.config import load_config

from conftest import CONFIG_DIR, make_record

torch = pytest.importorskip("torch")


def test_lora_attaches_only_to_text_tower():
    from peft import get_peft_model
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration
    from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig, Gemma4TextConfig, Gemma4VisionConfig

    from travel_planner.finetune.training.model import check_lora_targets, lora_config

    config = Gemma4Config(
        text_config=Gemma4TextConfig(num_hidden_layers=6, num_kv_shared_layers=2).to_dict(),
        vision_config=Gemma4VisionConfig().to_dict(),
        audio_config=Gemma4AudioConfig().to_dict(),
    )
    with torch.device("meta"):
        model = Gemma4ForConditionalGeneration(config)
    towers = Counter(name.split(".")[1] for name, _ in model.named_modules() if name.endswith("q_proj"))
    assert towers["vision_tower"] and towers["audio_tower"]  # 같은 이름의 모듈이 다른 타워에도 있다

    wrapped = check_lora_targets(get_peft_model(model, lora_config(load_config(CONFIG_DIR / "lora_gold_v1.yaml"))))
    kinds = Counter(name.rsplit(".", 1)[-1] for name in wrapped)
    assert kinds["q_proj"] == kinds["o_proj"] == kinds["down_proj"] == 6
    assert kinds["k_proj"] == kinds["v_proj"] == 4  # KV 공유 레이어 2개에는 k/v_proj가 없다


@pytest.mark.skipif(not os.environ.get("TRIPFIT_TOKENIZER"), reason="TRIPFIT_TOKENIZER가 없어 건너뜀")
def test_chat_template_prompt_is_prefix_of_full_conversation():
    from transformers import AutoTokenizer

    from travel_planner.finetune.data.sft import PromptTemplates, to_sft_example

    config = load_config(CONFIG_DIR / "common.yaml")
    tokenizer = AutoTokenizer.from_pretrained(os.environ["TRIPFIT_TOKENIZER"])
    example = to_sft_example(make_record(1, "hotel", "p"), PromptTemplates.from_config(config), config["prompt"]["chat_template_kwargs"])
    kwargs = example["chat_template_kwargs"]
    prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(example["prompt"] + example["completion"], tokenize=False, **kwargs)
    print(f"\n--- prompt ---\n{prompt}\n--- completion 부분 ---\n{full[len(prompt):]}")
    assert full.startswith(prompt), "prompt가 전체 대화의 앞부분이 아니면 completion mask 경계가 어긋난다"
    assert example["completion"][0]["content"] in full[len(prompt) :]

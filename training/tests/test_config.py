import re

import pytest

from travel_planner.finetune.config import ConfigError, load_config, require_model_id

from conftest import CONFIG_DIR

EXPERIMENT_CONFIGS = ["base_eval.yaml", "lora_gold_v1.yaml", "qlora_gold_v1.yaml"]


@pytest.mark.parametrize("name", EXPERIMENT_CONFIGS)
def test_experiment_configs_load(name):
    config = load_config(CONFIG_DIR / name)
    assert config["seed"] == 42
    assert config["data"]["split_manifest"].endswith("gold_split_v1.json")
    assert config["prompt"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_lora_and_qlora_share_hyperparameters_and_differ_in_quantization():
    lora = load_config(CONFIG_DIR / "lora_gold_v1.yaml")
    qlora = load_config(CONFIG_DIR / "qlora_gold_v1.yaml")
    assert (lora["method"], qlora["method"]) == ("lora", "qlora")
    assert lora["lora"] == qlora["lora"]
    assert lora["lora"]["r"] == 16 and lora["lora"]["alpha"] == 32
    assert lora["training"]["learning_rate"] == 2e-4 and lora["training"]["num_train_epochs"] == 3
    assert "quantization" not in lora
    assert qlora["quantization"]["bnb_4bit_quant_type"] == "nf4"


def test_target_regex_selects_text_tower_only():
    pattern = re.compile(load_config(CONFIG_DIR / "lora_gold_v1.yaml")["lora"]["target_modules"])
    assert pattern.fullmatch("model.language_model.layers.3.self_attn.q_proj")
    assert pattern.fullmatch("model.language_model.layers.3.mlp.down_proj")
    assert not pattern.fullmatch("model.vision_tower.encoder.layers.0.self_attn.q_proj")
    assert not pattern.fullmatch("model.audio_tower.layers.0.self_attn.q_proj")
    assert not pattern.fullmatch("model.language_model.layers.3.self_attn.q_proj.lora_A")


def test_override_and_model_id():
    config = load_config(CONFIG_DIR / "qlora_gold_v1.yaml", ["training.num_train_epochs=1", "model.name_or_path=some/model"])
    assert config["training"]["num_train_epochs"] == 1
    assert require_model_id(config) == "some/model"


@pytest.mark.parametrize("name", EXPERIMENT_CONFIGS)
def test_all_experiments_use_same_base_model(name):
    assert require_model_id(load_config(CONFIG_DIR / name)) == "google/gemma-4-E4B-it"


def test_empty_model_id_is_rejected():
    with pytest.raises(ConfigError, match="model.name_or_path"):
        require_model_id(load_config(CONFIG_DIR / "base_eval.yaml", ["model.name_or_path=null"]))


def test_invalid_method_rejected():
    with pytest.raises(ConfigError):
        load_config(CONFIG_DIR / "base_eval.yaml", ["method=full"])

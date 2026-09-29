"""PLE CPU offload · text_only 불러오기. 가중치를 받지 않고 작은 무작위 Gemma4로 확인한다."""

import pytest

torch = pytest.importorskip("torch")

from travel_planner.finetune.config import load_config
from travel_planner.finetune.training.model import loading_options
from travel_planner.finetune.training.offload import CpuOffloadedEmbedding, offload_per_layer_embeddings

from conftest import CONFIG_DIR


def tiny_gemma4(text_only: bool = True):
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration
    from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig, Gemma4TextConfig, Gemma4VisionConfig

    text = Gemma4TextConfig(
        vocab_size=512, vocab_size_per_layer_input=512, hidden_size=64, hidden_size_per_layer_input=16, intermediate_size=128,
        num_hidden_layers=4, num_kv_shared_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],  # 공유 레이어가 빌려 올 full 레이어가 앞에 있어야 한다
    )  # fmt: skip
    config = Gemma4Config(text_config=text.to_dict(), vision_config=Gemma4VisionConfig().to_dict(), audio_config=Gemma4AudioConfig().to_dict())
    if text_only:
        config.vision_config = config.audio_config = None  # load_model(text_only)과 같은 방법
    torch.manual_seed(0)
    return Gemma4ForConditionalGeneration(config).eval()


def test_text_only_config_builds_no_towers():
    names = {name.split(".")[1] for name, _ in tiny_gemma4(text_only=True).named_modules() if name.count(".") >= 1}
    assert "language_model" in names and not names & {"vision_tower", "audio_tower", "embed_vision", "embed_audio"}
    assert {"vision_tower", "audio_tower"} <= {name.split(".")[1] for name, _ in tiny_gemma4(text_only=False).named_modules() if name.count(".") >= 1}


def devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", devices())
def test_offloaded_ple_gives_identical_logits_and_stays_on_cpu(device):
    model = tiny_gemma4().to(device=device, dtype=torch.bfloat16)
    ids = torch.randint(0, 512, (2, 7), device=device)
    with torch.no_grad():
        expected = model(input_ids=ids).logits
    freed = offload_per_layer_embeddings(model)
    ple = model.model.language_model.embed_tokens_per_layer
    assert isinstance(ple, CpuOffloadedEmbedding) and ple.table.device.type == "cpu"
    assert freed == (512 * 4 * 16 * 2 if device == "cuda" else 0)
    model.to(device)  # Trainer · PEFT가 모델을 옮겨도 표는 CPU에 남는다
    assert ple.table.device.type == "cpu" and "table" not in dict(model.named_parameters())
    with torch.no_grad():
        got = model(input_ids=ids).logits
    assert got.device.type == device and torch.equal(got, expected)


@pytest.mark.parametrize("device", devices())
def test_lora_trains_through_offloaded_ple(device):
    from peft import get_peft_model

    from travel_planner.finetune.training.model import lora_config

    model = tiny_gemma4().to(device=device, dtype=torch.float32)
    offload_per_layer_embeddings(model)
    model = get_peft_model(model, lora_config(load_config(CONFIG_DIR / "qlora_busan_v2.yaml")))
    model.train()
    ids = torch.randint(0, 512, (1, 9), device=device)
    model(input_ids=ids, labels=ids).loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if "lora_B" in n]
    assert grads and all(g is not None for g in grads)
    assert not any("embed_tokens_per_layer" in n for n, p in model.named_parameters() if p.requires_grad)


def test_busan_v2_configs_turn_on_memory_options():
    for name in ("base_eval_busan_v2.yaml", "base4bit_eval_busan_v2.yaml", "qlora_busan_v2.yaml"):
        assert loading_options(load_config(CONFIG_DIR / name)) == {"text_only": True, "offload_per_layer_embeddings": True}
    assert loading_options(load_config(CONFIG_DIR / "base_eval.yaml")) == {"text_only": False, "offload_per_layer_embeddings": False}


@pytest.mark.parametrize("quantized", [True, False])
def test_load_model_wires_memory_options(monkeypatch, quantized):
    import transformers

    from travel_planner.finetune.training import model as model_module

    calls = {}

    class FakeConfig:
        vision_config, audio_config = "vision", "audio"

    def fake_from_pretrained(model_id, **kwargs):
        calls.update(kwargs)
        return tiny_gemma4(text_only=kwargs["config"].vision_config is None)

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: FakeConfig())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(model_module, "quantization_config", lambda config, precision: "4bit")

    config = load_config(CONFIG_DIR / ("qlora_busan_v2.yaml" if quantized else "base_eval_busan_v2.yaml"))
    loaded = model_module.load_model(config, "fp32", quantized=quantized)
    assert calls["config"].vision_config is None and calls["config"].audio_config is None
    # 4bit는 GPU에 다 올린 뒤 옮기고, 16bit는 PLE를 처음부터 CPU에 둔다
    assert calls["device_map"] == ({"": 0} if quantized else {"": 0, model_module.PLE_MODULE: "cpu"})
    assert isinstance(loaded.model.language_model.embed_tokens_per_layer, CpuOffloadedEmbedding)

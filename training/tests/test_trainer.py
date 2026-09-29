"""학습 설정 → SFTConfig, 선택한 체크포인트 기록. 모델은 불러오지 않는다."""

from types import SimpleNamespace

from travel_planner.finetune.config import load_config
from travel_planner.finetune.training.trainer import selected_checkpoint, sft_config
from travel_planner.finetune.utils.tracking import create_run

from conftest import CONFIG_DIR


def test_busan_v2_selects_best_checkpoint_by_eval_loss(tmp_path):
    config = load_config(CONFIG_DIR / "qlora_busan_v2.yaml", [f"output.artifacts_root={tmp_path / 'a'}", f"output.experiments_root={tmp_path / 'e'}"])
    args = sft_config(config, create_run(config), "fp32", has_validation=True)  # SFTConfig가 eval/save 주기 일치를 검사한다
    assert args.load_best_model_at_end and args.metric_for_best_model == "eval_loss" and args.greater_is_better is False
    assert args.eval_strategy == args.save_strategy == "epoch"


def test_selected_checkpoint_records_best_or_last_step():
    best = SimpleNamespace(best_model_checkpoint="C:/x/checkpoints/checkpoint-116", best_metric=0.0353, global_step=174)
    on = SimpleNamespace(load_best_model_at_end=True, metric_for_best_model="eval_loss")
    assert selected_checkpoint(best, on) == {
        "selected_by": "eval_loss", "selected_checkpoint": "checkpoint-116", "selected_step": 116, "selected_metric": 0.0353,
    }  # fmt: skip
    off = SimpleNamespace(load_best_model_at_end=False, metric_for_best_model=None)
    assert selected_checkpoint(best, off)["selected_by"] == "last_step"
    assert selected_checkpoint(best, off)["selected_step"] == 174


def test_busan_v2_keeps_effective_batch_16():
    t = load_config(CONFIG_DIR / "qlora_busan_v2.yaml")["training"]
    base = load_config(CONFIG_DIR / "qlora_gold_v1.yaml")["training"]
    assert t["per_device_train_batch_size"] == 1
    assert t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] == base["per_device_train_batch_size"] * base["gradient_accumulation_steps"] == 16
    assert {k: t[k] for k in ("learning_rate", "num_train_epochs", "lr_scheduler_type", "warmup_steps")} == {
        k: base[k] for k in ("learning_rate", "num_train_epochs", "lr_scheduler_type", "warmup_steps")
    }


def test_restore_adapter_dtypes_undoes_fp32_upcast():
    import pytest

    torch = pytest.importorskip("torch")
    from travel_planner.finetune.training.trainer import restore_adapter_dtypes

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2)).to(torch.bfloat16)
    model[1].requires_grad_(False)
    dtypes = {n: p.dtype for n, p in model.named_parameters() if p.requires_grad}
    before = model[0].weight.detach().clone()
    model[0].to(torch.float32)  # load_adapter가 하는 fp32 upcast
    assert restore_adapter_dtypes(model, dtypes) == 2
    assert model[0].weight.dtype == torch.bfloat16 and torch.equal(model[0].weight, before)
    assert restore_adapter_dtypes(model, dtypes) == 0

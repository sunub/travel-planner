"""실험 설정(YAML) 읽기.

- `defaults: <다른 yaml>`로 공통 설정을 물려받는다 (여러 단계 가능). 자식 값이 부모 값을 덮어쓴다.
- CLI의 `--set a.b=값`으로 한 값을 바꿀 수 있다. 값은 YAML로 해석한다 (`--set training.num_train_epochs=1`).
- 설정 안의 상대 경로는 저장소 루트 기준이다. 어느 폴더에서 실행해도 같은 파일을 가리킨다.
"""

import copy
from pathlib import Path
from typing import Any, Final

import yaml

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
METHODS: Final = ("base", "lora", "qlora")


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_with_defaults(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ConfigError(f"defaults가 순환합니다: {' -> '.join(map(str, (*seen, path)))}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("defaults", None)
    if parent is None:
        return data
    return deep_merge(_read_with_defaults(path.parent / parent, (*seen, path)), data)


def apply_override(config: dict, assignment: str) -> None:
    """'a.b.c=값'을 config에 반영한다."""
    key, sep, raw = assignment.partition("=")
    if not sep or not key:
        raise ConfigError(f"--set은 key=value 형식이어야 합니다: {assignment!r}")
    *parents, leaf = key.split(".")
    node = config
    for part in parents:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"{key}: {part}는 설정 묶음이 아닙니다")
    node[leaf] = yaml.safe_load(raw)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    config = _read_with_defaults(Path(path))
    for assignment in overrides or []:
        apply_override(config, assignment)
    config["config_path"] = str(Path(path).resolve())
    validate(config)
    return config


def validate(config: dict) -> None:
    """공통 설정(common.yaml)만으로도 통과한다. experiment_name · method가 있으면 실험 설정으로 검사한다."""
    for key in ("seed", "data", "prompt", "output"):
        if key not in config:
            raise ConfigError(f"설정에 {key}가 없습니다 ({config.get('config_path')})")
    if "method" in config or "experiment_name" in config:
        validate_experiment(config)


def validate_experiment(config: dict) -> None:
    for key in ("experiment_name", "method"):
        if key not in config:
            raise ConfigError(f"실험 설정에 {key}가 없습니다 ({config.get('config_path')})")
    if config["method"] not in METHODS:
        raise ConfigError(f"method는 {METHODS} 중 하나여야 합니다: {config['method']!r}")
    if config["method"] in ("lora", "qlora"):
        for key in ("lora", "training"):
            if key not in config:
                raise ConfigError(f"{config['method']} 설정에는 {key} 묶음이 필요합니다")
    if config["method"] == "qlora" and not config.get("quantization", {}).get("load_in_4bit"):
        raise ConfigError("qlora 설정에는 quantization.load_in_4bit: true가 필요합니다")


def uses_4bit(config: dict) -> bool:
    """base 모델을 4bit로 불러오는가. QLoRA는 항상, Base는 quantization.load_in_4bit: true일 때 (4bit Base 기준선)."""
    return config["method"] == "qlora" or (config["method"] == "base" and bool((config.get("quantization") or {}).get("load_in_4bit")))


def require_model_id(config: dict) -> str:
    """모델을 실제로 불러오기 직전에만 부른다. model_id가 비어 있으면 무엇을 채울지 알려준다."""
    model_id = config.get("model", {}).get("name_or_path")
    if not model_id:
        raise ConfigError(
            "model.name_or_path가 비어 있습니다. training/configs/common.yaml에 Hugging Face model_id를 적거나 "
            "--model <id> 로 넘기세요."
        )
    return model_id


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def get(config: dict, dotted: str, default: Any = None) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node

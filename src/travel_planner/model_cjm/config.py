"""모델 로드 설정."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    mode: str  # base | lora | qlora
    model_id: str
    revision: str | None = None  # 학습 때 고정한 base revision (None이면 main)
    load_in_4bit: bool = False  # qlora가 아니어도 NF4 4bit로 로드 (예: base를 qlora와 같은 정밀도로 비교)

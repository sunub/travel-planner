"""Gemma4 per-layer embedding(PLE) 표를 GPU 대신 CPU RAM에 둔다.

PLE(`model.language_model.embed_tokens_per_layer`)는 단어마다 레이어별 보조 벡터를 찾아 읽는 표다. 계산은 없고 크기만 크다
(E2B 2.35B · bf16 4.38GB, E4B 2.82B · 5.25GB). bitsandbytes 4bit는 nn.Linear만 줄이므로 QLoRA에서도 이 표는 bf16 그대로
GPU에 남고, 8GB GPU에서는 이것 때문에 학습이 OOM으로 멈췄다.

CpuOffloadedEmbedding은 표를 CPU에 두고, 입력 토큰의 행만 찾아 입력과 같은 장치로 보낸 뒤 원래와 같은 scale을 곱한다.
찾기는 값을 그대로 복사할 뿐이고 곱셈은 원래처럼 GPU에서 하므로, 결과는 표가 GPU에 있을 때와 같다.
표는 frozen이라 학습 대상이 아니다 (LoRA는 decoder Linear에만 붙는다).
"""

import torch
from torch import nn

PLE_NAME = "embed_tokens_per_layer"


class CpuOffloadedEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        # 표는 파라미터 · 버퍼로 등록하지 않는다. model.to("cuda")나 Trainer가 GPU로 되돌리지 않고, state_dict에도 들어가지 않는다.
        self.__dict__["table"] = embedding.weight.detach().to("cpu")
        self.padding_idx = embedding.padding_idx
        scale = getattr(embedding, "embed_scale", None)  # Gemma4TextScaledWordEmbedding의 scale (없으면 1)
        self.register_buffer("embed_scale", torch.ones(()) if scale is None else scale.detach().clone(), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows = nn.functional.embedding(input_ids.to(self.table.device), self.table, self.padding_idx).to(input_ids.device)
        return rows * self.embed_scale.to(device=rows.device, dtype=self.table.dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.table.shape)}, dtype={self.table.dtype}, device={self.table.device}"


def offload_per_layer_embeddings(model: nn.Module) -> int:
    """모델 안의 모든 PLE 표를 CPU 버전으로 바꾼다. GPU에서 비운 바이트 수를 돌려준다."""
    targets = [name for name, module in model.named_modules() if name.rsplit(".", 1)[-1] == PLE_NAME and isinstance(module, nn.Embedding)]
    if not targets:
        raise ValueError(f"{PLE_NAME}가 없는 모델입니다. model.offload_per_layer_embeddings는 Gemma4 E 계열에만 씁니다.")
    freed = 0
    for name in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, attr)
        if old.weight.is_cuda:
            freed += old.weight.numel() * old.weight.element_size()
        setattr(parent, attr, CpuOffloadedEmbedding(old))
        if isinstance(getattr(model, "hf_device_map", None), dict):  # accelerate가 이 모듈에 붙인 배치 정보는 이제 맞지 않는다
            model.hf_device_map.pop(name, None)
        del old
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return freed

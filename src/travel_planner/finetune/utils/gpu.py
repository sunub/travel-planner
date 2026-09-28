"""GPU 정보와 최대 VRAM 측정. torch는 함수 안에서만 불러와서 GPU 없는 테스트에서도 import할 수 있다."""


def device_info() -> dict:
    import torch

    info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda": torch.version.cuda}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info |= {
            "device": props.name,
            "total_vram_gb": round(props.total_memory / 2**30, 2),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    return info


def reset_peak_memory() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb() -> dict:
    """allocated: 텐서가 실제로 쓴 최대량. reserved: PyTorch가 잡아 둔 최대량 (nvidia-smi 값에 더 가깝다)."""
    import torch

    if not torch.cuda.is_available():
        return {"peak_vram_allocated_gb": None, "peak_vram_reserved_gb": None}
    return {
        "peak_vram_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }

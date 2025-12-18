import torch


def get_optimal_device() -> torch.device:
    """利用可能な最適なデバイスを返す"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_for_model(model_type: str) -> torch.device:
    """モデル種別に応じたデバイスを返す

    Args:
        model_type: "sam3", "blip2", "sam_audio", "default" のいずれか
    """
    device = get_optimal_device()

    # SAM Audioは現時点でMPS未検証のためCPUフォールバック
    if model_type == "sam_audio" and device.type == "mps":
        return torch.device("cpu")

    return device


def get_dtype_for_device(device: torch.device) -> torch.dtype:
    """デバイスに応じた適切なdtypeを返す"""
    if device.type == "cpu":
        return torch.float32
    return torch.float16


def clear_memory(device: torch.device) -> None:
    """デバイスのメモリをクリア"""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

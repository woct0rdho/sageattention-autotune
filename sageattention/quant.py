import torch

from . import _fused

_fused = torch.ops.sageattention_fused  # noqa: F811


def _layout_id(tensor_layout: str) -> int:
    if tensor_layout == "NHD":
        return 0
    if tensor_layout == "HND":
        return 1
    raise ValueError(f"Unknown tensor layout: {tensor_layout}")


def sub_mean(v: torch.Tensor, tensor_layout: str = "HND"):
    layout = _layout_id(tensor_layout)
    seq_dim = 1 if layout == 0 else 2

    vm = v.mean(dim=seq_dim)
    v_smoothed = torch.empty(v.shape, dtype=torch.float16, device=v.device)
    _fused.sub_mean_cuda(v, vm, v_smoothed, layout)

    return v_smoothed, vm

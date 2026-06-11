from typing import Optional

import torch

from . import _fused

_fused = torch.ops.sageattention_fused  # noqa: F811


def _layout_id(tensor_layout: str) -> int:
    if tensor_layout == "NHD":
        return 0
    if tensor_layout == "HND":
        return 1
    raise ValueError(f"Unknown tensor layout: {tensor_layout}")


def per_warp_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    km: Optional[torch.Tensor] = None,
    BLKQ: int = 128,
    WARPQ: int = 32,
    BLKK: int = 64,
    tensor_layout: str = "HND",
):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if tensor_layout == "HND":
        batch_size, num_qo_heads, qo_len, _ = q.shape
        _, num_kv_heads, kv_len, _ = k.shape
    elif tensor_layout == "NHD":
        batch_size, qo_len, num_qo_heads, _ = q.shape
        _, kv_len, num_kv_heads, _ = k.shape
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    q_scale = torch.empty(
        (batch_size, num_qo_heads, ((qo_len + BLKQ - 1) // BLKQ) * (BLKQ // WARPQ)),
        device=q.device,
        dtype=torch.float32,
    )
    k_scale = torch.empty(
        (batch_size, num_kv_heads, (kv_len + BLKK - 1) // BLKK),
        device=q.device,
        dtype=torch.float32,
    )

    layout = _layout_id(tensor_layout)
    _fused.quant_per_warp_int8_cuda(q, q_int8, q_scale, BLKQ, WARPQ, layout)

    if km is None:
        _fused.quant_per_block_int8_cuda(k, k_int8, k_scale, BLKK, layout)
    else:
        km = km.squeeze(1) if layout == 0 else km.squeeze(2)
        _fused.quant_per_block_int8_fuse_sub_mean_cuda(k, km, k_int8, k_scale, BLKK, layout)

    return q_int8, q_scale, k_int8, k_scale


def sub_mean(v: torch.Tensor, tensor_layout: str = "HND"):
    layout = _layout_id(tensor_layout)
    seq_dim = 1 if layout == 0 else 2

    vm = v.mean(dim=seq_dim)
    v_smoothed = torch.empty(v.shape, dtype=torch.float16, device=v.device)
    _fused.sub_mean_cuda(v, vm, v_smoothed, layout)

    return v_smoothed, vm

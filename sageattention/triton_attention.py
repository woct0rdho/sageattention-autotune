from typing import Optional

import torch

from .core import _pad_qkv
from .triton.attn_qk_int8_per_block import forward as _attn_forward
from .triton.quant_per_block import per_block_int8

LOG2_E = 1.44269504


def sageattn_qk_int8_pv_fp16_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32",
    smooth_k: bool = True,
    return_lse: bool = False,
):
    dtype = q.dtype
    if not q.is_cuda:
        raise ValueError("Input tensors must be CUDA tensors.")
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported dtype: {dtype}")
    if q.device != k.device or q.device != v.device:
        raise ValueError("All tensors must be on the same device.")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("All tensors must have the same dtype.")

    head_dim, q, k, v = _pad_qkv(q, k, v)
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("Last dimension of q, k, and v must be contiguous.")

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    if tensor_layout == "NHD":
        seq_dim = 1
        head_dim_index = 2
    elif tensor_layout == "HND":
        seq_dim = 2
        head_dim_index = 1
    else:
        raise ValueError("tensor_layout must be 'HND' or 'NHD'.")

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        num_qo_heads = q.size(head_dim_index)
        num_kv_heads = k.size(head_dim_index)
        if num_qo_heads % num_kv_heads != 0:
            raise ValueError("num_qo_heads must be divisible by num_kv_heads.")

        if return_lse:
            q_per_kv_heads = num_qo_heads // num_kv_heads
            km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=head_dim_index) if q_per_kv_heads > 1 else km

            if tensor_layout == "NHD":
                lse_correction = torch.matmul(
                    q.transpose(1, 2),
                    km_broadcast.transpose(1, 2).transpose(2, 3),
                ).squeeze(-1)
            else:
                lse_correction = torch.matmul(q, km_broadcast.transpose(2, 3)).squeeze(-1)
            lse_correction = lse_correction.to(torch.float32)
    else:
        km = None

    blk_q = 64 if q.size(-1) == 256 else 128
    blk_k = 32 if q.size(-1) == 256 else 64
    q_int8, q_scale, k_int8, k_scale = per_block_int8(
        q,
        k,
        km=km,
        BLKQ=blk_q,
        BLKK=blk_k,
        tensor_layout=tensor_layout,
    )

    output, lse = _attn_forward(
        q_int8,
        k_int8,
        v.to(torch.float16),
        q_scale,
        k_scale,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        sm_scale=sm_scale,
        pv_accum_dtype=pv_accum_dtype,
        output_dtype=dtype,
        return_lse=return_lse,
    )

    output = output[..., :head_dim]
    if not return_lse:
        return output

    lse = lse / LOG2_E
    if smooth_k:
        lse = lse + lse_correction * sm_scale
    return output, lse

from typing import Optional

import torch

from .triton.attn_qk_int8_per_block import forward as _attn_forward
from .triton.quant_per_block import per_block_int8
from .triton_autotune import _eager_triton_autotune_select, _sageattn_triton_autotuned
from .utils import _pad_qkv

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
    layout_i = {"NHD": 0, "HND": 1}[tensor_layout]
    pv_accum_i = {"fp32": 0, "fp16": 1}[pv_accum_dtype]

    if torch.compiler.is_compiling() and not return_lse:
        if sm_scale is None:
            sm_scale = q.size(-1) ** -0.5
        return _sageattn_triton_autotuned(
            q,
            k,
            v,
            layout_i,
            is_causal,
            float(sm_scale),
            pv_accum_i,
            smooth_k,
        )

    config = _eager_triton_autotune_select(
        q,
        k,
        v,
        layout_i,
        is_causal,
        pv_accum_i,
        sm_scale,
        smooth_k,
        return_lse,
    )

    return _sageattn_triton_configured(
        q,
        k,
        v,
        layout_i,
        is_causal,
        sm_scale,
        pv_accum_i,
        smooth_k,
        return_lse,
        config,
    )


def _sageattn_triton_configured(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout_i: int,
    is_causal: bool,
    sm_scale: Optional[float],
    pv_accum_i: int,
    smooth_k: bool,
    return_lse: bool,
    triton_config: tuple[int, int, int, int, int],
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

    if layout_i == 0:
        tensor_layout = "NHD"
        seq_dim = 1
        head_dim_index = 2
    elif layout_i == 1:
        tensor_layout = "HND"
        seq_dim = 2
        head_dim_index = 1
    else:
        raise ValueError("layout_i must be 0 (NHD) or 1 (HND).")

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

    pv_accum_dtype = {0: "fp32", 1: "fp16"}[pv_accum_i]

    block_m, block_n, quant_num_warps, attn_num_warps, attn_num_stages = triton_config

    q_int8, q_scale, k_int8, k_scale = per_block_int8(
        q,
        k,
        km=km,
        BLKQ=block_m,
        BLKK=block_n,
        tensor_layout=tensor_layout,
        quant_num_warps=quant_num_warps,
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        attn_num_warps=attn_num_warps,
        attn_num_stages=attn_num_stages,
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

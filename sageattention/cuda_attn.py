import warnings
from typing import Optional

import torch

from .cuda_autotune import _eager_autotune_select, _sageattn_autotuned
from .cuda_compile import _qattn_sm80
from .triton.quant_per_thread import per_thread_int8
from .utils import DEFAULT_PV_ACCUM_DTYPE, _pad_qkv

LOG2_E = 1.44269504


def sageattn_qk_int8_pv_fp16_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
) -> torch.Tensor:
    layout_i = {"NHD": 0, "HND": 1}[tensor_layout]
    pv_accum_i = {"fp32": 0, "fp16": 1, "fp16+fp32": 2}[pv_accum_dtype]

    if torch.compiler.is_compiling() and not return_lse:
        if sm_scale is None:
            sm_scale = q.size(-1) ** -0.5
        return _sageattn_autotuned(
            q,
            k,
            v,
            layout_i,
            is_causal,
            float(sm_scale),
            pv_accum_i,
            smooth_k,
            smooth_v,
        )

    qk_config = _eager_autotune_select(
        q,
        k,
        v,
        layout_i,
        is_causal,
        pv_accum_i,
        sm_scale,
        smooth_k,
        smooth_v,
        return_lse,
    )

    return _sageattn_configured(
        q,
        k,
        v,
        layout_i,
        is_causal,
        sm_scale,
        pv_accum_i,
        smooth_k,
        smooth_v,
        return_lse,
        qk_config,
    )


def _sageattn_configured(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout_i: int,
    is_causal: bool,
    sm_scale: Optional[float],
    pv_accum_i: int,
    smooth_k: bool,
    smooth_v: bool,
    return_lse: bool,
    qk_config: tuple[int, int, int, int],
) -> torch.Tensor:
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

    seq_dim = 1 if layout_i == 0 else 2
    head_dim_index = 2 if layout_i == 0 else 1

    is_causal_i = 1 if is_causal else 0
    return_lse_i = 1 if return_lse else 0

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        num_qo_heads = q.size(head_dim_index)
        num_kv_heads = k.size(head_dim_index)
        if num_qo_heads % num_kv_heads != 0:
            raise ValueError("num_qo_heads must be divisible by num_kv_heads.")

        if return_lse:
            q_per_kv_heads = num_qo_heads // num_kv_heads
            km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=head_dim_index) if q_per_kv_heads > 1 else km

            if layout_i == 0:
                lse_correction = torch.matmul(
                    q.transpose(1, 2),
                    km_broadcast.transpose(1, 2).transpose(2, 3),
                ).squeeze(-1)
            else:
                lse_correction = torch.matmul(q, km_broadcast.transpose(2, 3)).squeeze(-1)
            lse_correction = lse_correction.to(torch.float32)
    else:
        km = None

    if pv_accum_i in (0, 2) and smooth_v:
        warnings.warn("pv_accum_dtype is fp32 or fp16+fp32, smooth_v will be ignored.", stacklevel=2)
        smooth_v = False

    blk_q, blk_k, warp_q, warp_k = qk_config

    tensor_layout = "HND" if layout_i == 1 else "NHD"
    q_int8, q_scale, k_int8, k_scale = per_thread_int8(
        q,
        k,
        km,
        tensor_layout=tensor_layout,
        BLKQ=blk_q,
        WARPQ=warp_q,
        BLKK=blk_k,
        WARPK=warp_k,
    )

    output = torch.empty(q.size(), dtype=dtype, device=q.device)

    if pv_accum_i == 0:  # fp32
        lse = _qattn_sm80.qk_int8_sv_f16_accum_f32_attn(
            q_int8,
            k_int8,
            v.to(torch.float16),
            output,
            q_scale,
            k_scale,
            layout_i,
            is_causal_i,
            sm_scale,
            blk_q,
            blk_k,
            warp_q,
            warp_k,
            return_lse_i,
        )
    elif pv_accum_i == 1:  # fp16
        if smooth_v:
            vm = v.mean(dim=seq_dim)
            smoothed_v = (v - vm.unsqueeze(seq_dim)).to(torch.float16)
            lse = _qattn_sm80.qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(
                q_int8,
                k_int8,
                smoothed_v,
                output,
                q_scale,
                k_scale,
                vm,
                layout_i,
                is_causal_i,
                sm_scale,
                blk_q,
                blk_k,
                warp_q,
                warp_k,
                return_lse_i,
            )
        else:
            lse = _qattn_sm80.qk_int8_sv_f16_accum_f16_attn(
                q_int8,
                k_int8,
                v.to(torch.float16),
                output,
                q_scale,
                k_scale,
                layout_i,
                is_causal_i,
                sm_scale,
                blk_q,
                blk_k,
                warp_q,
                warp_k,
                return_lse_i,
            )
    else:  # fp16+fp32
        lse = _qattn_sm80.qk_int8_sv_f16_accum_f16_attn_inst_buf(
            q_int8,
            k_int8,
            v.to(torch.float16),
            output,
            q_scale,
            k_scale,
            layout_i,
            is_causal_i,
            sm_scale,
            blk_q,
            blk_k,
            warp_q,
            warp_k,
            return_lse_i,
        )

    output = output[..., :head_dim]
    if not return_lse:
        return output

    lse = lse / LOG2_E
    if smooth_k:
        lse = lse + lse_correction * sm_scale
    return output, lse

import warnings
from typing import Optional

import torch
import torch.nn.functional as F

from .autotune import (
    _PV_ACCUM_DTYPE_TO_ID,
    _padded_head_dim,
    _select_sm80_qk_config,
    _valid_sm80_qk_configs,
    register_sm80_autotune_op,
)
from .quant import sub_mean
from .sm80_compile import _qattn_sm80
from .triton.quant_per_thread import per_thread_int8

LOG2_E = 1.44269504


def _pad_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    head_dim = q.size(-1)
    pad_to = _padded_head_dim(head_dim)
    if pad_to == head_dim:
        return head_dim, q, k, v

    padding = (0, pad_to - head_dim)
    return head_dim, F.pad(q, padding), F.pad(k, padding), F.pad(v, padding)


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
) -> torch.Tensor:
    return sageattn_qk_int8_pv_fp16_cuda(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        sm_scale=sm_scale,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
        return_lse=return_lse,
    )


def sageattn_qk_int8_pv_fp16_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    qk_config: Optional[tuple[int, int, int, int]] = None,
) -> torch.Tensor:
    if tensor_layout not in ("HND", "NHD"):
        raise ValueError(f"Unsupported tensor_layout: {tensor_layout}")
    if pv_accum_dtype not in ("fp32", "fp16", "fp16+fp32"):
        raise ValueError(f"Unsupported pv_accum_dtype: {pv_accum_dtype}")

    if torch.compiler.is_compiling() and not return_lse and qk_config is None:
        if sm_scale is None:
            sm_scale = q.size(-1) ** -0.5
        layout_i = 1 if tensor_layout == "HND" else 0
        return _sageattn_qk_int8_pv_fp16_cuda_autotuned(
            q,
            k,
            v,
            layout_i,
            is_causal,
            float(sm_scale),
            _PV_ACCUM_DTYPE_TO_ID[pv_accum_dtype],
            smooth_k,
            smooth_v,
        )

    return _sageattn_qk_int8_pv_fp16_cuda_impl(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        sm_scale=sm_scale,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
        return_lse=return_lse,
        qk_config=qk_config,
    )


def _sageattn_qk_int8_pv_fp16_cuda_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    qk_config: Optional[tuple[int, int, int, int]] = None,
) -> torch.Tensor:
    dtype = q.dtype
    if not q.is_cuda:
        raise ValueError("Input tensors must be CUDA tensors.")
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported dtype: {dtype}")
    if tensor_layout not in ("HND", "NHD"):
        raise ValueError(f"Unsupported tensor_layout: {tensor_layout}")
    if pv_accum_dtype not in ("fp32", "fp16", "fp16+fp32"):
        raise ValueError(f"Unsupported pv_accum_dtype: {pv_accum_dtype}")
    if q.device != k.device or q.device != v.device:
        raise ValueError("All tensors must be on the same device.")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("All tensors must have the same dtype.")

    layout_i = 1 if tensor_layout == "HND" else 0
    is_causal_i = 1 if is_causal else 0
    return_lse_i = 1 if return_lse else 0

    head_dim, q, k, v = _pad_qkv(q, k, v)
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("Last dimension of q, k, and v must be contiguous.")

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    seq_dim = 1 if layout_i == 0 else 2
    head_dim_index = 2 if layout_i == 0 else 1

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        num_qo_heads = q.size(head_dim_index)
        num_kv_heads = k.size(head_dim_index)
        if num_qo_heads % num_kv_heads != 0:
            raise ValueError("num_qo_heads must be divisible by num_kv_heads.")

        q_per_kv_heads = num_qo_heads // num_kv_heads
        km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=head_dim_index) if q_per_kv_heads > 1 else km

        if return_lse:
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

    if pv_accum_dtype in ("fp32", "fp16+fp32") and smooth_v:
        warnings.warn(f"pv_accum_dtype is {pv_accum_dtype}, smooth_v will be ignored.", stacklevel=2)
        smooth_v = False

    if qk_config is None:

        def run_config(config: tuple[int, int, int, int]):
            return _sageattn_qk_int8_pv_fp16_cuda_impl(
                q,
                k,
                v,
                tensor_layout=tensor_layout,
                is_causal=is_causal,
                sm_scale=sm_scale,
                pv_accum_dtype=pv_accum_dtype,
                smooth_k=smooth_k,
                smooth_v=smooth_v,
                return_lse=return_lse,
                qk_config=config,
            )

        qk_config = _select_sm80_qk_config(
            q,
            k,
            v,
            tensor_layout,
            is_causal,
            pv_accum_dtype,
            smooth_k,
            smooth_v,
            return_lse,
            run_config,
        )
    elif qk_config not in _valid_sm80_qk_configs(q, is_causal):
        raise ValueError(f"Invalid sm80 QK config for this input: {qk_config}")

    blk_q, blk_k, warp_q, warp_k = qk_config

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

    if pv_accum_dtype == "fp32":
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
    elif pv_accum_dtype == "fp16":
        if smooth_v:
            smoothed_v, vm = sub_mean(v, tensor_layout=tensor_layout)
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
    else:
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


_sageattn_qk_int8_pv_fp16_cuda_autotuned = register_sm80_autotune_op(_sageattn_qk_int8_pv_fp16_cuda_impl)

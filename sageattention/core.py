import warnings
from typing import Optional

import torch
import torch.nn.functional as F

from .quant import per_warp_int8, sub_mean
from .sm80_compile import _qattn_sm80
from .triton.quant_per_thread import per_thread_int8

LOG2_E = 1.44269504

_SM80_QK_TILE_DEFAULT = 0
_SM80_QK_TILE_128X32 = 1
_SM80_QK_TILE_64X64 = 2


def _layout_id(tensor_layout: str) -> int:
    if tensor_layout == "NHD":
        return 0
    if tensor_layout == "HND":
        return 1
    raise ValueError(f"Unknown tensor layout: {tensor_layout}")


def _pad_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    head_dim = q.size(-1)
    if head_dim < 64:
        pad_to = 64
    elif 64 < head_dim < 128:
        pad_to = 128
    elif 128 < head_dim < 256:
        pad_to = 256
    elif head_dim in (64, 128, 256):
        return head_dim, q, k, v
    else:
        raise ValueError(f"Unsupported head_dim: {head_dim}")

    padding = (0, pad_to - head_dim)
    return head_dim, F.pad(q, padding), F.pad(k, padding), F.pad(v, padding)


def _get_sm80_qk_config(
    q: torch.Tensor,
    is_causal: bool,
    qk_quant_gran: str,
    pv_accum_dtype: str,
    smooth_v: bool,
):
    head_dim = q.size(-1)
    warp_q = 16 if head_dim > 64 and pv_accum_dtype == "fp16+fp32" else 32
    blk_q, blk_k, warp_k = 128, 64, 64
    tile_config = _SM80_QK_TILE_DEFAULT

    if pv_accum_dtype == "fp16" and not smooth_v:
        if qk_quant_gran == "per_thread" and head_dim == 128 and not is_causal:
            blk_k = 32
            warp_k = 32
            tile_config = _SM80_QK_TILE_128X32
        elif head_dim in (128, 256) and is_causal:
            blk_q = 64
            tile_config = _SM80_QK_TILE_64X64

    return blk_q, warp_q, blk_k, warp_k, tile_config


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
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
        qk_quant_gran=qk_quant_gran,
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
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
) -> torch.Tensor:
    dtype = q.dtype
    if not q.is_cuda:
        raise ValueError("Input tensors must be CUDA tensors.")
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Input tensors must be torch.float16 or torch.bfloat16.")
    if qk_quant_gran not in ("per_warp", "per_thread"):
        raise ValueError("qk_quant_gran must be 'per_warp' or 'per_thread'.")
    if pv_accum_dtype not in ("fp32", "fp16", "fp16+fp32"):
        raise ValueError(f"Unsupported pv_accum_dtype: {pv_accum_dtype}")
    if q.device != k.device or q.device != v.device:
        raise ValueError("All tensors must be on the same device.")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("All tensors must have the same dtype.")

    layout = _layout_id(tensor_layout)
    is_causal_i = 1 if is_causal else 0
    qk_quant_gran_i = 3 if qk_quant_gran == "per_thread" else 2
    return_lse_i = 1 if return_lse else 0

    head_dim, q, k, v = _pad_qkv(q, k, v)
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("Last dimension of q, k, and v must be contiguous.")

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    seq_dim = 1 if layout == 0 else 2
    head_dim_index = 2 if layout == 0 else 1

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

    blk_q, warp_q, blk_k, warp_k, qk_tile_config = _get_sm80_qk_config(
        q,
        is_causal,
        qk_quant_gran,
        pv_accum_dtype,
        smooth_v,
    )

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8(
            q,
            k,
            km,
            tensor_layout=tensor_layout,
            BLKQ=blk_q,
            WARPQ=warp_q,
            BLKK=blk_k,
        )
    else:
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

    if pv_accum_dtype in ("fp32", "fp16+fp32") and smooth_v:
        warnings.warn(f"pv_accum_dtype is {pv_accum_dtype}, smooth_v will be ignored.", stacklevel=2)
        smooth_v = False

    if pv_accum_dtype == "fp32":
        lse = _qattn_sm80.qk_int8_sv_f16_accum_f32_attn(
            q_int8,
            k_int8,
            v.to(torch.float16),
            output,
            q_scale,
            k_scale,
            layout,
            is_causal_i,
            qk_quant_gran_i,
            sm_scale,
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
                layout,
                is_causal_i,
                qk_quant_gran_i,
                sm_scale,
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
                layout,
                is_causal_i,
                qk_quant_gran_i,
                sm_scale,
                qk_tile_config,
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
            layout,
            is_causal_i,
            qk_quant_gran_i,
            sm_scale,
            return_lse_i,
        )

    output = output[..., :head_dim]
    if not return_lse:
        return output

    lse = lse / LOG2_E
    if smooth_k:
        lse = lse + lse_correction * sm_scale
    return output, lse

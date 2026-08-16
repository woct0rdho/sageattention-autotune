import importlib
from typing import Literal, overload

import torch

from .cutlass_autotune import _eager_autotune_select, _sageattn_cutlass_autotuned
from .triton.quant_per_block import per_block_int8
from .utils import (
    LOG2_E,
    _allocate_forward_outputs,
    _format_forward_outputs,
    _is_compiling_or_fake,
    _lse_correction,
    _pad_qkv,
)

importlib.import_module(f"{__package__}._qattn_cutlass_sm80")
_qattn_cutlass_sm80 = torch.ops.sageattention_qattn_cutlass_sm80

CutlassSageAttnResult = torch.Tensor | tuple[torch.Tensor, torch.Tensor]

_CUTLASS_QK_QUANT_CONFIG = (32, 64)


@overload
def sageattn_qk_int8_pv_fp16_cutlass(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    pv_accum_dtype: str = "fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: Literal[False] = False,
    attn_mask: object = None,
) -> torch.Tensor: ...


@overload
def sageattn_qk_int8_pv_fp16_cutlass(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    pv_accum_dtype: str = "fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: Literal[True] = True,
    attn_mask: object = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...


def sageattn_qk_int8_pv_fp16_cutlass(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    pv_accum_dtype: str = "fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    attn_mask: object = None,  # For ComfyUI compatibility. Not implemented yet.
) -> CutlassSageAttnResult:
    assert attn_mask is None
    if is_causal:
        raise ValueError("CUTLASS qattn currently supports non-causal attention only.")
    if pv_accum_dtype != "fp32":
        raise ValueError("CUTLASS qattn currently supports pv_accum_dtype='fp32' only.")
    if smooth_v:
        raise ValueError("CUTLASS qattn currently does not support smooth_v.")

    if _is_compiling_or_fake(q):
        output, lse = _allocate_forward_outputs(q, tensor_layout, return_lse)
        _sageattn_cutlass_autotuned(q, k, v, output, lse, tensor_layout, smooth_k, return_lse)
        return _format_forward_outputs(output, lse, q.size(-1), return_lse)

    qk_config = _eager_autotune_select(q, k, v, tensor_layout, smooth_k, return_lse)
    return _sageattn_cutlass_configured(q, k, v, tensor_layout, smooth_k, return_lse, qk_config)


@overload
def _sageattn_cutlass_configured(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    smooth_k: bool,
    return_lse: Literal[False],
    qk_config: tuple[int, int, int, int],
) -> torch.Tensor: ...


@overload
def _sageattn_cutlass_configured(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    smooth_k: bool,
    return_lse: Literal[True],
    qk_config: tuple[int, int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]: ...


def _sageattn_cutlass_configured(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    smooth_k: bool,
    return_lse: bool,
    qk_config: tuple[int, int, int, int],
) -> CutlassSageAttnResult:
    output, lse = _allocate_forward_outputs(q, tensor_layout, return_lse)
    _sageattn_cutlass_configured_out(
        q,
        k,
        v,
        output,
        lse,
        tensor_layout,
        smooth_k,
        return_lse,
        qk_config,
    )
    return _format_forward_outputs(output, lse, q.size(-1), return_lse)


def _sageattn_cutlass_configured_out(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor | None,
    tensor_layout: str,
    smooth_k: bool,
    return_lse: bool,
    qk_config: tuple[int, int, int, int],
) -> None:
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("Input tensors must be CUDA tensors.")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise ValueError("CUTLASS qattn currently supports fp16 q, k, and v only.")
    if q.device != k.device or q.device != v.device:
        raise ValueError("All tensors must be on the same device.")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape.")

    head_dim, q, k, v = _pad_qkv(q, k, v)
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("Last dimension of q, k, and v must be contiguous.")

    sm_scale = head_dim**-0.5

    if tensor_layout == "NHD":
        layout_i = 0
        seq_dim_index = 1
        head_dim_index = 2
    elif tensor_layout == "HND":
        layout_i = 1
        seq_dim_index = 2
        head_dim_index = 1
    else:
        raise ValueError("tensor_layout must be 'NHD' or 'HND'.")

    blk_q, blk_k, warp_q, warp_k = qk_config

    if smooth_k:
        km = k.mean(dim=seq_dim_index, keepdim=True)
    else:
        km = None

    quant_blk_q, quant_blk_k = _CUTLASS_QK_QUANT_CONFIG
    q_int8, q_scale, k_int8, k_scale = per_block_int8(
        q,
        k,
        km=km,
        BLKQ=quant_blk_q,
        BLKK=quant_blk_k,
        tensor_layout=tensor_layout,
    )

    _qattn_cutlass_sm80.qk_int8_sv_f16_accum_f32_attn_cutlass(
        q_int8,
        k_int8,
        v,
        output,
        lse,
        q_scale,
        k_scale,
        layout_i,
        sm_scale,
        blk_q,
        blk_k,
        warp_q,
        warp_k,
        return_lse,
    )

    if not return_lse:
        return

    assert lse is not None
    lse /= LOG2_E
    if smooth_k:
        assert km is not None
        lse += _lse_correction(q, km, tensor_layout, head_dim_index) * sm_scale

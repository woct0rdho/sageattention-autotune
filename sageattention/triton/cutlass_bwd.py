"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import triton
import triton.language as tl

from ..autotune_utils import _autotune_seq_len_bucket
from .quant_per_block import _INT8_SCALE_FLOOR, _INT8_SCALE_INV, _round_to_int8

_CUTLASS_BWD_BLOCK_M = 32
_CUTLASS_BWD_BLOCK_K = 16
_CUTLASS_BWD_K_BLOCK = 64
_CUTLASS_BWD_DIM_BLOCKS = 64 // _CUTLASS_BWD_BLOCK_K


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["SEQ_LEN_BUCKET", "HEAD_DIM", "IS_EVEN_M"],
)
@triton.jit
def _preprocess_delta_zero_dq_kernel(
    Out,
    DO,
    Delta,
    DQAccum,
    DOInt8,
    DOScale,
    DSQFactors,
    stride_ob,
    stride_os,
    stride_oh,
    stride_dob,
    stride_dos,
    stride_doh,
    stride_db,
    stride_dh,
    stride_dqab,
    stride_dqas,
    stride_dqah,
    stride_doib,
    stride_dois,
    stride_doih,
    stride_dosb,
    stride_dosh,
    stride_dosq,
    stride_dsqb,
    stride_dsqh,
    stride_dsqq,
    SEQ_LEN: tl.constexpr,
    SEQ_LEN_BUCKET: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DIM_BLOCKS: tl.constexpr,
    IS_EVEN_M: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)
    off_b = tl.program_id(2).to(tl.int64)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    if not IS_EVEN_M:
        mask_m = offs_m < SEQ_LEN

    out_ptrs = Out + off_b * stride_ob + offs_m[:, None] * stride_os + off_h * stride_oh + offs_d[None, :]
    do_ptrs = DO + off_b * stride_dob + offs_m[:, None] * stride_dos + off_h * stride_doh + offs_d[None, :]
    if IS_EVEN_M:
        out = tl.load(out_ptrs).to(tl.float32)
        do = tl.load(do_ptrs).to(tl.float32)
    else:
        out = tl.load(out_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    delta = tl.sum(out * do, axis=1)
    delta_ptrs = Delta + off_b * stride_db + off_h * stride_dh + offs_m
    if IS_EVEN_M:
        tl.store(delta_ptrs, delta)
    else:
        tl.store(delta_ptrs, delta, mask=mask_m)

    dq_accum_ptrs = (
        DQAccum + off_b * stride_dqab + offs_m[:, None] * stride_dqas + off_h * stride_dqah + offs_d[None, :]
    )
    if IS_EVEN_M:
        tl.store(dq_accum_ptrs, tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32))
    else:
        tl.store(dq_accum_ptrs, tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32), mask=mask_m[:, None])

    for dim_block in range(DIM_BLOCKS):
        block_d = dim_block * BLOCK_K + tl.arange(0, BLOCK_K)
        do_block_ptrs = DO + off_b * stride_dob + offs_m[:, None] * stride_dos + off_h * stride_doh + block_d[None, :]
        if IS_EVEN_M:
            do_block = tl.load(do_block_ptrs).to(tl.float32)
        else:
            do_block = tl.load(do_block_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
        max_abs = tl.max(tl.abs(do_block))
        scale = max_abs * _INT8_SCALE_INV + _INT8_SCALE_FLOOR
        do_i8 = _round_to_int8(do_block / scale)

        do_i8_ptrs = (
            DOInt8 + off_b * stride_doib + offs_m[:, None] * stride_dois + off_h * stride_doih + block_d[None, :]
        )
        if IS_EVEN_M:
            tl.store(do_i8_ptrs, do_i8)
        else:
            tl.store(do_i8_ptrs, do_i8, mask=mask_m[:, None])
        do_scale_ptr = DOScale + off_b * stride_dosb + off_h * stride_dosh + start_m * stride_dosq + dim_block
        tl.store(do_scale_ptr, scale)

    do_l2 = tl.sqrt(tl.sum(do * do, axis=1))
    do_l2_max = tl.max(tl.where(mask_m, do_l2, 0.0)) if not IS_EVEN_M else tl.max(do_l2)
    delta_abs_max = tl.max(tl.where(mask_m, tl.abs(delta), 0.0)) if not IS_EVEN_M else tl.max(tl.abs(delta))
    dsq_ptr = DSQFactors + off_b * stride_dsqb + off_h * stride_dsqh + start_m * stride_dsqq
    tl.store(dsq_ptr, do_l2_max)
    tl.store(dsq_ptr + 1, delta_abs_max)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["SEQ_LEN_BUCKET", "HEAD_DIM", "IS_EVEN_N"],
)
@triton.jit
def _v_l2_max_kernel(
    V,
    VScale,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vsb,
    stride_vsh,
    stride_vss,
    SEQ_LEN: tl.constexpr,
    SEQ_LEN_BUCKET: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_EVEN_N: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)
    off_b = tl.program_id(2).to(tl.int64)

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    if not IS_EVEN_N:
        mask_n = offs_n < SEQ_LEN

    v_ptrs = V + off_b * stride_vb + offs_n[:, None] * stride_vs + off_h * stride_vh + offs_d[None, :]
    if IS_EVEN_N:
        v = tl.load(v_ptrs).to(tl.float32)
    else:
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
    v_l2 = tl.sqrt(tl.sum(v * v, axis=1))
    v_l2_max = tl.max(tl.where(mask_n, v_l2, 0.0)) if not IS_EVEN_N else tl.max(v_l2)
    tl.store(VScale + off_b * stride_vsb + off_h * stride_vsh + start_n * stride_vss, v_l2_max)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["SEQ_LEN_BUCKET", "HEAD_DIM", "IS_EVEN_M"],
)
@triton.jit
def _convert_dq_kernel(
    DQAccum,
    DQ,
    stride_dqab,
    stride_dqas,
    stride_dqah,
    stride_dqb,
    stride_dqs,
    stride_dqh,
    SEQ_LEN: tl.constexpr,
    SEQ_LEN_BUCKET: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    IS_EVEN_M: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)
    off_b = tl.program_id(2).to(tl.int64)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    if not IS_EVEN_M:
        mask_m = offs_m < SEQ_LEN

    dq_accum_ptrs = (
        DQAccum + off_b * stride_dqab + offs_m[:, None] * stride_dqas + off_h * stride_dqah + offs_d[None, :]
    )
    dq_ptrs = DQ + off_b * stride_dqb + offs_m[:, None] * stride_dqs + off_h * stride_dqh + offs_d[None, :]
    if IS_EVEN_M:
        dq = tl.load(dq_accum_ptrs)
        tl.store(dq_ptrs, dq.to(DQ.type.element_ty))
    else:
        dq = tl.load(dq_accum_ptrs, mask=mask_m[:, None], other=0.0)
        tl.store(dq_ptrs, dq.to(DQ.type.element_ty), mask=mask_m[:, None])


def _logical_strides(tensor: torch.Tensor, tensor_layout: str) -> tuple[int, int, int]:
    if tensor_layout == "NHD":
        return tensor.stride(0), tensor.stride(1), tensor.stride(2)
    if tensor_layout == "HND":
        return tensor.stride(0), tensor.stride(2), tensor.stride(1)
    raise ValueError("tensor_layout must be 'NHD' or 'HND'.")


def preprocess_delta_zero_dq(
    output: torch.Tensor,
    grad_output: torch.Tensor,
    value: torch.Tensor,
    tensor_layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if output.ndim != 4 or output.shape != grad_output.shape:
        raise ValueError("output and grad_output must have the same four-dimensional shape.")
    if output.dtype != torch.float16 or grad_output.dtype != torch.float16:
        raise ValueError("CUTLASS backward preprocessing requires fp16 output and grad_output tensors.")
    if (
        not output.is_cuda
        or output.device != grad_output.device
        or value.device != output.device
        or value.shape != output.shape
        or value.dtype != torch.float16
    ):
        raise ValueError("output, grad_output, and value must be matching CUDA fp16 tensors.")

    batch = output.size(0)
    heads = output.size(1) if tensor_layout == "HND" else output.size(2)
    seq_len = output.size(2) if tensor_layout == "HND" else output.size(1)
    head_dim = output.size(-1)
    if head_dim != 64:
        raise ValueError("CUTLASS backward preprocessing currently supports head_dim 64 only.")

    output = output.contiguous()
    grad_output = grad_output.contiguous()
    stride_ob, stride_os, stride_oh = _logical_strides(output, tensor_layout)
    stride_dob, stride_dos, stride_doh = _logical_strides(grad_output, tensor_layout)

    q_blocks = triton.cdiv(seq_len, _CUTLASS_BWD_BLOCK_M)
    k_blocks = triton.cdiv(seq_len, _CUTLASS_BWD_K_BLOCK)
    delta = torch.empty((batch, heads, seq_len), device=output.device, dtype=torch.float32)
    dq_accum = torch.empty((batch, heads, seq_len, head_dim), device=output.device, dtype=torch.float32)
    do_int8 = torch.empty((batch, heads, seq_len, head_dim), device=output.device, dtype=torch.int8)
    do_scale = torch.empty((batch, heads, q_blocks, _CUTLASS_BWD_DIM_BLOCKS), device=output.device, dtype=torch.float32)

    ds_q_factors = torch.empty((batch, heads, q_blocks, 2), device=output.device, dtype=torch.float32)
    ds_k_factors = torch.empty((batch, heads, k_blocks), device=output.device, dtype=torch.float32)

    preprocess_grid = (q_blocks, heads, batch)
    _preprocess_delta_zero_dq_kernel[preprocess_grid](
        output,
        grad_output,
        delta,
        dq_accum,
        do_int8,
        do_scale,
        ds_q_factors,
        stride_ob,
        stride_os,
        stride_oh,
        stride_dob,
        stride_dos,
        stride_doh,
        delta.stride(0),
        delta.stride(1),
        dq_accum.stride(0),
        dq_accum.stride(2),
        dq_accum.stride(1),
        do_int8.stride(0),
        do_int8.stride(2),
        do_int8.stride(1),
        do_scale.stride(0),
        do_scale.stride(1),
        do_scale.stride(2),
        ds_q_factors.stride(0),
        ds_q_factors.stride(1),
        ds_q_factors.stride(2),
        SEQ_LEN=seq_len,
        SEQ_LEN_BUCKET=_autotune_seq_len_bucket(seq_len),
        HEAD_DIM=head_dim,
        BLOCK_M=_CUTLASS_BWD_BLOCK_M,
        BLOCK_K=_CUTLASS_BWD_BLOCK_K,
        DIM_BLOCKS=_CUTLASS_BWD_DIM_BLOCKS,
        IS_EVEN_M=seq_len % _CUTLASS_BWD_BLOCK_M == 0,
    )

    value = value.contiguous()
    stride_vb, stride_vs, stride_vh = _logical_strides(value, tensor_layout)
    _v_l2_max_kernel[(k_blocks, heads, batch)](
        value,
        ds_k_factors,
        stride_vb,
        stride_vs,
        stride_vh,
        ds_k_factors.stride(0),
        ds_k_factors.stride(1),
        ds_k_factors.stride(2),
        SEQ_LEN=seq_len,
        SEQ_LEN_BUCKET=_autotune_seq_len_bucket(seq_len),
        HEAD_DIM=head_dim,
        BLOCK_N=_CUTLASS_BWD_K_BLOCK,
        IS_EVEN_N=seq_len % _CUTLASS_BWD_K_BLOCK == 0,
    )

    return delta, dq_accum, do_int8, do_scale, ds_q_factors, ds_k_factors


def convert_dq(dq_accum: torch.Tensor, grad_query: torch.Tensor, tensor_layout: str) -> None:
    if dq_accum.ndim != 4 or dq_accum.dtype != torch.float32 or not dq_accum.is_contiguous():
        raise ValueError("dq_accum must be a contiguous fp32 [batch, heads, seq_len, head_dim] tensor.")
    if grad_query.ndim != 4 or grad_query.dtype != torch.float16 or not grad_query.is_cuda:
        raise ValueError("grad_query must be a CUDA fp16 tensor.")
    batch = grad_query.size(0)
    heads = grad_query.size(1) if tensor_layout == "HND" else grad_query.size(2)
    seq_len = grad_query.size(2) if tensor_layout == "HND" else grad_query.size(1)
    head_dim = grad_query.size(-1)
    if dq_accum.shape != (batch, heads, seq_len, head_dim):
        raise ValueError("dq_accum shape does not match grad_query.")

    _convert_dq_kernel[(triton.cdiv(seq_len, _CUTLASS_BWD_BLOCK_M), heads, batch)](
        dq_accum,
        grad_query,
        dq_accum.stride(0),
        dq_accum.stride(2),
        dq_accum.stride(1),
        grad_query.stride(0),
        grad_query.stride(2 if tensor_layout == "HND" else 1),
        grad_query.stride(1 if tensor_layout == "HND" else 2),
        SEQ_LEN=seq_len,
        SEQ_LEN_BUCKET=_autotune_seq_len_bucket(seq_len),
        HEAD_DIM=head_dim,
        BLOCK_M=_CUTLASS_BWD_BLOCK_M,
        IS_EVEN_M=seq_len % _CUTLASS_BWD_BLOCK_M == 0,
    )

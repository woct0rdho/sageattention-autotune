import os

import torch
import triton
import triton.language as tl

from ..autotune_utils import _autotune_seq_len_bucket
from .attn_bwd_autotune import _TRITON_BWD_FUSED_CONFIGS, _prune_bwd_fused_configs
from .attn_bwd_qk_int8 import LOG2_E, _bwd_preprocess_delta_do_quant
from .quant_per_block import _quantize_tile_to_int8


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["SEQ_LEN_BUCKET", "HEAD_DIM", "BLOCK_M", "DQ_SPLITS", "IS_EVEN_M"],
)
@triton.jit
def _zero_dq_accum_kernel(
    DQAccum,
    stride_dqax,
    stride_dqab,
    stride_dqas,
    stride_dqah,
    SEQ_LEN: tl.constexpr,
    SEQ_LEN_BUCKET: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    DQ_SPLITS: tl.constexpr,
    IS_EVEN_M: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)
    split_batch = tl.program_id(2)
    off_s = (split_batch % DQ_SPLITS).to(tl.int64)
    off_b = (split_batch // DQ_SPLITS).to(tl.int64)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    if not IS_EVEN_M:
        mask_m = offs_m < SEQ_LEN
    ptrs = (
        DQAccum
        + off_s * stride_dqax
        + off_b * stride_dqab
        + offs_m[:, None] * stride_dqas
        + off_h * stride_dqah
        + offs_d[None, :]
    )
    if IS_EVEN_M:
        tl.store(ptrs, tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32))
    else:
        tl.store(ptrs, tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32), mask=mask_m[:, None])


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["SEQ_LEN_BUCKET", "HEAD_DIM", "BLOCK_M", "DQ_SPLITS", "IS_EVEN_M"],
)
@triton.jit
def _convert_dq_accum_kernel(
    DQAccum,
    DQ,
    stride_dqax,
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
    DQ_SPLITS: tl.constexpr,
    IS_EVEN_M: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)
    off_b = tl.program_id(2).to(tl.int64)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    if not IS_EVEN_M:
        mask_m = offs_m < SEQ_LEN
    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for split in range(0, DQ_SPLITS):
        accum_ptrs = (
            DQAccum
            + split * stride_dqax
            + off_b * stride_dqab
            + offs_m[:, None] * stride_dqas
            + off_h * stride_dqah
            + offs_d[None, :]
        )
        if IS_EVEN_M:
            dq += tl.load(accum_ptrs)
        else:
            dq += tl.load(accum_ptrs, mask=mask_m[:, None], other=0.0)

    dq_ptrs = DQ + off_b * stride_dqb + offs_m[:, None] * stride_dqs + off_h * stride_dqh + offs_d[None, :]
    if IS_EVEN_M:
        tl.store(dq_ptrs, dq.to(DQ.type.element_ty))
    else:
        tl.store(dq_ptrs, dq.to(DQ.type.element_ty), mask=mask_m[:, None])


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps, num_stages in _TRITON_BWD_FUSED_CONFIGS
    ],
    key=[
        "SEQ_LEN_BUCKET",
        "HEAD_DIM",
        "BLOCK_M",
        "BLOCK_N",
        "HAS_KMEAN",
        "DQ_SPLITS",
        "N_BLOCKS",
        "IS_EVEN_M",
        "IS_EVEN_N",
    ],
    prune_configs_by={"early_config_prune": _prune_bwd_fused_configs},
    reset_to_zero=["DQAccum", "DK", "DV"],
)
@triton.jit
def _bwd_fused_kernel(
    Q,
    K,
    V,
    DO,
    DOInt8,
    DOScale,
    Lse,
    Delta,
    Q_scale,
    K_scale,
    KMean,
    DQAccum,
    DK,
    DV,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_dob,
    stride_dos,
    stride_doh,
    stride_doib,
    stride_dois,
    stride_doih,
    stride_dosb,
    stride_dosh,
    stride_lseb,
    stride_lseh,
    stride_db,
    stride_dh,
    stride_qsb,
    stride_qsh,
    stride_ksb,
    stride_ksh,
    stride_kmb,
    stride_kmh,
    stride_dqax,
    stride_dqab,
    stride_dqas,
    stride_dqah,
    stride_dkb,
    stride_dks,
    stride_dkh,
    stride_dvb,
    stride_dvs,
    stride_dvh,
    SM_SCALE,
    SM_SCALE_LOG2,
    SEQ_LEN: tl.constexpr,
    SEQ_LEN_BUCKET: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_KMEAN: tl.constexpr,
    DQ_SPLITS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    IS_EVEN_M: tl.constexpr,
    IS_EVEN_N: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)
    off_b = tl.program_id(2).to(tl.int64)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    if not IS_EVEN_N:
        mask_n = offs_n < SEQ_LEN
    dq_split = (start_n % DQ_SPLITS).to(tl.int64)

    k_s_ptrs = K + off_b * stride_kb + offs_n[None, :] * stride_ks + off_h * stride_kh + offs_d[:, None]
    if IS_EVEN_N:
        k_s = tl.load(k_s_ptrs).to(tl.int8)
    else:
        k_s = tl.load(k_s_ptrs, mask=mask_n[None, :], other=0).to(tl.int8)
    k_scale = tl.load(K_scale + off_b * stride_ksb + off_h * stride_ksh + start_n).to(tl.float32)

    v_ptrs = V + off_b * stride_vb + offs_n[:, None] * stride_vs + off_h * stride_vh + offs_d[None, :]
    if IS_EVEN_N:
        v = tl.load(v_ptrs)
    else:
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    acc_dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    acc_dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    km = tl.zeros([HEAD_DIM], dtype=tl.float32)
    if HAS_KMEAN:
        km = tl.load(KMean + off_b * stride_kmb + off_h * stride_kmh + offs_d).to(tl.float32) * SM_SCALE

    for start_m in tl.range(0, SEQ_LEN, BLOCK_M, disable_licm=True):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        q_offsets = start_m + offs_m
        if not IS_EVEN_M:
            mask_m = q_offsets < SEQ_LEN

        q_ptrs = Q + off_b * stride_qb + q_offsets[:, None] * stride_qs + off_h * stride_qh + offs_d[None, :]
        if IS_EVEN_M:
            q = tl.load(q_ptrs).to(tl.int8)
        else:
            q = tl.load(q_ptrs, mask=mask_m[:, None], other=0).to(tl.int8)
        q_scale = tl.load(Q_scale + off_b * stride_qsb + off_h * stride_qsh + start_m // BLOCK_M).to(tl.float32)

        qk = tl.dot(q, k_s).to(tl.float32) * (q_scale * k_scale * SM_SCALE_LOG2)
        if not IS_EVEN_M:
            qk = tl.where(mask_m[:, None], qk, -float("inf"))
        if not IS_EVEN_N:
            qk = tl.where(mask_n[None, :], qk, -float("inf"))
        if IS_EVEN_M:
            lse = tl.load(Lse + off_b * stride_lseb + off_h * stride_lseh + q_offsets)
        else:
            lse = tl.load(Lse + off_b * stride_lseb + off_h * stride_lseh + q_offsets, mask=mask_m, other=0.0)
        p = tl.math.exp2(qk - lse[:, None])

        p_i8, p_scale = _quantize_tile_to_int8(p)
        do_i8_ptrs = (
            DOInt8 + off_b * stride_doib + q_offsets[:, None] * stride_dois + off_h * stride_doih + offs_d[None, :]
        )
        if IS_EVEN_M:
            do_i8 = tl.load(do_i8_ptrs).to(tl.int8)
        else:
            do_i8 = tl.load(do_i8_ptrs, mask=mask_m[:, None], other=0).to(tl.int8)
        do_scale = tl.load(DOScale + off_b * stride_dosb + off_h * stride_dosh + start_m // BLOCK_M).to(tl.float32)
        acc_dv += tl.dot(tl.trans(p_i8), do_i8).to(tl.float32) * (p_scale * do_scale)

        do_ptrs = DO + off_b * stride_dob + q_offsets[:, None] * stride_dos + off_h * stride_doh + offs_d[None, :]
        if IS_EVEN_M:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        if IS_EVEN_M:
            delta = tl.load(Delta + off_b * stride_db + off_h * stride_dh + q_offsets)
        else:
            delta = tl.load(Delta + off_b * stride_db + off_h * stride_dh + q_offsets, mask=mask_m, other=0.0)
        dp = p * (dp - delta[:, None])
        if HAS_KMEAN:
            rowsum_ds = tl.sum(dp, axis=1)
        ds_i8, ds_scale = _quantize_tile_to_int8(dp)

        acc_dk += tl.dot(tl.trans(ds_i8), q).to(tl.float32) * (ds_scale * q_scale * SM_SCALE)
        dq_partial = tl.dot(ds_i8, tl.trans(k_s)).to(tl.float32) * (ds_scale * k_scale * SM_SCALE)
        if HAS_KMEAN:
            dq_partial += rowsum_ds[:, None] * km[None, :]

        dq_accum_ptrs = (
            DQAccum
            + dq_split * stride_dqax
            + off_b * stride_dqab
            + q_offsets[:, None] * stride_dqas
            + off_h * stride_dqah
            + offs_d[None, :]
        )
        if DQ_SPLITS >= N_BLOCKS:
            if IS_EVEN_M:
                tl.store(dq_accum_ptrs, dq_partial)
            else:
                tl.store(dq_accum_ptrs, dq_partial, mask=mask_m[:, None])
        elif IS_EVEN_M:
            tl.atomic_add(dq_accum_ptrs, dq_partial, sem="relaxed")
        else:
            tl.atomic_add(dq_accum_ptrs, dq_partial, sem="relaxed", mask=mask_m[:, None])

    dk_ptrs = DK + off_b * stride_dkb + offs_n[:, None] * stride_dks + off_h * stride_dkh + offs_d[None, :]
    dv_ptrs = DV + off_b * stride_dvb + offs_n[:, None] * stride_dvs + off_h * stride_dvh + offs_d[None, :]
    if IS_EVEN_N:
        tl.store(dk_ptrs, acc_dk.to(DK.type.element_ty))
        tl.store(dv_ptrs, acc_dv.to(DV.type.element_ty))
    else:
        tl.store(dk_ptrs, acc_dk.to(DK.type.element_ty), mask=mask_n[:, None])
        tl.store(dv_ptrs, acc_dv.to(DV.type.element_ty), mask=mask_n[:, None])


def _fused_dq_splits(n_blocks: int) -> int:
    if n_blocks <= 1:
        return 1

    override = os.environ.get("SAGEATTN_FUSED_DQ_SPLITS")
    if override is None:
        return 1

    dq_splits = int(override)
    if dq_splits < 1:
        raise ValueError("SAGEATTN_FUSED_DQ_SPLITS must be a positive integer.")
    return min(dq_splits, n_blocks)


def backward_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    k_mean: torch.Tensor | None,
    BLOCK_M: int,
    BLOCK_N: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.ndim != 4:
        raise ValueError("q must have shape [batch, seqlen, heads, head_dim].")
    if q.shape != k.shape or q.shape != v.shape or q.shape != do.shape or q.shape != out.shape:
        raise ValueError("q, k, v, do, and out must have the same NHD shape.")
    if q.dtype != torch.int8 or k.dtype != torch.int8:
        raise ValueError("q and k must be INT8 tensors from the forward quantization step.")
    if v.dtype != torch.float16 or do.dtype != torch.float16 or out.dtype != torch.float16:
        raise ValueError("fused SageBwd Triton backward only supports FP16 v, do, and out tensors.")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous() or not do.is_contiguous():
        raise ValueError("q, k, v, and do must be contiguous NHD tensors.")

    batch, seq_len, heads, head_dim = q.shape
    seq_len_bucket = _autotune_seq_len_bucket(seq_len)
    is_even_m = seq_len % BLOCK_M == 0
    is_even_n = seq_len % BLOCK_N == 0

    n_blocks = triton.cdiv(seq_len, BLOCK_N)
    dq_splits = _fused_dq_splits(n_blocks)

    dq = torch.empty_like(v)
    dk = torch.empty_like(v)
    dv = torch.empty_like(v)
    dq_accum = torch.empty((dq_splits, batch, seq_len, heads, head_dim), device=do.device, dtype=torch.float32)
    delta = torch.empty((batch, heads, seq_len), device=do.device, dtype=torch.float32)
    do_int8 = torch.empty(do.shape, device=do.device, dtype=torch.int8)
    do_scale = torch.empty((batch, heads, triton.cdiv(seq_len, BLOCK_M)), device=do.device, dtype=torch.float32)

    q_grid = (triton.cdiv(seq_len, BLOCK_M), heads, batch)
    q_split_grid = (triton.cdiv(seq_len, BLOCK_M), heads, batch * dq_splits)
    _zero_dq_accum_kernel[q_split_grid](
        dq_accum,
        dq_accum.stride(0),
        dq_accum.stride(1),
        dq_accum.stride(2),
        dq_accum.stride(3),
        SEQ_LEN=seq_len,
        SEQ_LEN_BUCKET=seq_len_bucket,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        DQ_SPLITS=dq_splits,
        IS_EVEN_M=is_even_m,
    )

    _bwd_preprocess_delta_do_quant[q_grid](
        out,
        do,
        delta,
        do_int8,
        do_scale,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        delta.stride(0),
        delta.stride(1),
        do_int8.stride(0),
        do_int8.stride(1),
        do_int8.stride(2),
        do_scale.stride(0),
        do_scale.stride(1),
        SEQ_LEN=seq_len,
        SEQ_LEN_BUCKET=seq_len_bucket,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        IS_EVEN_M=is_even_m,
    )

    has_kmean = k_mean is not None
    if k_mean is None:
        k_mean = torch.empty((1,), device=do.device, dtype=torch.float16)
        kmean_stride_b = 0
        kmean_stride_h = 0
    else:
        k_mean = k_mean.contiguous()
        kmean_stride_b = k_mean.stride(0)
        kmean_stride_h = k_mean.stride(1)

    sm_scale = head_dim**-0.5
    sm_scale_log2 = sm_scale * LOG2_E

    kv_grid = (n_blocks, heads, batch)
    _bwd_fused_kernel[kv_grid](
        q,
        k,
        v,
        do,
        do_int8,
        do_scale,
        lse,
        delta,
        q_scale,
        k_scale,
        k_mean,
        dq_accum,
        dk,
        dv,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        do_int8.stride(0),
        do_int8.stride(1),
        do_int8.stride(2),
        do_scale.stride(0),
        do_scale.stride(1),
        lse.stride(0),
        lse.stride(1),
        delta.stride(0),
        delta.stride(1),
        q_scale.stride(0),
        q_scale.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        kmean_stride_b,
        kmean_stride_h,
        dq_accum.stride(0),
        dq_accum.stride(1),
        dq_accum.stride(2),
        dq_accum.stride(3),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        sm_scale,
        sm_scale_log2,
        SEQ_LEN=seq_len,
        SEQ_LEN_BUCKET=seq_len_bucket,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HAS_KMEAN=has_kmean,
        DQ_SPLITS=dq_splits,
        N_BLOCKS=n_blocks,
        IS_EVEN_M=is_even_m,
        IS_EVEN_N=is_even_n,
    )

    _convert_dq_accum_kernel[q_grid](
        dq_accum,
        dq,
        dq_accum.stride(0),
        dq_accum.stride(1),
        dq_accum.stride(2),
        dq_accum.stride(3),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        SEQ_LEN=seq_len,
        SEQ_LEN_BUCKET=seq_len_bucket,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        DQ_SPLITS=dq_splits,
        IS_EVEN_M=is_even_m,
    )

    return dq, dk, dv

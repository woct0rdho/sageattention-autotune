import torch
import triton
import triton.language as tl

from ..autotune_utils import _autotune_seq_len_bucket
from .attn_bwd_autotune import _TRITON_BWD_CONFIGS, _prune_bwd_dkdv_configs, _prune_bwd_dq_configs

LOG2_E = 1.44269504088896340736


@triton.jit
def _quantize_tile_to_int8(x, mask):
    x = tl.where(mask, x, 0.0)
    scale = tl.max(tl.abs(x)) / 127.0 + 1e-7
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    x_int8 = tl.where(mask, x_int8, 0)
    return x_int8, scale


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["SEQ_LEN_BUCKET", "HEAD_DIM", "BLOCK_M"],
)
@triton.jit
def _bwd_preprocess_delta(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_os,
    stride_oh,
    stride_dob,
    stride_dos,
    stride_doh,
    stride_db,
    stride_dh,
    SEQ_LEN: tl.constexpr,
    SEQ_LEN_BUCKET: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)
    off_b = tl.program_id(2).to(tl.int64)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < SEQ_LEN

    out_ptrs = Out + off_b * stride_ob + offs_m[:, None] * stride_os + off_h * stride_oh + offs_d[None, :]
    do_ptrs = DO + off_b * stride_dob + offs_m[:, None] * stride_dos + off_h * stride_doh + offs_d[None, :]

    out = tl.load(out_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(out * do, axis=1)

    delta_ptrs = Delta + off_b * stride_db + off_h * stride_dh + offs_m
    tl.store(delta_ptrs, delta, mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages) for num_warps, num_stages in _TRITON_BWD_CONFIGS
    ],
    key=["SEQ_LEN_BUCKET", "HEAD_DIM", "BLOCK_M", "BLOCK_N", "HAS_KMEAN"],
    prune_configs_by={"early_config_prune": _prune_bwd_dq_configs},
)
@triton.jit
def _bwd_dq_kernel(
    Q,
    K,
    V,
    DO,
    Lse,
    Delta,
    Q_scale,
    K_scale,
    KMean,
    DQ,
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
    stride_dqb,
    stride_dqs,
    stride_dqh,
    SM_SCALE,
    SM_SCALE_LOG2,
    SEQ_LEN: tl.constexpr,
    SEQ_LEN_BUCKET: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_KMEAN: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)
    off_b = tl.program_id(2).to(tl.int64)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < SEQ_LEN

    q_ptrs = Q + off_b * stride_qb + offs_m[:, None] * stride_qs + off_h * stride_qh + offs_d[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0).to(tl.int8)
    q_scale = tl.load(Q_scale + off_b * stride_qsb + off_h * stride_qsh + start_m).to(tl.float32)

    do_ptrs = DO + off_b * stride_dob + offs_m[:, None] * stride_dos + off_h * stride_doh + offs_d[None, :]
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)

    lse = tl.load(Lse + off_b * stride_lseb + off_h * stride_lseh + offs_m, mask=mask_m, other=float("inf"))
    delta = tl.load(Delta + off_b * stride_db + off_h * stride_dh + offs_m, mask=mask_m, other=0.0)

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    km = tl.zeros([HEAD_DIM], dtype=tl.float32)
    if HAS_KMEAN:
        km = tl.load(KMean + off_b * stride_kmb + off_h * stride_kmh + offs_d).to(tl.float32)

    for start_n in range(0, SEQ_LEN, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_offsets = start_n + offs_n
        mask_n = kv_offsets < SEQ_LEN
        tile_mask = mask_m[:, None] & mask_n[None, :]

        k_s_ptrs = K + off_b * stride_kb + kv_offsets[None, :] * stride_ks + off_h * stride_kh + offs_d[:, None]
        k_s = tl.load(k_s_ptrs, mask=mask_n[None, :], other=0).to(tl.int8)
        k_scale = tl.load(K_scale + off_b * stride_ksb + off_h * stride_ksh + start_n // BLOCK_N).to(tl.float32)

        qk = tl.dot(q, k_s).to(tl.float32) * (q_scale * k_scale * SM_SCALE_LOG2)
        qk = tl.where(tile_mask, qk, -float("inf"))
        p = tl.math.exp2(qk - lse[:, None])
        p = tl.where(tile_mask, p, 0.0)

        v_ptrs = V + off_b * stride_vb + kv_offsets[:, None] * stride_vs + off_h * stride_vh + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - delta[:, None])
        ds = tl.where(tile_mask, ds, 0.0)

        ds_i8, ds_scale = _quantize_tile_to_int8(ds, tile_mask)

        k_dq_ptrs = K + off_b * stride_kb + kv_offsets[:, None] * stride_ks + off_h * stride_kh + offs_d[None, :]
        k_dq = tl.load(k_dq_ptrs, mask=mask_n[:, None], other=0).to(tl.int8)
        acc += tl.dot(ds_i8, k_dq).to(tl.float32) * (ds_scale * k_scale * SM_SCALE)

        if HAS_KMEAN:
            rowsum_ds = tl.sum(ds, axis=1)
            acc += rowsum_ds[:, None] * km[None, :] * SM_SCALE

    dq_ptrs = DQ + off_b * stride_dqb + offs_m[:, None] * stride_dqs + off_h * stride_dqh + offs_d[None, :]
    tl.store(dq_ptrs, acc.to(DQ.type.element_ty), mask=mask_m[:, None])


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages) for num_warps, num_stages in _TRITON_BWD_CONFIGS
    ],
    key=["SEQ_LEN_BUCKET", "HEAD_DIM", "BLOCK_M", "BLOCK_N"],
    prune_configs_by={"early_config_prune": _prune_bwd_dkdv_configs},
)
@triton.jit
def _bwd_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    Lse,
    Delta,
    Q_scale,
    K_scale,
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
    stride_lseb,
    stride_lseh,
    stride_db,
    stride_dh,
    stride_qsb,
    stride_qsh,
    stride_ksb,
    stride_ksh,
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
):
    start_n = tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)
    off_b = tl.program_id(2).to(tl.int64)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_n = offs_n < SEQ_LEN

    k_s_ptrs = K + off_b * stride_kb + offs_n[None, :] * stride_ks + off_h * stride_kh + offs_d[:, None]
    k_s = tl.load(k_s_ptrs, mask=mask_n[None, :], other=0).to(tl.int8)
    k_scale = tl.load(K_scale + off_b * stride_ksb + off_h * stride_ksh + start_n).to(tl.float32)

    v_ptrs = V + off_b * stride_vb + offs_n[:, None] * stride_vs + off_h * stride_vh + offs_d[None, :]
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    acc_dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    acc_dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    for start_m in range(0, SEQ_LEN, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        q_offsets = start_m + offs_m
        mask_m = q_offsets < SEQ_LEN
        tile_mask = mask_m[:, None] & mask_n[None, :]

        q_ptrs = Q + off_b * stride_qb + q_offsets[:, None] * stride_qs + off_h * stride_qh + offs_d[None, :]
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0).to(tl.int8)
        q_scale = tl.load(Q_scale + off_b * stride_qsb + off_h * stride_qsh + start_m // BLOCK_M).to(tl.float32)

        qk = tl.dot(q, k_s).to(tl.float32) * (q_scale * k_scale * SM_SCALE_LOG2)
        qk = tl.where(tile_mask, qk, -float("inf"))
        lse = tl.load(Lse + off_b * stride_lseb + off_h * stride_lseh + q_offsets, mask=mask_m, other=float("inf"))
        p = tl.math.exp2(qk - lse[:, None])
        p = tl.where(tile_mask, p, 0.0)

        do_ptrs = DO + off_b * stride_dob + q_offsets[:, None] * stride_dos + off_h * stride_doh + offs_d[None, :]
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)

        p_i8, p_scale = _quantize_tile_to_int8(p, tile_mask)
        do_i8, do_scale = _quantize_tile_to_int8(do.to(tl.float32), mask_m[:, None])
        acc_dv += tl.dot(tl.trans(p_i8), do_i8).to(tl.float32) * (p_scale * do_scale)

        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        delta = tl.load(Delta + off_b * stride_db + off_h * stride_dh + q_offsets, mask=mask_m, other=0.0)
        ds = p * (dp - delta[:, None])
        ds = tl.where(tile_mask, ds, 0.0)
        ds_i8, ds_scale = _quantize_tile_to_int8(ds, tile_mask)
        acc_dk += tl.dot(tl.trans(ds_i8), q).to(tl.float32) * (ds_scale * q_scale * SM_SCALE)

    dk_ptrs = DK + off_b * stride_dkb + offs_n[:, None] * stride_dks + off_h * stride_dkh + offs_d[None, :]
    dv_ptrs = DV + off_b * stride_dvb + offs_n[:, None] * stride_dvs + off_h * stride_dvh + offs_d[None, :]
    tl.store(dk_ptrs, acc_dk.to(DK.type.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptrs, acc_dv.to(DV.type.element_ty), mask=mask_n[:, None])


def backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    k_mean: torch.Tensor | None,
    BLOCK_M: int = 128,
    BLOCK_N: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.ndim != 4:
        raise ValueError("q must have shape [batch, seqlen, heads, head_dim].")
    if q.shape != k.shape or q.shape != v.shape or q.shape != do.shape or q.shape != out.shape:
        raise ValueError("q, k, v, do, and out must have the same NHD shape.")
    if q.dtype != torch.int8 or k.dtype != torch.int8:
        raise ValueError("q and k must be INT8 tensors from the forward quantization step.")
    if v.dtype != torch.float16 or do.dtype != torch.float16 or out.dtype != torch.float16:
        raise ValueError("initial SageBwd Triton backward only supports FP16 v, do, and out tensors.")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous() or not do.is_contiguous():
        raise ValueError("q, k, v, and do must be contiguous NHD tensors.")

    batch, seq_len, heads, head_dim = q.shape
    seq_len_bucket = _autotune_seq_len_bucket(seq_len)

    dq = torch.empty_like(v)
    dk = torch.empty_like(v)
    dv = torch.empty_like(v)
    delta = torch.empty((batch, heads, seq_len), device=do.device, dtype=torch.float32)

    preprocess_grid = (triton.cdiv(seq_len, BLOCK_M), heads, batch)
    _bwd_preprocess_delta[preprocess_grid](
        out,
        do,
        delta,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        delta.stride(0),
        delta.stride(1),
        SEQ_LEN=seq_len,
        SEQ_LEN_BUCKET=seq_len_bucket,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
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

    dq_grid = (triton.cdiv(seq_len, BLOCK_M), heads, batch)
    _bwd_dq_kernel[dq_grid](
        q,
        k,
        v,
        do,
        lse,
        delta,
        q_scale,
        k_scale,
        k_mean,
        dq,
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
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        sm_scale,
        sm_scale_log2,
        SEQ_LEN=seq_len,
        SEQ_LEN_BUCKET=seq_len_bucket,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HAS_KMEAN=has_kmean,
    )

    dkdv_grid = (triton.cdiv(seq_len, BLOCK_N), heads, batch)
    _bwd_dkdv_kernel[dkdv_grid](
        q,
        k,
        v,
        do,
        lse,
        delta,
        q_scale,
        k_scale,
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
        lse.stride(0),
        lse.stride(1),
        delta.stride(0),
        delta.stride(1),
        q_scale.stride(0),
        q_scale.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
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
    )

    return dq, dk, dv

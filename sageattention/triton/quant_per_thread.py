import torch
import triton
import triton.language as tl


@triton.jit
def _quant_query_per_thread_int8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    seq_len,
    stride_input_b,
    stride_input_h,
    stride_input_n,
    stride_output_b,
    stride_output_h,
    stride_output_n,
    stride_scale_b,
    stride_scale_h,
    head_dim: tl.constexpr,
    warp_block: tl.constexpr,
):
    block_id = tl.program_id(0) // 8
    thread_group_id = tl.program_id(0) % 8
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    offsets_n = block_id * warp_block + tl.arange(0, warp_block // 8) * 8 + thread_group_id
    offsets_d = tl.arange(0, head_dim)

    input_offsets = (
        batch_id * stride_input_b + head_id * stride_input_h + offsets_n[:, None] * stride_input_n + offsets_d[None, :]
    )
    output_offsets = (
        batch_id * stride_output_b
        + head_id * stride_output_h
        + offsets_n[:, None] * stride_output_n
        + offsets_d[None, :]
    )
    scale_offsets = batch_id * stride_scale_b + head_id * stride_scale_h + block_id * 8 + thread_group_id

    values = tl.load(input_ptr + input_offsets, mask=offsets_n[:, None] < seq_len).to(tl.float32)
    scale = tl.max(tl.abs(values)) / 127.0 + 0.0000001
    values = values / scale
    values += 0.5 * tl.where(values >= 0, 1, -1)

    tl.store(output_ptr + output_offsets, values.to(tl.int8), mask=offsets_n[:, None] < seq_len)
    tl.store(scale_ptr + scale_offsets, scale)


@triton.jit
def _quant_key_per_thread_int8_kernel(
    input_ptr,
    mean_ptr,
    output_ptr,
    scale_ptr,
    seq_len,
    stride_input_b,
    stride_input_h,
    stride_input_n,
    stride_mean_b,
    stride_mean_h,
    stride_mean_d,
    stride_output_b,
    stride_output_h,
    stride_output_n,
    stride_scale_b,
    stride_scale_h,
    head_dim: tl.constexpr,
    warp_block: tl.constexpr,
    HAS_MEAN: tl.constexpr,
):
    block_id = tl.program_id(0) // 4
    thread_group_id = tl.program_id(0) % 4
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    offsets_n0 = block_id * warp_block + tl.arange(0, warp_block // 8) * 8 + thread_group_id * 2
    offsets_n1 = offsets_n0 + 1
    offsets_d = tl.arange(0, head_dim)

    input_offsets0 = (
        batch_id * stride_input_b + head_id * stride_input_h + offsets_n0[:, None] * stride_input_n + offsets_d[None, :]
    )
    input_offsets1 = (
        batch_id * stride_input_b + head_id * stride_input_h + offsets_n1[:, None] * stride_input_n + offsets_d[None, :]
    )
    output_offsets0 = (
        batch_id * stride_output_b
        + head_id * stride_output_h
        + offsets_n0[:, None] * stride_output_n
        + offsets_d[None, :]
    )
    output_offsets1 = (
        batch_id * stride_output_b
        + head_id * stride_output_h
        + offsets_n1[:, None] * stride_output_n
        + offsets_d[None, :]
    )
    scale_offsets = batch_id * stride_scale_b + head_id * stride_scale_h + block_id * 4 + thread_group_id

    values0 = tl.load(input_ptr + input_offsets0, mask=offsets_n0[:, None] < seq_len).to(tl.float32)
    values1 = tl.load(input_ptr + input_offsets1, mask=offsets_n1[:, None] < seq_len).to(tl.float32)

    if HAS_MEAN:
        mean_offsets = batch_id * stride_mean_b + head_id * stride_mean_h + offsets_d * stride_mean_d
        mean = tl.load(mean_ptr + mean_offsets).to(tl.float32)
        values0 -= mean[None, :]
        values1 -= mean[None, :]

    scale = tl.maximum(tl.max(tl.abs(values0)), tl.max(tl.abs(values1))) / 127.0 + 0.0000001

    values0 = values0 / scale
    values1 = values1 / scale
    values0 += 0.5 * tl.where(values0 >= 0, 1, -1)
    values1 += 0.5 * tl.where(values1 >= 0, 1, -1)

    tl.store(output_ptr + output_offsets0, values0.to(tl.int8), mask=offsets_n0[:, None] < seq_len)
    tl.store(output_ptr + output_offsets1, values1.to(tl.int8), mask=offsets_n1[:, None] < seq_len)
    tl.store(scale_ptr + scale_offsets, scale)


def per_thread_int8(
    q,
    k,
    km=None,
    BLKQ=128,
    WARPQ=32,
    BLKK=64,
    WARPK=64,
    tensor_layout="HND",
):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if tensor_layout == "HND":
        batch_size, num_qo_heads, qo_len, head_dim = q.shape
        _, num_kv_heads, kv_len, _ = k.shape
        q_strides = (q.stride(0), q.stride(1), q.stride(2))
        q_int8_strides = (q_int8.stride(0), q_int8.stride(1), q_int8.stride(2))
        k_strides = (k.stride(0), k.stride(1), k.stride(2))
        k_int8_strides = (k_int8.stride(0), k_int8.stride(1), k_int8.stride(2))
        if km is not None:
            km = km.squeeze(2)
    elif tensor_layout == "NHD":
        batch_size, qo_len, num_qo_heads, head_dim = q.shape
        _, kv_len, num_kv_heads, _ = k.shape
        q_strides = (q.stride(0), q.stride(2), q.stride(1))
        q_int8_strides = (q_int8.stride(0), q_int8.stride(2), q_int8.stride(1))
        k_strides = (k.stride(0), k.stride(2), k.stride(1))
        k_int8_strides = (k_int8.stride(0), k_int8.stride(2), k_int8.stride(1))
        if km is not None:
            km = km.squeeze(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    has_mean = km is not None
    mean = km if has_mean else k
    mean_strides = (mean.stride(0), mean.stride(1), mean.stride(2)) if has_mean else (0, 0, 0)

    q_scale = torch.empty(
        (batch_size, num_qo_heads, ((qo_len + BLKQ - 1) // BLKQ) * (BLKQ // WARPQ) * 8),
        device=q.device,
        dtype=torch.float32,
    )
    k_scale = torch.empty(
        (batch_size, num_kv_heads, ((kv_len + BLKK - 1) // BLKK) * (BLKK // WARPK) * 4),
        device=q.device,
        dtype=torch.float32,
    )

    grid = ((qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8, num_qo_heads, batch_size)
    _quant_query_per_thread_int8_kernel[grid](
        q,
        q_int8,
        q_scale,
        qo_len,
        *q_strides,
        *q_int8_strides,
        q_scale.stride(0),
        q_scale.stride(1),
        head_dim=head_dim,
        warp_block=WARPQ,
    )

    grid = ((kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4, num_kv_heads, batch_size)
    _quant_key_per_thread_int8_kernel[grid](
        k,
        mean,
        k_int8,
        k_scale,
        kv_len,
        *k_strides,
        *mean_strides,
        *k_int8_strides,
        k_scale.stride(0),
        k_scale.stride(1),
        head_dim=head_dim,
        warp_block=WARPK,
        HAS_MEAN=has_mean,
    )

    return q_int8, q_scale, k_int8, k_scale

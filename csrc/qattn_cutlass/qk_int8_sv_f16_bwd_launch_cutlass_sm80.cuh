#pragma once

#include "attn_cutlass_sm80.h"
#include "qk_int8_sv_f16_bwd_kernel_cutlass_sm80.cuh"
#include "../utils.cuh"

#include <cuda_fp16.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cstdint>
#include <stdexcept>

namespace sageattention::qattn_cutlass_bwd {

enum class TensorLayout {
  kNHD,
  kHND,
};

inline constexpr int32_t div_ceil_int(const int32_t x, const int32_t y)
{
  return (x + y - 1) / y;
}

inline TensorLayout parse_tensor_layout(const int64_t tensor_layout)
{
  if (tensor_layout == 0)
  {
    return TensorLayout::kNHD;
  }
  if (tensor_layout == 1)
  {
    return TensorLayout::kHND;
  }
  throw std::invalid_argument("tensor_layout must be 0 (NHD) or 1 (HND)");
}

inline void check_half_tensor(const Tensor &tensor, const char *const name)
{
  CHECK_CUDA(tensor);
  CHECK_LASTDIM_CONTIGUOUS(tensor);
  CHECK_DTYPE(tensor, torch::headeronly::ScalarType::Half);
  CHECK_DIMS(tensor, 4);
}

inline void check_int8_tensor(const Tensor &tensor, const char *const name)
{
  CHECK_CUDA(tensor);
  CHECK_LASTDIM_CONTIGUOUS(tensor);
  CHECK_DTYPE(tensor, torch::headeronly::ScalarType::Char);
  CHECK_DIMS(tensor, 4);
}

inline void check_scale_tensor(const Tensor &tensor, const char *const name)
{
  CHECK_CUDA(tensor);
  CHECK_CONTIGUOUS(tensor);
  CHECK_DTYPE(tensor, torch::headeronly::ScalarType::Float);
  CHECK_DIMS(tensor, 3);
}

inline void check_workspace_tensor(const Tensor &tensor, const char *const name, const int64_t dims, const torch::headeronly::ScalarType dtype)
{
  CHECK_CUDA(tensor);
  CHECK_CONTIGUOUS(tensor);
  CHECK_DTYPE(tensor, dtype);
  CHECK_DIMS(tensor, dims);
}

inline Params prepare_params(const Tensor &query,
                                    const Tensor &key,
                                    const Tensor &query_scale,
                                    const Tensor &key_scale,
                                    const Tensor &value,
                                    const Tensor &dO,
                                    const Tensor &lse,
                                    const Tensor &delta,
                                    const Tensor &dQ_accum,
                                    const Tensor &dS_sum,
                                    const Tensor &dO_int8,
                                    const Tensor &dO_scale,
                                    const Tensor &dS_q_factors,
                                    const Tensor &dS_k_factors,
                                    const Tensor &dK,
                                    const Tensor &dV,
                                    const TensorLayout tensor_layout,
                                    const int64_t blk_q,
                                    const int64_t blk_k,
                                    const int64_t bwd_block_m,
                                    const int64_t bwd_block_n,
                                    const bool smooth_k)
{
  check_int8_tensor(query, "query");
  check_int8_tensor(key, "key");
  check_scale_tensor(query_scale, "query_scale");
  check_scale_tensor(key_scale, "key_scale");
  check_half_tensor(value, "value");
  check_half_tensor(dO, "dO");
  check_half_tensor(dK, "dK");
  check_half_tensor(dV, "dV");
  CHECK_CUDA(lse);
  CHECK_CONTIGUOUS(lse);
  CHECK_DTYPE(lse, torch::headeronly::ScalarType::Float);
  CHECK_DIMS(lse, 3);
  check_workspace_tensor(delta, "delta", 3, torch::headeronly::ScalarType::Float);
  check_workspace_tensor(dQ_accum, "dQ_accum", 4, torch::headeronly::ScalarType::Float);
  check_workspace_tensor(dS_sum, "dS_sum", 3, torch::headeronly::ScalarType::Float);
  check_workspace_tensor(dO_int8, "dO_int8", 4, torch::headeronly::ScalarType::Char);
  check_workspace_tensor(dO_scale, "dO_scale", 4, torch::headeronly::ScalarType::Float);
  check_workspace_tensor(dS_q_factors, "dS_q_factors", 4, torch::headeronly::ScalarType::Float);
  check_workspace_tensor(dS_k_factors, "dS_k_factors", 3, torch::headeronly::ScalarType::Float);

  STD_TORCH_CHECK(query.size(3) == 64 || query.size(3) == 128, "CUTLASS qattn backward currently supports head_dim 64 or 128");
  STD_TORCH_CHECK(blk_q == 32 && blk_k == 64, "Focused CUTLASS qattn backward requires QBlock=32 and KBlock=64");
  Params params = {
    static_cast<int32_t>(query.size(3)),
    static_cast<int32_t>(query.size(0)),
    0,
    0,
    static_cast<int32_t>(bwd_block_m),
    static_cast<int32_t>(bwd_block_n),
    static_cast<int32_t>(blk_q),
    static_cast<int32_t>(blk_k),
    static_cast<int32_t>(query.stride(0)), 0, 0,
    static_cast<int32_t>(key.stride(0)), 0, 0,
    static_cast<int32_t>(value.stride(0)), 0, 0,
    static_cast<int32_t>(dO.stride(0)), 0, 0,
    static_cast<int32_t>(dQ_accum.stride(0)),
    static_cast<int32_t>(dQ_accum.stride(1)),
    static_cast<int32_t>(dK.stride(0)), 0, 0,
    static_cast<int32_t>(dV.stride(0)), 0, 0,
    static_cast<int32_t>(query_scale.stride(0)),
    static_cast<int32_t>(query_scale.stride(1)),
    static_cast<int32_t>(key_scale.stride(0)),
    static_cast<int32_t>(key_scale.stride(1)),
  };

  if (tensor_layout == TensorLayout::kNHD)
  {
    params.seq_len = static_cast<int32_t>(query.size(1));
    params.num_heads = static_cast<int32_t>(query.size(2));
    CHECK_SHAPE(key, params.batch_size, params.seq_len, params.num_heads, params.head_dim);
    CHECK_SHAPE(value, params.batch_size, params.seq_len, params.num_heads, params.head_dim);
    CHECK_SHAPE(dO, params.batch_size, params.seq_len, params.num_heads, params.head_dim);
    CHECK_SHAPE(dK, params.batch_size, params.seq_len, params.num_heads, params.head_dim);
    CHECK_SHAPE(dV, params.batch_size, params.seq_len, params.num_heads, params.head_dim);

    params.stride_seq_q = static_cast<int32_t>(query.stride(1));
    params.stride_seq_k = static_cast<int32_t>(key.stride(1));
    params.stride_seq_v = static_cast<int32_t>(value.stride(1));
    params.stride_seq_dO = static_cast<int32_t>(dO.stride(1));
    params.stride_seq_dK = static_cast<int32_t>(dK.stride(1));
    params.stride_seq_dV = static_cast<int32_t>(dV.stride(1));

    params.stride_h_q = static_cast<int32_t>(query.stride(2));
    params.stride_h_k = static_cast<int32_t>(key.stride(2));
    params.stride_h_v = static_cast<int32_t>(value.stride(2));
    params.stride_h_dO = static_cast<int32_t>(dO.stride(2));
    params.stride_h_dK = static_cast<int32_t>(dK.stride(2));
    params.stride_h_dV = static_cast<int32_t>(dV.stride(2));
  }
  else
  {
    params.num_heads = static_cast<int32_t>(query.size(1));
    params.seq_len = static_cast<int32_t>(query.size(2));
    CHECK_SHAPE(key, params.batch_size, params.num_heads, params.seq_len, params.head_dim);
    CHECK_SHAPE(value, params.batch_size, params.num_heads, params.seq_len, params.head_dim);
    CHECK_SHAPE(dO, params.batch_size, params.num_heads, params.seq_len, params.head_dim);
    CHECK_SHAPE(dK, params.batch_size, params.num_heads, params.seq_len, params.head_dim);
    CHECK_SHAPE(dV, params.batch_size, params.num_heads, params.seq_len, params.head_dim);

    params.stride_seq_q = static_cast<int32_t>(query.stride(2));
    params.stride_seq_k = static_cast<int32_t>(key.stride(2));
    params.stride_seq_v = static_cast<int32_t>(value.stride(2));
    params.stride_seq_dO = static_cast<int32_t>(dO.stride(2));
    params.stride_seq_dK = static_cast<int32_t>(dK.stride(2));
    params.stride_seq_dV = static_cast<int32_t>(dV.stride(2));

    params.stride_h_q = static_cast<int32_t>(query.stride(1));
    params.stride_h_k = static_cast<int32_t>(key.stride(1));
    params.stride_h_v = static_cast<int32_t>(value.stride(1));
    params.stride_h_dO = static_cast<int32_t>(dO.stride(1));
    params.stride_h_dK = static_cast<int32_t>(dK.stride(1));
    params.stride_h_dV = static_cast<int32_t>(dV.stride(1));
  }

  const int32_t q_scale_blocks = div_ceil_int(params.seq_len, 32);
  const int32_t k_scale_blocks = div_ceil_int(params.seq_len, 64);
  const int32_t q_summary_blocks = div_ceil_int(params.seq_len, 32);
  const int32_t k_summary_blocks = div_ceil_int(params.seq_len, 64);
  CHECK_SHAPE(delta, params.batch_size, params.num_heads, params.seq_len);
  const int32_t dq_accum_rows = div_ceil_int(params.seq_len, 32) * 32;
  CHECK_SHAPE(dQ_accum, params.batch_size, params.num_heads, dq_accum_rows, params.head_dim);
  if (smooth_k)
  {
    CHECK_SHAPE(dS_sum, params.batch_size, params.num_heads, dq_accum_rows);
  }
  CHECK_SHAPE(dO_int8, params.batch_size, params.num_heads, params.seq_len, params.head_dim);
  CHECK_SHAPE(dO_scale, params.batch_size, params.num_heads, q_summary_blocks, params.head_dim / 16);
  CHECK_SHAPE(dS_q_factors, params.batch_size, params.num_heads, q_summary_blocks, 2);
  CHECK_SHAPE(dS_k_factors, params.batch_size, params.num_heads, k_summary_blocks);
  CHECK_SHAPE(query_scale, params.batch_size, params.num_heads, q_scale_blocks);
  CHECK_SHAPE(key_scale, params.batch_size, params.num_heads, k_scale_blocks);
  CHECK_SHAPE(lse, params.batch_size, params.num_heads, params.seq_len);
  return params;
}

template <int32_t HeadDim, int32_t BlockM, int32_t BlockN, int32_t NumWarps, int32_t QuantBlockQ, int32_t QuantBlockK>
void launch_mma(const Tensor &query,
                const Tensor &key,
                const Tensor &query_scale,
                const Tensor &key_scale,
                const Tensor &value,
                const Tensor &dO,
                const Tensor &lse,
                const Tensor &delta,
                const Tensor &dQ_accum,
                const Tensor &dS_sum,
                const Tensor &dO_int8,
                const Tensor &dO_scale,
                const Tensor &dS_q_factors,
                const Tensor &dS_k_factors,
                const Tensor &dK,
                const Tensor &dV,
                const Params &params,
                const double sm_scale,
                const bool smooth_k)
{
  static_assert(QuantBlockQ == 32 && QuantBlockK == 64, "Focused backward A/B uses the selected QBlock=32/KBlock=64 quantization format");
  constexpr int32_t kKernelWarps = NumWarps;
  using KernelTraits = BwdTileTraits<HeadDim, BlockM, BlockN, kKernelWarps>;
  const auto device_guard = make_device_guard(query);
  const auto stream = get_current_cuda_stream(query);
  const int32_t kv_blocks = div_ceil_int(params.seq_len, KernelTraits::kCtaN);

  const dim3 kv_grid(kv_blocks, params.num_heads, params.batch_size);

  constexpr int32_t smem_size = []() {
    if constexpr (HeadDim == 64)
    {
      return static_cast<int32_t>(sizeof(SharedStorage2DWarp<KernelTraits>));
    }
    else
    {
      return static_cast<int32_t>(sizeof(SharedStorageHD128<KernelTraits>));
    }
  }();
  const auto launch_kernel = [&](auto kernel) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<kv_grid, dim3(32, kKernelWarps), smem_size, stream>>>(
      reinterpret_cast<const int8_t*>(query.const_data_ptr()),
      reinterpret_cast<const int8_t*>(key.const_data_ptr()),
      reinterpret_cast<const float*>(query_scale.const_data_ptr()),
      reinterpret_cast<const float*>(key_scale.const_data_ptr()),
      reinterpret_cast<const half*>(value.const_data_ptr()),
      reinterpret_cast<const half*>(dO.const_data_ptr()),
      reinterpret_cast<const int8_t*>(dO_int8.const_data_ptr()),
      reinterpret_cast<const float*>(dO_scale.const_data_ptr()),
      reinterpret_cast<const float*>(dS_q_factors.const_data_ptr()),
      reinterpret_cast<const float*>(dS_k_factors.const_data_ptr()),
      reinterpret_cast<const float*>(lse.const_data_ptr()),
      reinterpret_cast<const float*>(delta.const_data_ptr()),
      reinterpret_cast<float*>(dQ_accum.mutable_data_ptr()),
      reinterpret_cast<float*>(dS_sum.mutable_data_ptr()),
      reinterpret_cast<half*>(dK.mutable_data_ptr()),
      reinterpret_cast<half*>(dV.mutable_data_ptr()),
      params,
      static_cast<float>(sm_scale));
  };
  if constexpr (HeadDim == 64)
  {
    if (smooth_k)
    {
      auto kernel = params.seq_len % BlockN == 0
        ? fused_mma_kernel_k128_8warp<64, BlockM, BlockN, kKernelWarps, QuantBlockQ, QuantBlockK, true, true>
        : fused_mma_kernel_k128_8warp<64, BlockM, BlockN, kKernelWarps, QuantBlockQ, QuantBlockK, false, true>;
      launch_kernel(kernel);
    }
    else
    {
      auto kernel = params.seq_len % BlockN == 0
        ? fused_mma_kernel_k128_8warp<64, BlockM, BlockN, kKernelWarps, QuantBlockQ, QuantBlockK, true, false>
        : fused_mma_kernel_k128_8warp<64, BlockM, BlockN, kKernelWarps, QuantBlockQ, QuantBlockK, false, false>;
      launch_kernel(kernel);
    }
  }
  else
  {
    if (smooth_k)
    {
      auto kernel = params.seq_len % BlockN == 0
        ? fused_mma_kernel_hd128_2d<128, BlockM, BlockN, kKernelWarps, QuantBlockQ, QuantBlockK, true, true>
        : fused_mma_kernel_hd128_2d<128, BlockM, BlockN, kKernelWarps, QuantBlockQ, QuantBlockK, false, true>;
      launch_kernel(kernel);
    }
    else
    {
      auto kernel = params.seq_len % BlockN == 0
        ? fused_mma_kernel_hd128_2d<128, BlockM, BlockN, kKernelWarps, QuantBlockQ, QuantBlockK, true, false>
        : fused_mma_kernel_hd128_2d<128, BlockM, BlockN, kKernelWarps, QuantBlockQ, QuantBlockK, false, false>;
      launch_kernel(kernel);
    }
  }
}

} // namespace sageattention::qattn_cutlass_bwd

#include "generated/qk_int8_sv_f16_accum_f32_attn_bwd_cutlass_dispatch.cuh"

#ifndef SAGEATTENTION_QATTN_CUTLASS_BWD_INSTANTIATION
void qk_int8_sv_f16_accum_f32_attn_bwd_cutlass(const Tensor &query,
                                               const Tensor &key,
                                               const Tensor &query_scale,
                                               const Tensor &key_scale,
                                               const Tensor &value,
                                               const Tensor &dO,
                                               const Tensor &lse,
                                               const Tensor &delta,
                                               const Tensor &dQ_accum,
                                               const Tensor &dS_sum,
                                               const Tensor &dO_int8,
                                               const Tensor &dO_scale,
                                               const Tensor &dS_q_factors,
                                               const Tensor &dS_k_factors,
                                               const Tensor &dK,
                                               const Tensor &dV,
                                               const int64_t tensor_layout,
                                               const double sm_scale,
                                               const int64_t blk_q,
                                               const int64_t blk_k,
                                               const int64_t bwd_block_m,
                                               const int64_t bwd_block_n,
                                               const bool smooth_k)
{
  namespace bwd = sageattention::qattn_cutlass_bwd;

  const auto layout = bwd::parse_tensor_layout(tensor_layout);
  const auto params = bwd::prepare_params(query, key, query_scale, key_scale, value, dO, lse, delta, dQ_accum, dS_sum, dO_int8, dO_scale, dS_q_factors, dS_k_factors, dK, dV, layout, blk_q, blk_k, bwd_block_m, bwd_block_n, smooth_k);
  bwd::launch_configured_mma(query, key, query_scale, key_scale, value, dO, lse, delta, dQ_accum, dS_sum, dO_int8, dO_scale, dS_q_factors, dS_k_factors, dK, dV, params, sm_scale, smooth_k);
}
#endif

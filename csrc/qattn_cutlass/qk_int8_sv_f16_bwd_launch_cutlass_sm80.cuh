#pragma once

#include "attn_cutlass_sm80.h"
#include "qk_int8_sv_f16_bwd_kernel_cutlass_sm80.cuh"
#include "../utils.cuh"

#include <cuda_fp16.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cstdint>
#include <optional>
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

inline Params prepare_params(const Tensor &query,
                                    const Tensor &key,
                                    const Tensor &query_scale,
                                    const Tensor &key_scale,
                                    const Tensor &value,
                                    const Tensor &output,
                                    const Tensor &grad_output,
                                    const Tensor &lse,
                                    const Tensor &grad_query,
                                    const Tensor &grad_key,
                                    const Tensor &grad_value,
                                    const TensorLayout tensor_layout,
                                    const int64_t blk_q,
                                    const int64_t blk_k,
                                    const int64_t bwd_block_m,
                                    const int64_t bwd_block_n)
{
  check_int8_tensor(query, "query");
  check_int8_tensor(key, "key");
  check_scale_tensor(query_scale, "query_scale");
  check_scale_tensor(key_scale, "key_scale");
  check_half_tensor(value, "value");
  check_half_tensor(output, "output");
  check_half_tensor(grad_output, "grad_output");
  check_half_tensor(grad_query, "grad_query");
  check_half_tensor(grad_key, "grad_key");
  check_half_tensor(grad_value, "grad_value");
  CHECK_CUDA(lse);
  CHECK_CONTIGUOUS(lse);
  CHECK_DTYPE(lse, torch::headeronly::ScalarType::Float);
  CHECK_DIMS(lse, 3);

  STD_TORCH_CHECK(blk_q > 0 && blk_k > 0, "blk_q and blk_k must be positive");
  STD_TORCH_CHECK(bwd_block_m == 64 && (bwd_block_n == 64 || bwd_block_n == 128),
                  "Focused CUTLASS qattn backward requires a 64x64 or 64x128 CTA");
  STD_TORCH_CHECK((blk_q == 32 || blk_q == 128) && (blk_k == 64 || blk_k == 128),
                  "Focused CUTLASS qattn backward requires QBlock=32 or 128 and KBlock=64 or 128");

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
    static_cast<int32_t>(output.stride(0)), 0, 0,
    static_cast<int32_t>(grad_output.stride(0)), 0, 0,
    static_cast<int32_t>(grad_query.stride(0)), 0, 0,
    static_cast<int32_t>(grad_key.stride(0)), 0, 0,
    static_cast<int32_t>(grad_value.stride(0)), 0, 0,
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
    CHECK_SHAPE(output, params.batch_size, params.seq_len, params.num_heads, params.head_dim);
    CHECK_SHAPE(grad_output, params.batch_size, params.seq_len, params.num_heads, params.head_dim);
    CHECK_SHAPE(grad_query, params.batch_size, params.seq_len, params.num_heads, params.head_dim);
    CHECK_SHAPE(grad_key, params.batch_size, params.seq_len, params.num_heads, params.head_dim);
    CHECK_SHAPE(grad_value, params.batch_size, params.seq_len, params.num_heads, params.head_dim);

    params.stride_seq_q = static_cast<int32_t>(query.stride(1));
    params.stride_seq_k = static_cast<int32_t>(key.stride(1));
    params.stride_seq_v = static_cast<int32_t>(value.stride(1));
    params.stride_seq_o = static_cast<int32_t>(output.stride(1));
    params.stride_seq_do = static_cast<int32_t>(grad_output.stride(1));
    params.stride_seq_dq = static_cast<int32_t>(grad_query.stride(1));
    params.stride_seq_dk = static_cast<int32_t>(grad_key.stride(1));
    params.stride_seq_dv = static_cast<int32_t>(grad_value.stride(1));

    params.stride_h_q = static_cast<int32_t>(query.stride(2));
    params.stride_h_k = static_cast<int32_t>(key.stride(2));
    params.stride_h_v = static_cast<int32_t>(value.stride(2));
    params.stride_h_o = static_cast<int32_t>(output.stride(2));
    params.stride_h_do = static_cast<int32_t>(grad_output.stride(2));
    params.stride_h_dq = static_cast<int32_t>(grad_query.stride(2));
    params.stride_h_dk = static_cast<int32_t>(grad_key.stride(2));
    params.stride_h_dv = static_cast<int32_t>(grad_value.stride(2));
  }
  else
  {
    params.num_heads = static_cast<int32_t>(query.size(1));
    params.seq_len = static_cast<int32_t>(query.size(2));
    CHECK_SHAPE(key, params.batch_size, params.num_heads, params.seq_len, params.head_dim);
    CHECK_SHAPE(value, params.batch_size, params.num_heads, params.seq_len, params.head_dim);
    CHECK_SHAPE(output, params.batch_size, params.num_heads, params.seq_len, params.head_dim);
    CHECK_SHAPE(grad_output, params.batch_size, params.num_heads, params.seq_len, params.head_dim);
    CHECK_SHAPE(grad_query, params.batch_size, params.num_heads, params.seq_len, params.head_dim);
    CHECK_SHAPE(grad_key, params.batch_size, params.num_heads, params.seq_len, params.head_dim);
    CHECK_SHAPE(grad_value, params.batch_size, params.num_heads, params.seq_len, params.head_dim);

    params.stride_seq_q = static_cast<int32_t>(query.stride(2));
    params.stride_seq_k = static_cast<int32_t>(key.stride(2));
    params.stride_seq_v = static_cast<int32_t>(value.stride(2));
    params.stride_seq_o = static_cast<int32_t>(output.stride(2));
    params.stride_seq_do = static_cast<int32_t>(grad_output.stride(2));
    params.stride_seq_dq = static_cast<int32_t>(grad_query.stride(2));
    params.stride_seq_dk = static_cast<int32_t>(grad_key.stride(2));
    params.stride_seq_dv = static_cast<int32_t>(grad_value.stride(2));

    params.stride_h_q = static_cast<int32_t>(query.stride(1));
    params.stride_h_k = static_cast<int32_t>(key.stride(1));
    params.stride_h_v = static_cast<int32_t>(value.stride(1));
    params.stride_h_o = static_cast<int32_t>(output.stride(1));
    params.stride_h_do = static_cast<int32_t>(grad_output.stride(1));
    params.stride_h_dq = static_cast<int32_t>(grad_query.stride(1));
    params.stride_h_dk = static_cast<int32_t>(grad_key.stride(1));
    params.stride_h_dv = static_cast<int32_t>(grad_value.stride(1));
  }

  const int32_t q_scale_blocks = div_ceil_int(params.seq_len, static_cast<int32_t>(blk_q));
  const int32_t k_scale_blocks = div_ceil_int(params.seq_len, static_cast<int32_t>(blk_k));
  CHECK_SHAPE(query_scale, params.batch_size, params.num_heads, q_scale_blocks);
  CHECK_SHAPE(key_scale, params.batch_size, params.num_heads, k_scale_blocks);
  CHECK_SHAPE(lse, params.batch_size, params.num_heads, params.seq_len);
  return params;
}

template <int32_t HeadDim,
          int32_t BlockM,
          int32_t BlockN,
          int32_t NumWarps,
          int32_t QuantBlockQ,
          int32_t QuantBlockK>
void launch_mma(const Tensor &query,
                    const Tensor &key,
                    const Tensor &query_scale,
                    const Tensor &key_scale,
                    const Tensor &value,
                    const Tensor &output,
                    const Tensor &grad_output,
                    const Tensor &lse,
                    const Tensor &grad_query,
                    const Tensor &grad_key,
                    const Tensor &grad_value,
                    const Params &params,
                    const double sm_scale)
{
  static_assert(
    HeadDim == 64 && BlockM == 64 &&
      ((BlockN == 64 && NumWarps == 4) || (BlockN == 128 && NumWarps == 8)),
    "Only the focused head-64 64x64 and 64x128 backward matmul configurations are built");
  static_assert((QuantBlockQ == 32 || QuantBlockQ == 128) && QuantBlockK == BlockN,
                "Focused backward quantization specializations require QBlock=32 or 128 and KBlock=CTA N");
  using MicroTraits = BwdTileTraits<64>;
  constexpr int32_t kKernelWarps = 2 * NumWarps;
  using KernelTraits = BwdTileTraits<64, BlockM, BlockN, kKernelWarps>;
  const auto device_guard = make_device_guard(query);
  const auto stream = get_current_cuda_stream(query);
  const int32_t q_blocks = div_ceil_int(params.seq_len, MicroTraits::kBlockM);
  const int32_t pair_blocks = div_ceil_int(params.seq_len, 2 * MicroTraits::kBlockM);
  const int32_t kv_blocks = div_ceil_int(params.seq_len, KernelTraits::kCtaN);

  constexpr int32_t dim_blocks = 64 / MicroTraits::kBlockK;
  const auto delta = torch::stable::new_empty(lse, {params.batch_size, params.num_heads, params.seq_len}, std::make_optional(torch::headeronly::ScalarType::Float));
  const auto dq_accum = torch::stable::new_empty(lse, {params.batch_size, params.num_heads, params.seq_len, HeadDim}, std::make_optional(torch::headeronly::ScalarType::Float));
  const auto do_int8 = torch::stable::new_empty(lse, {params.batch_size, params.num_heads, params.seq_len, HeadDim}, std::make_optional(torch::headeronly::ScalarType::Char));
  const auto do_scale = torch::stable::new_empty(lse, {params.batch_size, params.num_heads, pair_blocks, dim_blocks}, std::make_optional(torch::headeronly::ScalarType::Float));

  const dim3 q_grid(q_blocks, params.num_heads, params.batch_size);
  const dim3 pair_grid(pair_blocks, params.num_heads, params.batch_size);
  const dim3 kv_grid(kv_blocks, params.num_heads, params.batch_size);

  preprocess_delta_zero_dq_kernel<HeadDim><<<pair_grid, 128, 0, stream>>>(
    reinterpret_cast<const half*>(output.const_data_ptr()),
    reinterpret_cast<const half*>(grad_output.const_data_ptr()),
    reinterpret_cast<float*>(delta.mutable_data_ptr()),
    reinterpret_cast<float*>(dq_accum.mutable_data_ptr()),
    reinterpret_cast<int8_t*>(do_int8.mutable_data_ptr()),
    reinterpret_cast<float*>(do_scale.mutable_data_ptr()),
    params);

  auto kernel = params.seq_len % BlockN == 0
    ? fused_mma_kernel_2d_warp<64, BlockM, BlockN, kKernelWarps, QuantBlockQ, QuantBlockK, true>
    : fused_mma_kernel_2d_warp<64, BlockM, BlockN, kKernelWarps, QuantBlockQ, QuantBlockK, false>;
  constexpr int32_t smem_size = sizeof(SharedStorage2DWarp<KernelTraits>);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  kernel<<<kv_grid, dim3(32, kKernelWarps), smem_size, stream>>>(
    reinterpret_cast<const int8_t*>(query.const_data_ptr()),
    reinterpret_cast<const int8_t*>(key.const_data_ptr()),
    reinterpret_cast<const float*>(query_scale.const_data_ptr()),
    reinterpret_cast<const float*>(key_scale.const_data_ptr()),
    reinterpret_cast<const half*>(value.const_data_ptr()),
    reinterpret_cast<const half*>(grad_output.const_data_ptr()),
    reinterpret_cast<const int8_t*>(do_int8.const_data_ptr()),
    reinterpret_cast<const float*>(do_scale.const_data_ptr()),
    reinterpret_cast<const float*>(lse.const_data_ptr()),
    reinterpret_cast<const float*>(delta.const_data_ptr()),
    reinterpret_cast<float*>(dq_accum.mutable_data_ptr()),
    reinterpret_cast<half*>(grad_key.mutable_data_ptr()),
    reinterpret_cast<half*>(grad_value.mutable_data_ptr()),
    params,
    static_cast<float>(sm_scale));

  convert_dq_kernel<HeadDim><<<q_grid, 128, 0, stream>>>(
    reinterpret_cast<const float*>(dq_accum.const_data_ptr()),
    reinterpret_cast<half*>(grad_query.mutable_data_ptr()),
    params);
}

} // namespace sageattention::qattn_cutlass_bwd

#include "generated/qk_int8_sv_f16_accum_f32_attn_bwd_cutlass_dispatch.cuh"

#ifndef SAGEATTENTION_QATTN_CUTLASS_BWD_INSTANTIATION
void qk_int8_sv_f16_accum_f32_attn_bwd_cutlass(const Tensor &query,
                                               const Tensor &key,
                                               const Tensor &query_scale,
                                               const Tensor &key_scale,
                                               const Tensor &value,
                                               const Tensor &output,
                                               const Tensor &grad_output,
                                               const Tensor &lse,
                                               const Tensor &grad_query,
                                               const Tensor &grad_key,
                                               const Tensor &grad_value,
                                               const int64_t tensor_layout,
                                               const double sm_scale,
                                               const int64_t blk_q,
                                               const int64_t blk_k,
                                               const int64_t bwd_block_m,
                                               const int64_t bwd_block_n)
{
  namespace bwd = sageattention::qattn_cutlass_bwd;

  const auto layout = bwd::parse_tensor_layout(tensor_layout);
  const auto params = bwd::prepare_params(query, key, query_scale, key_scale, value, output, grad_output, lse, grad_query, grad_key, grad_value, layout, blk_q, blk_k, bwd_block_m, bwd_block_n);
  STD_TORCH_CHECK(params.head_dim == 64,
                  "Focused CUTLASS qattn backward currently supports head_dim 64 only");

  bwd::launch_configured_mma(query, key, query_scale, key_scale, value, output, grad_output, lse, grad_query, grad_key, grad_value, params, sm_scale);
}
#endif

#pragma once

#include "attn_cutlass_sm80.h"
#include "qk_int8_sv_f16_params_cutlass_sm80.cuh"
#include "../utils.cuh"

#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cstdint>
#include <optional>
#include <stdexcept>

namespace sageattention::qattn_cutlass {

enum class TensorLayout {
  kNHD,
  kHND,
};

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

inline void check_launch_tensors(const Tensor &query,
                                 const Tensor &key,
                                 const Tensor &value,
                                 const Tensor &output,
                                 const Tensor &query_scale,
                                 const Tensor &key_scale)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);

  CHECK_DTYPE(query, torch::headeronly::ScalarType::Char);
  CHECK_DTYPE(key, torch::headeronly::ScalarType::Char);
  CHECK_DTYPE(value, torch::headeronly::ScalarType::Half);
  CHECK_DTYPE(output, torch::headeronly::ScalarType::Half);
  CHECK_DTYPE(query_scale, torch::headeronly::ScalarType::Float);
  CHECK_DTYPE(key_scale, torch::headeronly::ScalarType::Float);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);
}

inline LaunchParams prepare_launch_params(const Tensor &query,
                                          const Tensor &key,
                                          const Tensor &value,
                                          const Tensor &output,
                                          const Tensor &query_scale,
                                          const Tensor &key_scale,
                                          const TensorLayout tensor_layout,
                                          const double sm_scale,
                                          const int64_t blk_q,
                                          const int64_t blk_k,
                                          const int64_t warp_q,
                                          const int64_t warp_k)
{
  check_launch_tensors(query, key, value, output, query_scale, key_scale);

  LaunchParams params = {
    query.const_data_ptr<int8_t>(),
    key.const_data_ptr<int8_t>(),
    reinterpret_cast<const half*>(value.const_data_ptr<c10::Half>()),
    reinterpret_cast<half*>(output.mutable_data_ptr<c10::Half>()),
    nullptr,
    query_scale.const_data_ptr<float>(),
    key_scale.const_data_ptr<float>(),
    static_cast<int32_t>(query.size(3)),
    static_cast<int32_t>(query.size(0)),
    0,
    0,
    0,
    0,
    0,
    static_cast<int32_t>(query.stride(0)),
    0,
    0,
    static_cast<int32_t>(key.stride(0)),
    0,
    0,
    static_cast<int32_t>(value.stride(0)),
    0,
    0,
    static_cast<int32_t>(output.stride(0)),
    0,
    0,
    static_cast<int32_t>(blk_q),
    static_cast<int32_t>(blk_k),
    static_cast<int32_t>(warp_q),
    static_cast<int32_t>(warp_k),
    static_cast<float>(sm_scale),
    get_current_cuda_stream(query),
  };

  if (tensor_layout == TensorLayout::kNHD)
  {
    params.qo_len = static_cast<int32_t>(query.size(1));
    params.kv_len = static_cast<int32_t>(key.size(1));
    params.num_qo_heads = static_cast<int32_t>(query.size(2));
    params.num_kv_heads = static_cast<int32_t>(key.size(2));
    CHECK_SHAPE(key, params.batch_size, params.kv_len, params.num_kv_heads, params.head_dim);
    CHECK_SHAPE(value, params.batch_size, params.kv_len, params.num_kv_heads, params.head_dim);
    CHECK_SHAPE(output, params.batch_size, params.qo_len, params.num_qo_heads, params.head_dim);

    params.stride_seq_q = static_cast<int32_t>(query.stride(1));
    params.stride_seq_k = static_cast<int32_t>(key.stride(1));
    params.stride_seq_v = static_cast<int32_t>(value.stride(1));
    params.stride_seq_o = static_cast<int32_t>(output.stride(1));

    params.stride_h_q = static_cast<int32_t>(query.stride(2));
    params.stride_h_k = static_cast<int32_t>(key.stride(2));
    params.stride_h_v = static_cast<int32_t>(value.stride(2));
    params.stride_h_o = static_cast<int32_t>(output.stride(2));
  }
  else
  {
    params.qo_len = static_cast<int32_t>(query.size(2));
    params.kv_len = static_cast<int32_t>(key.size(2));
    params.num_qo_heads = static_cast<int32_t>(query.size(1));
    params.num_kv_heads = static_cast<int32_t>(key.size(1));
    CHECK_SHAPE(key, params.batch_size, params.num_kv_heads, params.kv_len, params.head_dim);
    CHECK_SHAPE(value, params.batch_size, params.num_kv_heads, params.kv_len, params.head_dim);
    CHECK_SHAPE(output, params.batch_size, params.num_qo_heads, params.qo_len, params.head_dim);

    params.stride_seq_q = static_cast<int32_t>(query.stride(2));
    params.stride_seq_k = static_cast<int32_t>(key.stride(2));
    params.stride_seq_v = static_cast<int32_t>(value.stride(2));
    params.stride_seq_o = static_cast<int32_t>(output.stride(2));

    params.stride_h_q = static_cast<int32_t>(query.stride(1));
    params.stride_h_k = static_cast<int32_t>(key.stride(1));
    params.stride_h_v = static_cast<int32_t>(value.stride(1));
    params.stride_h_o = static_cast<int32_t>(output.stride(1));
  }

  STD_TORCH_CHECK(params.head_dim == 64 || params.head_dim == 128, "CUTLASS qattn currently supports head_dim 64 or 128");
  STD_TORCH_CHECK(params.qo_len > 0 && params.kv_len > 0, "CUTLASS qattn requires positive sequence lengths");
  STD_TORCH_CHECK(params.num_qo_heads % params.num_kv_heads == 0, "num_qo_heads must be divisible by num_kv_heads");
  params.num_kv_groups = params.num_qo_heads / params.num_kv_heads;

  constexpr int32_t kQuantBlockQ = 32;
  constexpr int32_t kQuantBlockK = 64;
  STD_TORCH_CHECK(params.blk_q % kQuantBlockQ == 0 && kQuantBlockQ % params.warp_q == 0, "CUTLASS forward requires Q32 scales aligned to CTA and warp rows");
  STD_TORCH_CHECK(kQuantBlockK % params.blk_k == 0 && params.warp_k == params.blk_k, "CUTLASS forward requires K64 scales aligned to CTA and warp columns");
  CHECK_SHAPE(query_scale, params.batch_size, params.num_qo_heads, ceil_div_int(params.qo_len, kQuantBlockQ));
  CHECK_SHAPE(key_scale, params.batch_size, params.num_kv_heads, ceil_div_int(params.kv_len, kQuantBlockK));

  return params;
}

} // namespace sageattention::qattn_cutlass

#include "generated/qk_int8_sv_f16_accum_f32_attn_cutlass_dispatch.cuh"

Tensor qk_int8_sv_f16_accum_f32_attn_cutlass(const Tensor &query,
                                             const Tensor &key,
                                             const Tensor &value,
                                             const Tensor &output,
                                             const Tensor &query_scale,
                                             const Tensor &key_scale,
                                             const int64_t tensor_layout,
                                             const double sm_scale,
                                             const int64_t blk_q,
                                             const int64_t blk_k,
                                             const int64_t warp_q,
                                             const int64_t warp_k,
                                             const bool return_lse)
{
  namespace qattn_cutlass = sageattention::qattn_cutlass;

  const auto layout = qattn_cutlass::parse_tensor_layout(tensor_layout);
  const auto device_guard = make_device_guard(query);
  auto launch_params = qattn_cutlass::prepare_launch_params(
    query,
    key,
    value,
    output,
    query_scale,
    key_scale,
    layout,
    sm_scale,
    blk_q,
    blk_k,
    warp_q,
    warp_k);

  auto lse = return_lse
    ? torch::stable::new_empty(query, {launch_params.batch_size, launch_params.num_qo_heads, launch_params.qo_len}, std::make_optional(torch::headeronly::ScalarType::Float))
    : torch::stable::new_empty(query, {0}, std::make_optional(torch::headeronly::ScalarType::Float));
  launch_params.lse = return_lse ? lse.mutable_data_ptr<float>() : nullptr;

  qattn_cutlass::launch_configured_kernel(launch_params, return_lse);
  return lse;
}

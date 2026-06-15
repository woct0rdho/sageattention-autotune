/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "qk_int8_sv_f8_kernel_sm89.cuh"
#include "../dispatch_utils.h"
#include "../utils.cuh"

#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <algorithm>
#include <stdexcept>

using torch::stable::Tensor;

template <typename T>
const T *const_ptr(const Tensor &tensor)
{
  return reinterpret_cast<const T*>(tensor.const_data_ptr());
}

template <typename T>
T *mutable_ptr(const Tensor &tensor)
{
  return reinterpret_cast<T*>(tensor.mutable_data_ptr());
}

enum class Sm89TensorLayout {
  kNHD,
  kHND,
};

inline Sm89TensorLayout sm89_parse_tensor_layout(const int64_t tensor_layout)
{
  if (tensor_layout == 0)
  {
    return Sm89TensorLayout::kNHD;
  }
  if (tensor_layout == 1)
  {
    return Sm89TensorLayout::kHND;
  }
  throw std::invalid_argument("tensor_layout must be 0 (NHD) or 1 (HND)");
}

struct Sm89QkLaunchParams {
  int head_dim;
  int batch_size;
  int qo_len;
  int kv_len;       // real (unpadded) kv length
  int num_qo_heads;
  int num_kv_heads;
  int num_kv_groups;
  int stride_bz_q;
  int stride_seq_q;
  int stride_h_q;
  int stride_bz_k;
  int stride_seq_k;
  int stride_h_k;
  int stride_bz_v;  // V is transposed: [batch, num_kv_heads, head_dim, kv_len_pad]
  int stride_h_v;
  int stride_d_v;
  int stride_bz_o;
  int stride_seq_o;
  int stride_h_o;
  Tensor lse;
};

inline Sm89QkLaunchParams prepare_sm89_qk_launch_params(const Tensor &query,
                                                        const Tensor &key,
                                                        const Tensor &value,
                                                        const Tensor &output,
                                                        const Tensor &query_scale,
                                                        const Tensor &key_scale,
                                                        const Tensor &value_scale,
                                                        const Sm89TensorLayout tensor_layout,
                                                        const bool return_lse)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);
  CHECK_CUDA(value_scale);

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);
  CHECK_CONTIGUOUS(value_scale);

  CHECK_DTYPE(query, torch::headeronly::ScalarType::Char);
  CHECK_DTYPE(key, torch::headeronly::ScalarType::Char);
  CHECK_DTYPE(value, torch::headeronly::ScalarType::Char); // fp8 e4m3 stored as int8
  CHECK_DTYPE(query_scale, torch::headeronly::ScalarType::Float);
  CHECK_DTYPE(key_scale, torch::headeronly::ScalarType::Float);
  CHECK_DTYPE(value_scale, torch::headeronly::ScalarType::Float);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);
  CHECK_DIMS(value_scale, 3);

  Sm89QkLaunchParams params = {};
  params.head_dim = static_cast<int>(query.size(3));
  params.batch_size = static_cast<int>(query.size(0));
  params.stride_bz_q = static_cast<int>(query.stride(0));
  params.stride_bz_k = static_cast<int>(key.stride(0));
  params.stride_bz_o = static_cast<int>(output.stride(0));

  if (tensor_layout == Sm89TensorLayout::kNHD)
  {
    params.qo_len = static_cast<int>(query.size(1));
    params.kv_len = static_cast<int>(key.size(1));
    params.num_qo_heads = static_cast<int>(query.size(2));
    params.num_kv_heads = static_cast<int>(key.size(2));

    params.stride_seq_q = static_cast<int>(query.stride(1));
    params.stride_seq_k = static_cast<int>(key.stride(1));
    params.stride_seq_o = static_cast<int>(output.stride(1));

    params.stride_h_q = static_cast<int>(query.stride(2));
    params.stride_h_k = static_cast<int>(key.stride(2));
    params.stride_h_o = static_cast<int>(output.stride(2));
  }
  else
  {
    params.qo_len = static_cast<int>(query.size(2));
    params.kv_len = static_cast<int>(key.size(2));
    params.num_qo_heads = static_cast<int>(query.size(1));
    params.num_kv_heads = static_cast<int>(key.size(1));

    params.stride_seq_q = static_cast<int>(query.stride(2));
    params.stride_seq_k = static_cast<int>(key.stride(2));
    params.stride_seq_o = static_cast<int>(output.stride(2));

    params.stride_h_q = static_cast<int>(query.stride(1));
    params.stride_h_k = static_cast<int>(key.stride(1));
    params.stride_h_o = static_cast<int>(output.stride(1));
  }

  // V is always pre-transposed to [batch, num_kv_heads, head_dim, kv_len_pad]
  params.stride_bz_v = static_cast<int>(value.stride(0));
  params.stride_h_v = static_cast<int>(value.stride(1));
  params.stride_d_v = static_cast<int>(value.stride(2));

  if (params.num_qo_heads % params.num_kv_heads != 0) {
    STD_TORCH_CHECK(false, "num_qo_heads (", params.num_qo_heads, ") must be divisible by num_kv_heads (", params.num_kv_heads, ")");
  }
  params.num_kv_groups = params.num_qo_heads / params.num_kv_heads;

  CHECK_SHAPE(value, params.batch_size, params.num_kv_heads, params.head_dim, static_cast<int64_t>(value.size(3)));
  CHECK_SHAPE(value_scale, params.batch_size, params.num_kv_heads, params.head_dim);

  params.lse = return_lse
    ? torch::stable::new_empty(query, {params.batch_size, params.num_qo_heads, params.qo_len}, std::make_optional(torch::headeronly::ScalarType::Float))
    : torch::stable::new_empty(query, {0}, std::make_optional(torch::headeronly::ScalarType::Float));

  return params;
}

struct Sm89QkLaunchContext {
  const Tensor &query;
  const Tensor &key;
  const Tensor &value;
  const Tensor &output;
  const Tensor &query_scale;
  const Tensor &key_scale;
  const Tensor &value_scale;
  const Sm89QkLaunchParams &params;
  const void *value_mean;
  double sm_scale;
  int64_t blk_q;
  int64_t blk_k;
  int64_t warp_q;
  int64_t warp_k;
  bool is_causal;
  bool return_lse;
};

template <int HeadDim, bool IsCausal, bool ReturnLse, typename DTypeOut, int CtaQ, int CtaK, int WarpQ, int WarpK,
          typename DTypeSVAccum, bool UseInstBuffer, ComputeUnit DenominatorAccumUnit, bool FuseVScale, bool FuseVMean, bool UsePvFp16Accu>
void launch_sm89_qk_kernel(const Sm89QkLaunchContext &ctx)
{
  CHECK_SHAPE(ctx.query_scale, ctx.params.batch_size, ctx.params.num_qo_heads, div_ceil(ctx.params.qo_len, CtaQ) * (CtaQ / WarpQ) * 8);
  CHECK_SHAPE(ctx.key_scale, ctx.params.batch_size, ctx.params.num_kv_heads, div_ceil(ctx.params.kv_len, CtaK) * (CtaK / WarpK) * 4);

  constexpr MaskMode mask_mode = IsCausal ? MaskMode::kCausal : MaskMode::kNone;
  // smem: Q (CtaQ*hd int8) + K (CtaK*hd int8) + V (CtaK*hd fp8) vs O (CtaQ*hd half)
  const size_t smem_max = std::max(
      static_cast<size_t>(CtaQ) * HeadDim * sizeof(int8_t) + static_cast<size_t>(CtaK) * HeadDim * sizeof(int8_t) + static_cast<size_t>(CtaK) * HeadDim * sizeof(int8_t),
      static_cast<size_t>(CtaQ) * HeadDim * sizeof(half));

  auto kernel_func = qk_int8_sv_f8_attn_kernel<CtaQ, CtaK, WarpQ, WarpK, HeadDim, DTypeSVAccum, UseInstBuffer, DTypeOut, DenominatorAccumUnit, mask_mode, ReturnLse, FuseVScale, FuseVMean, UsePvFp16Accu>;
  cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

  const dim3 grid(div_ceil(ctx.params.qo_len, CtaQ), ctx.params.num_qo_heads, ctx.params.batch_size);
  const dim3 block(32, (CtaQ / WarpQ) * (CtaK / WarpK));
  const auto device_guard = make_device_guard(ctx.query);
  const auto stream = get_current_cuda_stream(ctx.query);

  kernel_func<<<grid, block, smem_max, stream>>>(
    const_ptr<int8_t>(ctx.query),
    const_ptr<int8_t>(ctx.key),
    const_ptr<int8_t>(ctx.value),
    mutable_ptr<DTypeOut>(ctx.output),
    ReturnLse ? mutable_ptr<float>(ctx.params.lse) : nullptr,
    const_ptr<float>(ctx.query_scale),
    const_ptr<float>(ctx.key_scale),
    const_ptr<float>(ctx.value_scale),
    static_cast<const float*>(ctx.value_mean),
    ctx.params.qo_len,
    ctx.params.kv_len,
    ctx.params.num_kv_groups,
    ctx.params.stride_bz_q, ctx.params.stride_seq_q, ctx.params.stride_h_q,
    ctx.params.stride_bz_k, ctx.params.stride_seq_k, ctx.params.stride_h_k,
    ctx.params.stride_bz_v, ctx.params.stride_h_v, ctx.params.stride_d_v,
    ctx.params.stride_bz_o, ctx.params.stride_seq_o, ctx.params.stride_h_o,
    ctx.sm_scale);
}

template <int HeadDim, bool IsCausal, bool ReturnLse, typename DTypeOut,
          typename DTypeSVAccum, bool UseInstBuffer, ComputeUnit DenominatorAccumUnit, bool FuseVScale, bool FuseVMean, bool UsePvFp16Accu>
void launch_configured_sm89_qk_kernel(const Sm89QkLaunchContext &ctx)
{
  // Only num_warps_k == 1 (warp_k == blk_k) configs are supported: the kernel
  // does not reduce the m/d/o state across k-warps.
  if (ctx.blk_q == 128 && ctx.blk_k == 64 && ctx.warp_q == 32 && ctx.warp_k == 64)
  {
    launch_sm89_qk_kernel<HeadDim, IsCausal, ReturnLse, DTypeOut, 128, 64, 32, 64, DTypeSVAccum, UseInstBuffer, DenominatorAccumUnit, FuseVScale, FuseVMean, UsePvFp16Accu>(ctx);
  }
  else if (ctx.blk_q == 64 && ctx.blk_k == 64 && ctx.warp_q == 32 && ctx.warp_k == 64)
  {
    launch_sm89_qk_kernel<HeadDim, IsCausal, ReturnLse, DTypeOut, 64, 64, 32, 64, DTypeSVAccum, UseInstBuffer, DenominatorAccumUnit, FuseVScale, FuseVMean, UsePvFp16Accu>(ctx);
  }
  else if (ctx.blk_q == 128 && ctx.blk_k == 64 && ctx.warp_q == 16 && ctx.warp_k == 64)
  {
    launch_sm89_qk_kernel<HeadDim, IsCausal, ReturnLse, DTypeOut, 128, 64, 16, 64, DTypeSVAccum, UseInstBuffer, DenominatorAccumUnit, FuseVScale, FuseVMean, UsePvFp16Accu>(ctx);
  }
  else
  {
    throw std::invalid_argument("Unsupported blk_q/blk_k/warp_q/warp_k configuration for sm89");
  }
}

template <typename DTypeSVAccum, bool UseInstBuffer, ComputeUnit DenominatorAccumUnit, bool FuseVScale, bool FuseVMean, bool UsePvFp16Accu>
Tensor run_sm89_qk_attn(const Tensor &query,
                        const Tensor &key,
                        const Tensor &value,
                        const Tensor &output,
                        const Tensor &query_scale,
                        const Tensor &key_scale,
                        const Tensor &value_scale,
                        const Tensor *value_mean,
                        const int64_t tensor_layout,
                        const bool is_causal,
                        const double sm_scale,
                        const int64_t blk_q,
                        const int64_t blk_k,
                        const int64_t warp_q,
                        const int64_t warp_k,
                        const bool return_lse)
{
  const auto layout = sm89_parse_tensor_layout(tensor_layout);
  const auto params = prepare_sm89_qk_launch_params(query, key, value, output, query_scale, key_scale, value_scale, layout, return_lse);
  const void *value_mean_ptr = nullptr;

  if constexpr (FuseVMean)
  {
    STD_TORCH_CHECK(value_mean != nullptr, "value_mean is required when fusing V mean");
    const Tensor &value_mean_tensor = *value_mean;
    CHECK_CUDA(value_mean_tensor);
    CHECK_CONTIGUOUS(value_mean_tensor);
    CHECK_DIMS(value_mean_tensor, 3);
    CHECK_DTYPE(value_mean_tensor, torch::headeronly::ScalarType::Float);
    CHECK_SHAPE(value_mean_tensor, params.batch_size, params.num_kv_heads, params.head_dim);
    value_mean_ptr = value_mean_tensor.const_data_ptr();
  }

  const Sm89QkLaunchContext ctx{
    query,
    key,
    value,
    output,
    query_scale,
    key_scale,
    value_scale,
    params,
    value_mean_ptr,
    sm_scale,
    blk_q,
    blk_k,
    warp_q,
    warp_k,
    is_causal,
    return_lse,
  };

  sageattention::dispatch::fp16_dtype(ctx.output.scalar_type(), [&]<typename DTypeOut>() {
    sageattention::dispatch::head_dim(ctx.params.head_dim, [&]<int HeadDim>() {
      sageattention::dispatch::boolean(ctx.is_causal, [&]<bool IsCausal>() {
        // ReturnLse is disabled for compilation speed, matching the sm80 path
        launch_configured_sm89_qk_kernel<HeadDim, IsCausal, false, DTypeOut, DTypeSVAccum, UseInstBuffer, DenominatorAccumUnit, FuseVScale, FuseVMean, UsePvFp16Accu>(ctx);
      });
    });
  });
  return params.lse;
}

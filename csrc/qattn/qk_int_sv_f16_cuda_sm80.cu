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

#include "qk_int_sv_f16_kernel_sm80.cuh"

#define LAUNCH_QK_INT_SV_F16_KERNEL(CTA_Q_VALUE, CTA_K_VALUE, WARP_Q_VALUE, WARP_K_VALUE, DTYPE_SV_ACCUM, USE_INST_BUFFER, DENOMINATOR_ACCUM_UNIT, FUSE_V_MEAN, VALUE_MEAN_PTR) \
  do { \
    constexpr int CTA_Q_LOCAL = (CTA_Q_VALUE); \
    constexpr int CTA_K_LOCAL = (CTA_K_VALUE); \
    constexpr int WARP_Q_LOCAL = (WARP_Q_VALUE); \
    constexpr int WARP_K_LOCAL = (WARP_K_VALUE); \
    constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone; \
    if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp)) \
    { \
      CHECK_SHAPE(query_scale, params.batch_size, params.num_qo_heads, div_ceil(params.qo_len, CTA_Q_LOCAL) * (CTA_Q_LOCAL / WARP_Q_LOCAL)); \
      CHECK_SHAPE(key_scale, params.batch_size, params.num_kv_heads, div_ceil(params.kv_len, CTA_K_LOCAL) * (CTA_K_LOCAL / WARP_K_LOCAL)); \
    } \
    else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread)) \
    { \
      CHECK_SHAPE(query_scale, params.batch_size, params.num_qo_heads, div_ceil(params.qo_len, CTA_Q_LOCAL) * (CTA_Q_LOCAL / WARP_Q_LOCAL) * 8); \
      CHECK_SHAPE(key_scale, params.batch_size, params.num_kv_heads, div_ceil(params.kv_len, CTA_K_LOCAL) * (CTA_K_LOCAL / WARP_K_LOCAL) * 4); \
    } \
    else \
    { \
      static_assert(QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp) || QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread), "Unsupported quantization granularity"); \
    } \
    size_t smem_max = std::max(CTA_Q_LOCAL * HEAD_DIM * sizeof(int8_t) + CTA_K_LOCAL * HEAD_DIM * sizeof(int8_t) + CTA_K_LOCAL * HEAD_DIM * sizeof(half), CTA_Q_LOCAL * HEAD_DIM * sizeof(half)); \
    auto kernel_func = qk_int_sv_f16_attn_kernel<CTA_Q_LOCAL, CTA_K_LOCAL, WARP_Q_LOCAL, WARP_K_LOCAL, HEAD_DIM, static_cast<QuantGranularity>(QK_QUANT_GRAN), static_cast<QuantGranularity>(QK_QUANT_GRAN), DTYPE_SV_ACCUM, USE_INST_BUFFER, DTypeOut, DENOMINATOR_ACCUM_UNIT, \
                                                        mask_mode, RETURN_LSE, FUSE_V_MEAN>; \
    cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max); \
    dim3 grid(div_ceil(params.qo_len, CTA_Q_LOCAL), params.num_qo_heads, params.batch_size); \
    dim3 block(32, (CTA_Q_LOCAL / WARP_Q_LOCAL) * (CTA_K_LOCAL / WARP_K_LOCAL)); \
    kernel_func<<<grid, block, smem_max>>>( \
      const_ptr<int8_t>(query), \
      const_ptr<int8_t>(key), \
      const_ptr<half>(value), \
      mutable_ptr<DTypeOut>(output), \
      (RETURN_LSE) ? mutable_ptr<float>(params.lse) : nullptr, \
      const_ptr<float>(query_scale), \
      const_ptr<float>(key_scale), \
      VALUE_MEAN_PTR, \
      params.qo_len, \
      params.kv_len, \
      params.num_kv_groups, \
      params.stride_bz_q, params.stride_seq_q, params.stride_h_q, \
      params.stride_bz_k, params.stride_seq_k, params.stride_h_k, \
      params.stride_bz_v, params.stride_seq_v, params.stride_h_v, \
      params.stride_bz_o, params.stride_seq_o, params.stride_h_o, \
      sm_scale); \
  } while (0)

#define LAUNCH_QK_INT_SV_F16_CONFIGURED(BLK_Q_VALUE, BLK_K_VALUE, WARP_Q_VALUE, WARP_K_VALUE, DTYPE_SV_ACCUM, USE_INST_BUFFER, DENOMINATOR_ACCUM_UNIT, FUSE_V_MEAN, VALUE_MEAN_PTR) \
  do { \
    if ((BLK_Q_VALUE) == 128 && (BLK_K_VALUE) == 64 && (WARP_Q_VALUE) == 32 && (WARP_K_VALUE) == 64) \
    { \
      LAUNCH_QK_INT_SV_F16_KERNEL(128, 64, 32, 64, DTYPE_SV_ACCUM, USE_INST_BUFFER, DENOMINATOR_ACCUM_UNIT, FUSE_V_MEAN, VALUE_MEAN_PTR); \
    } \
    else if ((BLK_Q_VALUE) == 128 && (BLK_K_VALUE) == 32 && (WARP_Q_VALUE) == 32 && (WARP_K_VALUE) == 32) \
    { \
      if constexpr (!IS_CAUSAL) \
      { \
        LAUNCH_QK_INT_SV_F16_KERNEL(128, 32, 32, 32, DTYPE_SV_ACCUM, USE_INST_BUFFER, DENOMINATOR_ACCUM_UNIT, FUSE_V_MEAN, VALUE_MEAN_PTR); \
      } \
      else \
      { \
        throw std::invalid_argument("blk_q=128 blk_k=32 is not supported for causal attention"); \
      } \
    } \
    else if ((BLK_Q_VALUE) == 64 && (BLK_K_VALUE) == 64 && (WARP_Q_VALUE) == 32 && (WARP_K_VALUE) == 64) \
    { \
      LAUNCH_QK_INT_SV_F16_KERNEL(64, 64, 32, 64, DTYPE_SV_ACCUM, USE_INST_BUFFER, DENOMINATOR_ACCUM_UNIT, FUSE_V_MEAN, VALUE_MEAN_PTR); \
    } \
    else if ((BLK_Q_VALUE) == 128 && (BLK_K_VALUE) == 64 && (WARP_Q_VALUE) == 16 && (WARP_K_VALUE) == 64) \
    { \
      LAUNCH_QK_INT_SV_F16_KERNEL(128, 64, 16, 64, DTYPE_SV_ACCUM, USE_INST_BUFFER, DENOMINATOR_ACCUM_UNIT, FUSE_V_MEAN, VALUE_MEAN_PTR); \
    } \
    else \
    { \
      throw std::invalid_argument("Unsupported blk_q/blk_k/warp_q/warp_k configuration"); \
    } \
  } while (0)

struct Sm80QkLaunchParams {
  int head_dim;
  int batch_size;
  int qo_len;
  int kv_len;
  int num_qo_heads;
  int num_kv_heads;
  int num_kv_groups;
  int stride_bz_q;
  int stride_seq_q;
  int stride_h_q;
  int stride_bz_k;
  int stride_seq_k;
  int stride_h_k;
  int stride_bz_v;
  int stride_seq_v;
  int stride_h_v;
  int stride_bz_o;
  int stride_seq_o;
  int stride_h_o;
  Tensor lse;
};

Sm80QkLaunchParams prepare_sm80_qk_launch_params(const Tensor &query,
                                                 const Tensor &key,
                                                 const Tensor &value,
                                                 const Tensor &output,
                                                 const Tensor &query_scale,
                                                 const Tensor &key_scale,
                                                 const int64_t tensor_layout,
                                                 const int64_t return_lse)
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
  CHECK_DTYPE(query_scale, torch::headeronly::ScalarType::Float);
  CHECK_DTYPE(key_scale, torch::headeronly::ScalarType::Float);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);

  Sm80QkLaunchParams params = {
    static_cast<int>(query.size(3)),
    static_cast<int>(query.size(0)),
    0,
    0,
    0,
    0,
    0,
    static_cast<int>(query.stride(0)),
    0,
    0,
    static_cast<int>(key.stride(0)),
    0,
    0,
    static_cast<int>(value.stride(0)),
    0,
    0,
    static_cast<int>(output.stride(0)),
    0,
    0,
    torch::stable::new_empty(query, {0}, std::make_optional(torch::headeronly::ScalarType::Float)),
  };

  if (tensor_layout == 0)
  {
    params.qo_len = static_cast<int>(query.size(1));
    params.kv_len = static_cast<int>(key.size(1));
    params.num_qo_heads = static_cast<int>(query.size(2));
    params.num_kv_heads = static_cast<int>(key.size(2));
    CHECK_SHAPE(key, params.batch_size, params.kv_len, params.num_kv_heads, params.head_dim);
    CHECK_SHAPE(value, params.batch_size, params.kv_len, params.num_kv_heads, params.head_dim);

    params.stride_seq_q = static_cast<int>(query.stride(1));
    params.stride_seq_k = static_cast<int>(key.stride(1));
    params.stride_seq_v = static_cast<int>(value.stride(1));
    params.stride_seq_o = static_cast<int>(output.stride(1));

    params.stride_h_q = static_cast<int>(query.stride(2));
    params.stride_h_k = static_cast<int>(key.stride(2));
    params.stride_h_v = static_cast<int>(value.stride(2));
    params.stride_h_o = static_cast<int>(output.stride(2));
  }
  else if (tensor_layout == 1)
  {
    params.qo_len = static_cast<int>(query.size(2));
    params.kv_len = static_cast<int>(key.size(2));
    params.num_qo_heads = static_cast<int>(query.size(1));
    params.num_kv_heads = static_cast<int>(key.size(1));
    CHECK_SHAPE(key, params.batch_size, params.num_kv_heads, params.kv_len, params.head_dim);
    CHECK_SHAPE(value, params.batch_size, params.num_kv_heads, params.kv_len, params.head_dim);

    params.stride_seq_q = static_cast<int>(query.stride(2));
    params.stride_seq_k = static_cast<int>(key.stride(2));
    params.stride_seq_v = static_cast<int>(value.stride(2));
    params.stride_seq_o = static_cast<int>(output.stride(2));

    params.stride_h_q = static_cast<int>(query.stride(1));
    params.stride_h_k = static_cast<int>(key.stride(1));
    params.stride_h_v = static_cast<int>(value.stride(1));
    params.stride_h_o = static_cast<int>(output.stride(1));
  }
  else
  {
    throw std::invalid_argument("tensor_layout must be 0 or 1");
  }

  if (params.num_qo_heads % params.num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads (" << params.num_qo_heads << ") must be divisible by num_kv_heads (" << params.num_kv_heads << ")";
    throw std::invalid_argument(err_msg.str());
  }

  params.num_kv_groups = params.num_qo_heads / params.num_kv_heads;
  params.lse = return_lse
    ? torch::stable::new_empty(query, {params.batch_size, params.num_qo_heads, params.qo_len}, std::make_optional(torch::headeronly::ScalarType::Float))
    : torch::stable::new_empty(query, {0}, std::make_optional(torch::headeronly::ScalarType::Float));

  return params;
}

// tensor_layout 0 for [B, N, H, D], 1 for [B, H, N, D]
Tensor qk_int8_sv_f16_accum_f32_attn(const Tensor &query,
                    const Tensor &key,
                    const Tensor &value,
                    const Tensor &output,
                    const Tensor &query_scale,
                    const Tensor &key_scale,
                    const int64_t tensor_layout,
                    const int64_t is_causal,
                    const int64_t qk_quant_gran,
                    const double sm_scale,
                    const int64_t blk_q,
                    const int64_t blk_k,
                    const int64_t warp_q,
                    const int64_t warp_k,
                    const int64_t return_lse)
{
  const auto params = prepare_sm80_qk_launch_params(query, key, value, output, query_scale, key_scale, tensor_layout, return_lse);
  const auto output_dtype = output.scalar_type();

  DISPATCH_HEAD_DIM(params.head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
        DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
          DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
            LAUNCH_QK_INT_SV_F16_CONFIGURED(blk_q, blk_k, warp_q, warp_k, float, false, ComputeUnit::kTensorCore, false, nullptr);
          });
        });
      });
    });
  });

  return params.lse;
}

Tensor qk_int8_sv_f16_accum_f16_attn(const Tensor &query,
                    const Tensor &key,
                    const Tensor &value,
                    const Tensor &output,
                    const Tensor &query_scale,
                    const Tensor &key_scale,
                    const int64_t tensor_layout,
                    const int64_t is_causal,
                    const int64_t qk_quant_gran,
                    const double sm_scale,
                    const int64_t blk_q,
                    const int64_t blk_k,
                    const int64_t warp_q,
                    const int64_t warp_k,
                    const int64_t return_lse)
{
  const auto params = prepare_sm80_qk_launch_params(query, key, value, output, query_scale, key_scale, tensor_layout, return_lse);
  const auto output_dtype = output.scalar_type();

  DISPATCH_HEAD_DIM(params.head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
        DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
          DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
            LAUNCH_QK_INT_SV_F16_CONFIGURED(blk_q, blk_k, warp_q, warp_k, half, false, ComputeUnit::kTensorCore, false, nullptr);
          });
        });
      });
    });
  });

  return params.lse;
}

Tensor qk_int8_sv_f16_accum_f16_attn_inst_buf(const Tensor &query,
                    const Tensor &key,
                    const Tensor &value,
                    const Tensor &output,
                    const Tensor &query_scale,
                    const Tensor &key_scale,
                    const int64_t tensor_layout,
                    const int64_t is_causal,
                    const int64_t qk_quant_gran,
                    const double sm_scale,
                    const int64_t blk_q,
                    const int64_t blk_k,
                    const int64_t warp_q,
                    const int64_t warp_k,
                    const int64_t return_lse)
{
  const auto params = prepare_sm80_qk_launch_params(query, key, value, output, query_scale, key_scale, tensor_layout, return_lse);
  const auto output_dtype = output.scalar_type();

  DISPATCH_HEAD_DIM(params.head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
        DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
          DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
            LAUNCH_QK_INT_SV_F16_CONFIGURED(blk_q, blk_k, warp_q, warp_k, float, true, ComputeUnit::kTensorCore, false, nullptr);
          });
        });
      });
    });
  });

  return params.lse;
}

Tensor qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(const Tensor &query,
                    const Tensor &key,
                    const Tensor &value,
                    const Tensor &output,
                    const Tensor &query_scale,
                    const Tensor &key_scale,
                    const Tensor &value_mean,
                    const int64_t tensor_layout,
                    const int64_t is_causal,
                    const int64_t qk_quant_gran,
                    const double sm_scale,
                    const int64_t blk_q,
                    const int64_t blk_k,
                    const int64_t warp_q,
                    const int64_t warp_k,
                    const int64_t return_lse)
{
  CHECK_CUDA(value_mean);
  CHECK_CONTIGUOUS(value_mean);
  CHECK_DIMS(value_mean, 3);

  const auto params = prepare_sm80_qk_launch_params(query, key, value, output, query_scale, key_scale, tensor_layout, return_lse);
  const auto output_dtype = output.scalar_type();
  const auto value_mean_dtype = value_mean.scalar_type();

  STD_TORCH_CHECK(value_mean_dtype == output_dtype, "value_mean and output must have the same dtype");

  DISPATCH_HEAD_DIM(params.head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
        DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
          DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
            CHECK_SHAPE(value_mean, params.batch_size, params.num_kv_heads, params.head_dim);
            LAUNCH_QK_INT_SV_F16_CONFIGURED(blk_q, blk_k, warp_q, warp_k, half, false, ComputeUnit::kTensorCore, true, const_ptr<DTypeOut>(value_mean));
          });
        });
      });
    });
  });

  return params.lse;
}

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace sageattention::qattn_cutlass {

inline int32_t ceil_div_int(const int32_t value, const int32_t divisor)
{
  return (value + divisor - 1) / divisor;
}

struct LaunchParams {
  const int8_t *query;
  const int8_t *key;
  const half *value;
  half *output;
  float *lse;
  const float *query_scale;
  const float *key_scale;
  int32_t head_dim;
  int32_t batch_size;
  int32_t qo_len;
  int32_t kv_len;
  int32_t num_qo_heads;
  int32_t num_kv_heads;
  int32_t num_kv_groups;
  int32_t stride_bz_q;
  int32_t stride_seq_q;
  int32_t stride_h_q;
  int32_t stride_bz_k;
  int32_t stride_seq_k;
  int32_t stride_h_k;
  int32_t stride_bz_v;
  int32_t stride_seq_v;
  int32_t stride_h_v;
  int32_t stride_bz_o;
  int32_t stride_seq_o;
  int32_t stride_h_o;
  int32_t blk_q;
  int32_t blk_k;
  int32_t warp_q;
  int32_t warp_k;
  float sm_scale;
  cudaStream_t stream;
};

template <int32_t HeadDim, int32_t CtaQ, int32_t CtaK, int32_t WarpQ, int32_t WarpK, bool ReturnLse>
void launch_kernel(const LaunchParams &params);

} // namespace sageattention::qattn_cutlass

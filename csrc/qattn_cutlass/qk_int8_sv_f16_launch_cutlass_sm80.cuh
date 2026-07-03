#pragma once

#include "qk_int8_sv_f16_kernel_cutlass_sm80.cuh"
#include "qk_int8_sv_f16_params_cutlass_sm80.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>

namespace sageattention::qattn_cutlass {

template <int32_t HeadDim, int32_t CtaQ, int32_t CtaK, int32_t WarpQ, int32_t WarpK, bool ReturnLse>
void launch_kernel(const LaunchParams &params)
{
  constexpr size_t smem_max = std::max(
    CtaQ * HeadDim * sizeof(int8_t) + CtaK * HeadDim * sizeof(int8_t) + CtaK * HeadDim * sizeof(half),
    CtaQ * HeadDim * sizeof(half));
  auto kernel_func = qk_int8_sv_f16_accum_f32_attn_kernel<CtaQ, CtaK, WarpQ, WarpK, HeadDim, ReturnLse>;
  cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

  const int32_t num_q_blocks = ceil_div_int(params.qo_len, CtaQ);
  const int32_t num_warps = (CtaQ / WarpQ) * (CtaK / WarpK);
  const int32_t num_lanes = 32;
  const dim3 grid(num_q_blocks, params.num_qo_heads, params.batch_size);
  const dim3 block(num_lanes, num_warps);
  kernel_func<<<grid, block, smem_max, params.stream>>>(
    params.query,
    params.key,
    params.value,
    params.output,
    ReturnLse ? params.lse : nullptr,
    params.query_scale,
    params.key_scale,
    params.qo_len,
    params.kv_len,
    params.num_kv_groups,
    params.stride_bz_q, params.stride_seq_q, params.stride_h_q,
    params.stride_bz_k, params.stride_seq_k, params.stride_h_k,
    params.stride_bz_v, params.stride_seq_v, params.stride_h_v,
    params.stride_bz_o, params.stride_seq_o, params.stride_h_o,
    params.sm_scale);
}

} // namespace sageattention::qattn_cutlass

#include "qk_int8_sv_f16_launch_sm80.cuh"

Tensor qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(const Tensor &query,
                                                 const Tensor &key,
                                                 const Tensor &value,
                                                 const Tensor &output,
                                                 const Tensor &query_scale,
                                                 const Tensor &key_scale,
                                                 const Tensor &value_mean,
                                                 const int64_t tensor_layout,
                                                 const bool is_causal,
                                                 const double sm_scale,
                                                 const int64_t blk_q,
                                                 const int64_t blk_k,
                                                 const int64_t warp_q,
                                                 const int64_t warp_k,
                                                 const bool return_lse)
{
  return run_sm80_qk_attn<half, false, ComputeUnit::kTensorCore, true>(
    query, key, value, output, query_scale, key_scale, &value_mean, tensor_layout, is_causal,
    sm_scale, blk_q, blk_k, warp_q, warp_k, return_lse);
}

#include "qk_int8_sv_f8_launch_sm89.cuh"

Tensor qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(const Tensor &query,
                                                          const Tensor &key,
                                                          const Tensor &value,
                                                          const Tensor &output,
                                                          const Tensor &query_scale,
                                                          const Tensor &key_scale,
                                                          const Tensor &value_scale,
                                                          const int64_t tensor_layout,
                                                          const bool is_causal,
                                                          const double sm_scale,
                                                          const int64_t blk_q,
                                                          const int64_t blk_k,
                                                          const int64_t warp_q,
                                                          const int64_t warp_k,
                                                          const bool return_lse)
{
  // DTypeSVAccum=float, UseInstBuffer=true, DenominatorAccumUnit=kCudaCore,
  // FuseVScale=true, FuseVMean=false, UsePvFp16Accu=true (SageAttention2++ sv_f8)
  return run_sm89_qk_attn<float, true, ComputeUnit::kCudaCore, true, false, true>(
    query, key, value, output, query_scale, key_scale, value_scale, nullptr, tensor_layout, is_causal,
    sm_scale, blk_q, blk_k, warp_q, warp_k, return_lse);
}

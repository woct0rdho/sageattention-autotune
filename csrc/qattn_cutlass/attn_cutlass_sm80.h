#pragma once

#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

Tensor qk_int8_sv_f16_accum_f32_attn_cutlass(const Tensor &query,
                                             const Tensor &key,
                                             const Tensor &value,
                                             const Tensor &output,
                                             const Tensor &query_scale,
                                             const Tensor &key_scale,
                                             int64_t tensor_layout,
                                             double sm_scale,
                                             int64_t blk_q,
                                             int64_t blk_k,
                                             int64_t warp_q,
                                             int64_t warp_k,
                                             bool return_lse);

void qk_int8_sv_f16_accum_f32_attn_bwd_cutlass(const Tensor &query,
                                               const Tensor &key,
                                               const Tensor &query_scale,
                                               const Tensor &key_scale,
                                               const Tensor &value,
                                               const Tensor &output,
                                               const Tensor &dO,
                                               const Tensor &lse,
                                               const Tensor &delta,
                                               const Tensor &dQ_accum,
                                               const Tensor &dO_int8,
                                               const Tensor &dO_scale,
                                               const Tensor &dS_q_factors,
                                               const Tensor &dS_k_factors,
                                               const Tensor &dQ,
                                               const Tensor &dK,
                                               const Tensor &dV,
                                               int64_t tensor_layout,
                                               double sm_scale,
                                               int64_t blk_q,
                                               int64_t blk_k,
                                               int64_t bwd_block_m,
                                               int64_t bwd_block_n);

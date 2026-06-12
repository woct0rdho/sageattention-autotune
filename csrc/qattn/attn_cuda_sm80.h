#pragma once

#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

Tensor qk_int8_sv_f16_accum_f32_attn(const Tensor &query,
                                     const Tensor &key,
                                     const Tensor &value,
                                     const Tensor &output,
                                     const Tensor &query_scale,
                                     const Tensor &key_scale,
                                     int64_t tensor_layout,
                                     int64_t is_causal,
                                     double sm_scale,
                                     int64_t blk_q,
                                     int64_t blk_k,
                                     int64_t warp_q,
                                     int64_t warp_k,
                                     int64_t return_lse);

Tensor qk_int8_sv_f16_accum_f16_attn(const Tensor &query,
                                     const Tensor &key,
                                     const Tensor &value,
                                     const Tensor &output,
                                     const Tensor &query_scale,
                                     const Tensor &key_scale,
                                     int64_t tensor_layout,
                                     int64_t is_causal,
                                     double sm_scale,
                                     int64_t blk_q,
                                     int64_t blk_k,
                                     int64_t warp_q,
                                     int64_t warp_k,
                                     int64_t return_lse);

Tensor qk_int8_sv_f16_accum_f16_attn_inst_buf(const Tensor &query,
                                              const Tensor &key,
                                              const Tensor &value,
                                              const Tensor &output,
                                              const Tensor &query_scale,
                                              const Tensor &key_scale,
                                              int64_t tensor_layout,
                                              int64_t is_causal,
                                              double sm_scale,
                                              int64_t blk_q,
                                              int64_t blk_k,
                                              int64_t warp_q,
                                              int64_t warp_k,
                                              int64_t return_lse);

Tensor qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(const Tensor &query,
                                                 const Tensor &key,
                                                 const Tensor &value,
                                                 const Tensor &output,
                                                 const Tensor &query_scale,
                                                 const Tensor &key_scale,
                                                 const Tensor &value_mean,
                                                 int64_t tensor_layout,
                                                 int64_t is_causal,
                                                 double sm_scale,
                                                 int64_t blk_q,
                                                 int64_t blk_k,
                                                 int64_t warp_q,
                                                 int64_t warp_k,
                                                 int64_t return_lse);

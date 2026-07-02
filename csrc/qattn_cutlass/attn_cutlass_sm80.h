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

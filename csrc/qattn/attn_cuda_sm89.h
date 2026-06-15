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

#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

// int8 QK / fp8 (e4m3) PV attention, f32 accumulation, per-channel V dequant
// fused in the kernel. V must be pre-transposed and fp8-quantized to
// [batch, num_kv_heads, head_dim, kv_len_padded] (int8 holding e4m3).
Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(const Tensor &query,
                                                 const Tensor &key,
                                                 const Tensor &value,
                                                 const Tensor &output,
                                                 const Tensor &query_scale,
                                                 const Tensor &key_scale,
                                                 const Tensor &value_scale,
                                                 int64_t tensor_layout,
                                                 bool is_causal,
                                                 double sm_scale,
                                                 int64_t blk_q,
                                                 int64_t blk_k,
                                                 int64_t warp_q,
                                                 int64_t warp_k,
                                                 bool return_lse);

// Same as above but using the SageAttention2++ instruction-buffer path with
// fp16 PV accumulation (the "sv_f8" fast path). Slightly lower precision,
// higher throughput.
Tensor qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(const Tensor &query,
                                                          const Tensor &key,
                                                          const Tensor &value,
                                                          const Tensor &output,
                                                          const Tensor &query_scale,
                                                          const Tensor &key_scale,
                                                          const Tensor &value_scale,
                                                          int64_t tensor_layout,
                                                          bool is_causal,
                                                          double sm_scale,
                                                          int64_t blk_q,
                                                          int64_t blk_k,
                                                          int64_t warp_q,
                                                          int64_t warp_k,
                                                          bool return_lse);

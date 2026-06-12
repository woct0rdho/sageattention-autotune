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

#include "qk_int8_sv_f16_launch_sm80.cuh"

Tensor qk_int8_sv_f16_accum_f16_attn_inst_buf(const Tensor &query,
                                              const Tensor &key,
                                              const Tensor &value,
                                              const Tensor &output,
                                              const Tensor &query_scale,
                                              const Tensor &key_scale,
                                              const int64_t tensor_layout,
                                              const int64_t is_causal,
                                              const double sm_scale,
                                              const int64_t blk_q,
                                              const int64_t blk_k,
                                              const int64_t warp_q,
                                              const int64_t warp_k,
                                              const int64_t return_lse)
{
  return run_sm80_qk_attn<float, true, ComputeUnit::kTensorCore, false>(
    query, key, value, output, query_scale, key_scale, nullptr, tensor_layout, is_causal,
    sm_scale, blk_q, blk_k, warp_q, warp_k, return_lse);
}

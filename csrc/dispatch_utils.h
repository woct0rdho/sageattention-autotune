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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cstdint>

namespace sageattention::dispatch {

template <typename Func>
inline void fp16_dtype(const torch::headeronly::ScalarType dtype, const Func &func)
{
  if (dtype == torch::headeronly::ScalarType::Half)
  {
    func.template operator()<half>();
  }
  else if (dtype == torch::headeronly::ScalarType::BFloat16)
  {
    func.template operator()<nv_bfloat16>();
  }
  else
  {
    STD_TORCH_CHECK(false, "Unsupported fp16/bf16 dtype");
  }
}

template <typename Func>
inline void head_dim(const int64_t head_dim, const Func &func)
{
  if (head_dim == 64)
  {
    func.template operator()<64>();
  }
  else if (head_dim == 128)
  {
    func.template operator()<128>();
  }
  else if (head_dim == 256)
  {
    func.template operator()<256>();
  }
  else
  {
    STD_TORCH_CHECK(false, "Unsupported head dim: ", head_dim);
  }
}

template <typename Func>
inline void block_size(const int64_t block_size, const Func &func)
{
  if (block_size == 64)
  {
    func.template operator()<64>();
  }
  else if (block_size == 128)
  {
    func.template operator()<128>();
  }
  else
  {
    STD_TORCH_CHECK(false, "Unsupported block_size: ", block_size);
  }
}

template <typename Func>
inline void warp_block_size(const int64_t warp_block_size, const Func &func)
{
  if (warp_block_size == 16)
  {
    func.template operator()<16>();
  }
  else if (warp_block_size == 32)
  {
    func.template operator()<32>();
  }
  else
  {
    STD_TORCH_CHECK(false, "Unsupported warp_block_size: ", warp_block_size);
  }
}

template <typename Func>
inline void boolean(const int64_t value, const char *name, const Func &func)
{
  if (value == 1)
  {
    func.template operator()<true>();
  }
  else if (value == 0)
  {
    func.template operator()<false>();
  }
  else
  {
    STD_TORCH_CHECK(false, "Unsupported ", name, ": ", value);
  }
}

} // namespace sageattention::dispatch

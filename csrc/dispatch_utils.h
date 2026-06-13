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
inline void boolean(const bool value, const Func &func)
{
  if (value)
  {
    func.template operator()<true>();
  }
  else
  {
    func.template operator()<false>();
  }
}

} // namespace sageattention::dispatch

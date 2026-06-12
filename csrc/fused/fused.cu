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

#include "../dispatch_utils.h"
#include "../reduction_utils.cuh"
#include "../utils.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cstdint>
#include <type_traits>

using torch::stable::Tensor;

namespace sa_dispatch = sageattention::dispatch;

template <typename T>
const T *const_ptr(const Tensor &tensor)
{
  return reinterpret_cast<const T*>(tensor.const_data_ptr());
}

template <typename T>
T *mutable_ptr(const Tensor &tensor)
{
  return reinterpret_cast<T*>(tensor.mutable_data_ptr());
}

__device__ __forceinline__ int8_t float_to_int8_rn(float x)
{
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
}

template <typename T>
__device__ __forceinline__ float convert_to_float(T val)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");

  if constexpr (std::is_same<T, half>::value)
  {
    return __half2float(val);
  }
  else if constexpr (std::is_same<T, nv_bfloat16>::value)
  {
    return __bfloat162float(val);
  }
}

template <uint32_t head_dim, uint32_t BLOCK_SIZE, uint32_t num_pack_per_thread = 1, bool sub_mean = false, typename T>
__global__ void QuantInt8Kernel(const T *__restrict__ input, const T *__restrict__ mean, int8_t *__restrict__ output, float *__restrict__ scale, const uint32_t num_tokens,
                                const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input,
                                const uint32_t stride_bz_mean, const uint32_t stride_h_mean,
                                const uint32_t stride_bz_output, const uint32_t stride_seq_output, const uint32_t stride_h_output,
                                const uint32_t stride_bz_scale, const uint32_t stride_h_scale)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");
  static_assert(num_pack_per_thread > 0, "The number of pack per thread must be greater than 0");

  constexpr uint32_t pack_size = 8; // float4 contains 8 half or 8 bfloat16
  constexpr uint32_t num_threads_per_token = head_dim / pack_size;

  static_assert(num_threads_per_token <= 32, "The number of threads per token must be less than or equal to warp size");

  T x_val[num_pack_per_thread][8];
  T mean_val[8];
  float x_val_float[num_pack_per_thread][8];
  float mean_val_float[8];

  const uint32_t bx = blockIdx.x;
  const uint32_t head_id = blockIdx.y;
  const uint32_t batch_id = blockIdx.z;
  const uint32_t thread_id = threadIdx.x;

  const uint32_t thread_base_token = bx * BLOCK_SIZE + thread_id / num_threads_per_token;
  const T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + thread_base_token * stride_seq_input + thread_id % num_threads_per_token * pack_size;
  int8_t *const output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + thread_base_token * stride_seq_output + thread_id % num_threads_per_token * pack_size;
  float *const scale_ptr_base = scale + batch_id * stride_bz_scale + head_id * stride_h_scale + bx;

  if constexpr (sub_mean)
  {
    const T *mean_ptr_base = mean + batch_id * stride_bz_mean + head_id * stride_h_mean + thread_id % num_threads_per_token * pack_size;
    *reinterpret_cast<float4*>(&mean_val[0]) = *reinterpret_cast<const float4*>(mean_ptr_base);
#pragma unroll
    for (uint32_t j = 0; j < 8; j++)
    {
      mean_val_float[j] = convert_to_float(mean_val[j]);
    }
  }

  constexpr uint32_t iter_stride = BLOCK_SIZE / num_pack_per_thread;

  // load the data
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
    if (thread_base_token + i * iter_stride < num_tokens)
    {
      *reinterpret_cast<float4*>(&x_val[i][0]) = *reinterpret_cast<const float4*>(input_ptr_base + i * iter_stride * stride_seq_input);
#pragma unroll
      for (uint32_t j = 0; j < 8; j++)
      {
        x_val_float[i][j] = convert_to_float(x_val[i][j]);
      }

      if constexpr (sub_mean)
      {
#pragma unroll
        for (uint32_t j = 0; j < 8; j++)
        {
          x_val_float[i][j] -= mean_val_float[j];
        }
      }
    }
    else
    {
#pragma unroll
      for (uint32_t j = 0; j < 8; j++)
      {
        x_val_float[i][j] = 0.0f;
      }
    }
  }

  float amax_val = 0.0000001f; // prevent from dividing by zero

#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < 8; j++)
    {
      amax_val = fmaxf(amax_val, fabsf(x_val_float[i][j]));
    }
  }

  __shared__ float s_amax;
  const float block_amax_val = vllm::blockReduceMax(amax_val);
  if (thread_id == 0)
  {
    s_amax = block_amax_val;
    scale_ptr_base[0] = s_amax / 127.0f;
  }

  __syncthreads();

  const float tmp_scale = 127.0f / s_amax;

  char4 o_val[num_pack_per_thread][2];

#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < 2; j += 1)
    {
      o_val[i][j] = make_char4(
        float_to_int8_rn(x_val_float[i][j * 4 + 0] * tmp_scale),
        float_to_int8_rn(x_val_float[i][j * 4 + 1] * tmp_scale),
        float_to_int8_rn(x_val_float[i][j * 4 + 2] * tmp_scale),
        float_to_int8_rn(x_val_float[i][j * 4 + 3] * tmp_scale)
      );
    }
  }

  // int8 result
#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {

    if (thread_base_token + i * iter_stride < num_tokens)
    {
      *reinterpret_cast<float2*>(output_ptr_base + i * iter_stride * stride_seq_output) = *reinterpret_cast<float2*>(&o_val[i][0]);
    }
  }
}

template <uint32_t head_dim, uint32_t BLOCK_SIZE, uint32_t num_pack_per_thread = 1, typename T>
__global__ void SubMeanKernel(const T *__restrict__ input, const T *__restrict__ mean, half *__restrict__ output, const uint32_t num_tokens,
                              const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input,
                              const uint32_t stride_bz_mean, const uint32_t stride_h_mean,
                              const uint32_t stride_bz_output, const uint32_t stride_seq_output, const uint32_t stride_h_output)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");
  static_assert(num_pack_per_thread > 0, "The number of pack per thread must be greater than 0");

  using T2 = typename std::conditional<std::is_same<T, half>::value, half2, nv_bfloat162>::type;

  constexpr uint32_t pack_size = 8; // float4 contains 8 half or 8 bfloat16
  constexpr uint32_t num_threads_per_token = head_dim / pack_size;

  static_assert(num_threads_per_token <= 32, "The number of threads per token must be less than or equal to warp size");

  T2 x_val[num_pack_per_thread][4];
  T2 mean_val[4];

  const uint32_t bx = blockIdx.x;
  const uint32_t head_id = blockIdx.y;
  const uint32_t batch_id = blockIdx.z;
  const uint32_t thread_id = threadIdx.x;

  const uint32_t thread_base_token = bx * BLOCK_SIZE + thread_id / num_threads_per_token;
  const T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + thread_base_token * stride_seq_input + thread_id % num_threads_per_token * pack_size;
  const T *mean_ptr_base = mean + batch_id * stride_bz_mean + head_id * stride_h_mean + thread_id % num_threads_per_token * pack_size;
  half *const output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + thread_base_token * stride_seq_output + thread_id % num_threads_per_token * pack_size;

  *reinterpret_cast<float4*>(&mean_val[0]) = *reinterpret_cast<const float4*>(mean_ptr_base);

  constexpr uint32_t iter_stride = BLOCK_SIZE / num_pack_per_thread;

  // load the data
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
    if (thread_base_token + i * iter_stride < num_tokens)
    {
      *reinterpret_cast<float4*>(&x_val[i][0]) = *reinterpret_cast<const float4*>(input_ptr_base + i * iter_stride * stride_seq_input);
#pragma unroll
      for (uint32_t j = 0; j < 4; j++)
      {
        x_val[i][j] = __hsub2(x_val[i][j], mean_val[j]);

        if constexpr (std::is_same<T, nv_bfloat16>::value)
        {
          ((half2*)x_val[i])[j] = __float22half2_rn(__bfloat1622float2(x_val[i][j]));
        }
      }
    }
  }

#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
    if (thread_base_token + i * iter_stride < num_tokens)
    {
      *reinterpret_cast<float4*>(output_ptr_base + i * iter_stride * stride_seq_output) = *reinterpret_cast<float4*>(&x_val[i][0]);
    }
  }
}

void quant_per_block_int8_cuda(const Tensor &input,
                               const Tensor &output,
                               const Tensor &scale,
                               const int64_t block_size,
                               const int64_t tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);

  CHECK_DTYPE(output, torch::headeronly::ScalarType::Char);
  CHECK_DTYPE(scale, torch::headeronly::ScalarType::Float);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  const int stride_bz_input = input.stride(0);
  const int stride_bz_output = output.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_h_input = input.stride(2);
    stride_seq_output = output.stride(1);
    stride_h_output = output.stride(2);
  }
  else
  {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_h_input = input.stride(1);
    stride_seq_output = output.stride(2);
    stride_h_output = output.stride(1);
  }

  const auto input_dtype = input.scalar_type();

  sa_dispatch::fp16_dtype(input_dtype, [&]<typename CType>() {
    sa_dispatch::block_size(block_size, [&]<int BLOCK_SIZE>() {
      sa_dispatch::head_dim(head_dim, [&]<int HEAD_DIM>() {
        CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
        CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE);

        dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);

        constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

        dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

        QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, false, CType><<<grid, block>>>(
          const_ptr<CType>(input),
          nullptr,
          mutable_ptr<int8_t>(output),
          mutable_ptr<float>(scale),
          num_tokens,
          stride_bz_input, stride_seq_input, stride_h_input,
          0, 0,
          stride_bz_output, stride_seq_output, stride_h_output,
          scale.stride(0), scale.stride(1)
        );
      });
    });
  });
}

void quant_per_block_int8_fuse_sub_mean_cuda(const Tensor &input,
                                             const Tensor &mean,
                                             const Tensor &output,
                                             const Tensor &scale,
                                             const int64_t block_size,
                                             const int64_t tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(mean);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);

  CHECK_DTYPE(output, torch::headeronly::ScalarType::Char);
  CHECK_DTYPE(scale, torch::headeronly::ScalarType::Float);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(mean);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(mean, 3);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  const int stride_bz_input = input.stride(0);
  const int stride_bz_output = output.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_h_input = input.stride(2);
    stride_seq_output = output.stride(1);
    stride_h_output = output.stride(2);
  }
  else
  {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_h_input = input.stride(1);
    stride_seq_output = output.stride(2);
    stride_h_output = output.stride(1);
  }

  const auto input_dtype = input.scalar_type();
  const auto mean_dtype = mean.scalar_type();

  STD_TORCH_CHECK(input_dtype == mean_dtype, "Input and mean must have the same data type");

  sa_dispatch::fp16_dtype(input_dtype, [&]<typename CType>() {
    sa_dispatch::block_size(block_size, [&]<int BLOCK_SIZE>() {
      sa_dispatch::head_dim(head_dim, [&]<int HEAD_DIM>() {
        CHECK_SHAPE(mean, batch_size, num_heads, head_dim);
        CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
        CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE);

        dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);

        constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

        dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

        QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, true, CType><<<grid, block>>>(
          const_ptr<CType>(input),
          const_ptr<CType>(mean),
          mutable_ptr<int8_t>(output),
          mutable_ptr<float>(scale),
          num_tokens,
          stride_bz_input, stride_seq_input, stride_h_input,
          mean.stride(0), mean.stride(1),
          stride_bz_output, stride_seq_output, stride_h_output,
          scale.stride(0), scale.stride(1)
        );
      });
    });
  });
}

// use block size 128 and warp_block size 32
void quant_per_warp_int8_cuda(const Tensor &input,
                              const Tensor &output,
                              const Tensor &scale,
                              const int64_t block_size,
                              const int64_t warp_block_size,
                              const int64_t tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);

  CHECK_DTYPE(output, torch::headeronly::ScalarType::Char);
  CHECK_DTYPE(scale, torch::headeronly::ScalarType::Float);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  const int stride_bz_input = input.stride(0);
  const int stride_bz_output = output.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_h_input = input.stride(2);
    stride_seq_output = output.stride(1);
    stride_h_output = output.stride(2);
  }
  else
  {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_h_input = input.stride(1);
    stride_seq_output = output.stride(2);
    stride_h_output = output.stride(1);
  }

  const auto input_dtype = input.scalar_type();

  sa_dispatch::fp16_dtype(input_dtype, [&]<typename CType>() {
    sa_dispatch::block_size(block_size, [&]<int BLOCK_SIZE>() {
      sa_dispatch::warp_block_size(warp_block_size, [&]<int WARP_BLOCK_SIZE>() {
        sa_dispatch::head_dim(head_dim, [&]<int HEAD_DIM>() {
          CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
          CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE * (BLOCK_SIZE / WARP_BLOCK_SIZE));

          dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE * (BLOCK_SIZE / WARP_BLOCK_SIZE), num_heads, batch_size);

          constexpr int num_pack_per_thread = (WARP_BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

          dim3 block(WARP_BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

          QuantInt8Kernel<HEAD_DIM, WARP_BLOCK_SIZE, num_pack_per_thread, false, CType><<<grid, block>>>(
            const_ptr<CType>(input),
            nullptr,
            mutable_ptr<int8_t>(output),
            mutable_ptr<float>(scale),
            num_tokens,
            stride_bz_input, stride_seq_input, stride_h_input,
            0, 0,
            stride_bz_output, stride_seq_output, stride_h_output,
            scale.stride(0), scale.stride(1)
          );
        });
      });
    });
  });
}

void sub_mean_cuda(const Tensor &input,
                   const Tensor &mean,
                   const Tensor &output,
                   const int64_t tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(mean);
  CHECK_CUDA(output);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(mean);
  CHECK_CONTIGUOUS(output);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(mean, 3);
  CHECK_DIMS(output, 4);

  CHECK_DTYPE(output, torch::headeronly::ScalarType::Half);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  const int stride_bz_input = input.stride(0);
  const int stride_bz_output = output.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_h_input = input.stride(2);
    stride_seq_output = output.stride(1);
    stride_h_output = output.stride(2);
  }
  else
  {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_h_input = input.stride(1);
    stride_seq_output = output.stride(2);
    stride_h_output = output.stride(1);
  }

  const auto input_dtype = input.scalar_type();
  const auto mean_dtype = mean.scalar_type();

  STD_TORCH_CHECK(input_dtype == mean_dtype, "Input and mean must have the same data type");

  sa_dispatch::fp16_dtype(input_dtype, [&]<typename CType>() {
    sa_dispatch::head_dim(head_dim, [&]<int HEAD_DIM>() {
      CHECK_SHAPE(mean, batch_size, num_heads, head_dim);
      CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));

      constexpr int BLOCK_SIZE = (HEAD_DIM == 128) ? 64 : 128;

      dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);

      constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

      dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

      SubMeanKernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, CType><<<grid, block>>>(
        const_ptr<CType>(input),
        const_ptr<CType>(mean),
        mutable_ptr<half>(output),
        num_tokens,
        stride_bz_input, stride_seq_input, stride_h_input,
        mean.stride(0), mean.stride(1),
        stride_bz_output, stride_seq_output, stride_h_output
      );
    });
  });
}

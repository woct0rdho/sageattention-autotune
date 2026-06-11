/*
 * Adapted from Flashinfer, https://github.com/flashinfer-ai/flashinfer/blob/v0.1.5/include/flashinfer/mma.cuh
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Modifications copyright (c) 2024 by SageAttention team.
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

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace mma {

enum class MMAMode {
  kInit = 0U,
  kInplaceUpdate = 1U,
};

template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t* r, T* smem_ptr)
{
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(smem_int_ptr));
}

template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t* r, T* smem_ptr)
{
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(smem_int_ptr));
}

template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f32(
    float* c,
    uint32_t* a,
    uint32_t* b)
{
  if constexpr (mma_mode == MMAMode::kInplaceUpdate) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]),
          "f"(c[2]), "f"(c[3]));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(c[4]), "=f"(c[5]), "=f"(c[6]), "=f"(c[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[2]), "r"(b[3]), "f"(c[4]), "f"(c[5]),
          "f"(c[6]), "f"(c[7]));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(0.f), "f"(0.f),
          "f"(0.f), "f"(0.f));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(c[4]), "=f"(c[5]), "=f"(c[6]), "=f"(c[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[2]), "r"(b[3]), "f"(0.f), "f"(0.f),
          "f"(0.f), "f"(0.f));
  }
}

template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f16(
    uint32_t* c,
    uint32_t* a,
    uint32_t* b)
{
  if constexpr (mma_mode == MMAMode::kInplaceUpdate) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(c[0]), "=r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[2]), "r"(b[3]), "r"(c[2]), "r"(c[3]));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(c[0]), "=r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[2]), "r"(b[3]), "r"(0), "r"(0));
  }
}

template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k32_row_col_s8s8s32(
    int32_t* c,
    uint32_t* a,
    uint32_t* b)
{
  if constexpr (mma_mode == MMAMode::kInplaceUpdate) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]),
          "r"(c[2]), "r"(c[3]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(c[4]), "=r"(c[5]), "=r"(c[6]), "=r"(c[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[2]), "r"(b[3]), "r"(c[4]), "r"(c[5]),
          "r"(c[6]), "r"(c[7]));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0), "r"(0),
          "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(c[4]), "=r"(c[5]), "=r"(c[6]), "=r"(c[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[2]), "r"(b[3]), "r"(0), "r"(0),
          "r"(0), "r"(0));
  }
}

__device__ __forceinline__ void rowsum_f16f16f32(float* d, uint32_t* s)
{
  asm volatile(
      "{\n"
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,   _,  %1,   _},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  0.,  %9,  0.};\n"
      "}\n"
      : "=f"(d[0]), "=f"(d[1])
      : "r"(s[0]), "r"(s[1]), "r"(s[2]), "r"(s[3]), "r"(1006648320), "r"(1006648320), "f"(d[0]), "f"(d[1]));
}

} // namespace mma

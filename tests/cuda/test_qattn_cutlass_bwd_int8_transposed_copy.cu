#include <cuda_runtime.h>

#include "csrc/qattn_cutlass/int8_transposed_ldsm_cute.cuh"

#include <cute/algorithm/clear.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/tensor.hpp>

#include <cstdint>
#include <cstdio>

namespace sageattention::qattn_cutlass::test {

using namespace cute;
using MmaAtom = MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>;
using TiledMma = TiledMMA<
    MmaAtom,
    Layout<Shape<_1, _1, _1>>,
    Tile<_16, _16, _32>>;

__global__ void test_kernel(const int8_t *a, const int8_t *b, int32_t *c)
{
  __shared__ __align__(16) int8_t smem_a[16 * 32];
  __shared__ __align__(16) int8_t smem_b[32 * 16];
  const int32_t lane_id = static_cast<int32_t>(threadIdx.x);
  for (int32_t idx = lane_id; idx < 16 * 32; idx += 32)
  {
    smem_a[idx] = a[idx];
    smem_b[idx] = b[idx];
  }
  __syncthreads();

  auto sA = make_tensor(make_smem_ptr(smem_a), Layout<Shape<_16, _32>, Stride<_32, _1>>{});
  auto sB = make_tensor(make_smem_ptr(smem_b), Layout<Shape<_16, _32>, Stride<_1, _16>>{});

  TiledMma tiled_mma;
  const auto thr_mma = tiled_mma.get_thread_slice(lane_id);
  auto tCrA = thr_mma.partition_fragment_A(sA);
  auto tCrB = thr_mma.partition_fragment_B(sB);
  auto tCrC = partition_fragment_C(tiled_mma, Shape<_16, _16>{});
  clear(tCrC);

  const auto tiled_copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, int8_t>{}, tiled_mma);
  const auto thr_copy_a = tiled_copy_a.get_thread_slice(lane_id);
  copy(tiled_copy_a, thr_copy_a.partition_S(sA), thr_copy_a.retile_D(tCrA));

  const auto tiled_copy_b = make_tiled_copy_B(
      sageattention::qattn_cutlass::SmemCopyAtomInt8TransposedB{}, tiled_mma);
  const auto thr_copy_b = tiled_copy_b.get_thread_slice(lane_id);
  copy(tiled_copy_b, thr_copy_b.partition_S(sB), thr_copy_b.retile_D(tCrB));
  gemm(thr_mma, tCrA, tCrB, tCrC);

  const auto tCcC = thr_mma.partition_C(make_identity_tensor(Shape<_16, _16>{}));
  for (int32_t idx = 0; idx < size(tCrC); ++idx)
  {
    const auto coord = tCcC(idx);
    c[static_cast<int32_t>(get<0>(coord)) * 16 + static_cast<int32_t>(get<1>(coord))] = tCrC(idx);
  }
}

} // namespace sageattention::qattn_cutlass::test

int main()
{
  int8_t host_a[16 * 32];
  int8_t host_b[32 * 16];
  for (int32_t m = 0; m < 16; ++m)
  {
    for (int32_t k = 0; k < 32; ++k)
    {
      host_a[m * 32 + k] = static_cast<int8_t>(((m * 3 + k * 5) % 17) - 8);
    }
  }
  for (int32_t k = 0; k < 32; ++k)
  {
    for (int32_t n = 0; n < 16; ++n)
    {
      host_b[k * 16 + n] = static_cast<int8_t>(((k * 7 + n * 11) % 19) - 9);
    }
  }

  int8_t *device_a = nullptr;
  int8_t *device_b = nullptr;
  int32_t *device_c = nullptr;
  cudaMalloc(&device_a, sizeof(host_a));
  cudaMalloc(&device_b, sizeof(host_b));
  cudaMalloc(&device_c, 16 * 16 * sizeof(int32_t));
  cudaMemcpy(device_a, host_a, sizeof(host_a), cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b, sizeof(host_b), cudaMemcpyHostToDevice);

  sageattention::qattn_cutlass::test::test_kernel<<<1, 32>>>(device_a, device_b, device_c);
  const cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess)
  {
    std::printf("CUDA error: %s\n", cudaGetErrorString(error));
    return 1;
  }

  int32_t host_c[16 * 16];
  cudaMemcpy(host_c, device_c, sizeof(host_c), cudaMemcpyDeviceToHost);
  int32_t failures = 0;
  for (int32_t m = 0; m < 16; ++m)
  {
    for (int32_t n = 0; n < 16; ++n)
    {
      int32_t reference = 0;
      for (int32_t k = 0; k < 32; ++k)
      {
        reference += static_cast<int32_t>(host_a[m * 32 + k]) * static_cast<int32_t>(host_b[k * 16 + n]);
      }
      if (host_c[m * 16 + n] != reference)
      {
        if (failures < 12)
        {
          std::printf("m=%d n=%d got=%d expected=%d\n", m, n, host_c[m * 16 + n], reference);
        }
        ++failures;
      }
    }
  }

  cudaFree(device_c);
  cudaFree(device_b);
  cudaFree(device_a);
  std::printf("failures=%d\n", failures);
  return failures == 0 ? 0 : 1;
}

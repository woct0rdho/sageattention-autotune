#include <cuda_runtime.h>

#include "csrc/qattn_cutlass/qk_int8_sv_f16_bwd_kernel_cutlass_sm80.cuh"

#include <cstdint>
#include <cstdio>

namespace sageattention::qattn_cutlass_bwd::test {

using Traits = BwdTileTraits<64, 64, 128, 8>;

__global__ void test_kernel(int32_t *mismatch_count, int32_t *first_mismatch)
{
  __shared__ __align__(16) uint4 raw[cute::cosize_v<typename Traits::SmemLayoutScorePair>];

  const int32_t lane_id = static_cast<int32_t>(threadIdx.x);
  typename Traits::ScoreMMA score_mma;
  const auto thr_mma = score_mma.get_thread_slice(lane_id);
  const auto tiled_copy_c = make_tiled_copy_C_warpcontiguousN<2>(
    typename Traits::SmemCopyAtomScoreC{}, score_mma);
  const auto tiled_copy_a = cute::make_tiled_copy_A(typename Traits::SmemCopyAtomScoreA{}, score_mma);
  const auto thr_copy_c = tiled_copy_c.get_thread_slice(lane_id);
  const auto thr_copy_a = tiled_copy_a.get_thread_slice(lane_id);

  auto storage = cute::make_tensor(
    cute::make_smem_ptr(raw), typename Traits::SmemLayoutScorePair{});
  auto score = cute::recast<int8_t>(storage);
  constexpr auto pair_shape = cute::make_shape(
    cute::Int<Traits::kBlockN>{}, cute::Int<Traits::kScoreAtomK>{});
  auto pair = cute::local_tile(score, pair_shape, cute::make_coord(cute::_0{}, cute::_1{}));
  constexpr auto half_shape = cute::make_shape(
    cute::Int<Traits::kBlockN>{}, cute::Int<Traits::kBlockM>{});
  auto store_0 = make_transposed_tensor(
    cute::local_tile(pair, half_shape, cute::make_coord(cute::_0{}, cute::_0{})));
  auto store_1 = make_transposed_tensor(
    cute::local_tile(pair, half_shape, cute::make_coord(cute::_0{}, cute::_1{})));

  auto accumulator = cute::partition_fragment_C(score_mma, typename Traits::BlockMNShape{});
  auto source_0 = cute::make_fragment_like<int8_t>(accumulator);
  auto source_1 = cute::make_fragment_like<int8_t>(accumulator);
#pragma unroll
  for (int32_t idx = 0; idx < cute::size(source_0); ++idx)
  {
    source_0(idx) = static_cast<int8_t>(lane_id * 8 + idx);
    source_1(idx) = static_cast<int8_t>(37 + lane_id * 8 + idx);
  }

  auto direct = thr_mma.partition_fragment_A(pair);
  transpose_score_c_fragment_to_mma_a<0>(source_0, direct, lane_id);
  transpose_score_c_fragment_to_mma_a<1>(source_1, direct, lane_id);

  cute::copy(tiled_copy_c, thr_copy_c.retile_S(source_0), thr_copy_c.partition_D(store_0));
  cute::copy(tiled_copy_c, thr_copy_c.retile_S(source_1), thr_copy_c.partition_D(store_1));
  __syncthreads();

  auto canonical = thr_mma.partition_fragment_A(pair);
  cute::copy(tiled_copy_a, thr_copy_a.partition_S(pair), thr_copy_a.retile_D(canonical));
#pragma unroll
  for (int32_t idx = 0; idx < cute::size(canonical); ++idx)
  {
    if (direct(idx) != canonical(idx))
    {
      const int32_t mismatch_index = atomicAdd(mismatch_count, 1);
      if (mismatch_index == 0)
      {
        first_mismatch[0] = lane_id;
        first_mismatch[1] = idx;
        first_mismatch[2] = static_cast<int32_t>(direct(idx));
        first_mismatch[3] = static_cast<int32_t>(canonical(idx));
      }
    }
  }
}

} // namespace sageattention::qattn_cutlass_bwd::test

int main()
{
  int32_t *mismatch_count = nullptr;
  int32_t *first_mismatch = nullptr;
  cudaMallocManaged(&mismatch_count, sizeof(int32_t));
  cudaMallocManaged(&first_mismatch, 4 * sizeof(int32_t));
  *mismatch_count = 0;

  sageattention::qattn_cutlass_bwd::test::test_kernel<<<1, 32>>>(
    mismatch_count, first_mismatch);
  const cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess)
  {
    std::printf("CUDA error: %s\n", cudaGetErrorString(error));
    return 1;
  }
  if (*mismatch_count != 0)
  {
    std::printf(
      "mismatches=%d first=(lane=%d,index=%d,direct=%d,canonical=%d)\n",
      *mismatch_count,
      first_mismatch[0],
      first_mismatch[1],
      first_mismatch[2],
      first_mismatch[3]);
    return 1;
  }

  cudaFree(first_mismatch);
  cudaFree(mismatch_count);
  std::printf("C-pair to MMA-A fragment: 512/512 bytes matched\n");
  return 0;
}

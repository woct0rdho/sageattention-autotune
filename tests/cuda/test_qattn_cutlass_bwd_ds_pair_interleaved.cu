#include <cuda_runtime.h>

#include "csrc/qattn_cutlass/qk_int8_sv_f16_bwd_kernel_cutlass_sm80.cuh"

#include <cstdint>
#include <cstdio>

namespace sageattention::qattn_cutlass_bwd::test {

using Traits = BwdTileTraits<64, 64, 128, 8>;

__device__ __forceinline__ int8_t source_value(const int32_t warp_id, const int32_t m_half, const int32_t lane_id, const int32_t idx)
{
  uint32_t value = static_cast<uint32_t>(0x9e3779b9u + 0x6d2b79f5u * (warp_id * 2 + m_half + 1));
  value ^= static_cast<uint32_t>(lane_id + 1) * 0x85ebca6bu;
  value ^= static_cast<uint32_t>(idx + 1) * 0xc2b2ae35u;
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  return static_cast<int8_t>(static_cast<int32_t>(value % 251u) - 125);
}

__global__ void test_kernel(int32_t *mismatch_count, int32_t *first_mismatch)
{
  using Layout = typename Traits::SmemLayoutdSdKV;
  constexpr int32_t slot_offset = cute::cosize_v<Layout>;
  constexpr auto dS_tile_shape = cute::make_shape(
    cute::Int<Traits::kBlockM>{}, cute::Int<Traits::kScoreAtomK>{});
  constexpr auto score_store_shape = cute::make_shape(
    cute::Int<Traits::kBlockM>{}, cute::Int<Traits::kBlockN>{});

  __shared__ __align__(16) uint4 reference_raw[8 * slot_offset];
  __shared__ __align__(16) uint4 candidate_raw[8 * slot_offset];

  const int32_t lane_id = static_cast<int32_t>(threadIdx.x);
  const int32_t warp_id = static_cast<int32_t>(threadIdx.y);
  typename Traits::ScoreMMA score_mma;
  const auto thr_mma = score_mma.get_thread_slice(lane_id);
  const auto tiled_copy_c = make_tiled_copy_C_warpcontiguousN<2>(
    typename Traits::SmemCopyAtomScoreC{}, score_mma);
  const auto tiled_copy_a = cute::make_tiled_copy_A(typename Traits::SmemCopyAtomScoreA{}, score_mma);
  const auto thr_copy_c = tiled_copy_c.get_thread_slice(lane_id);
  const auto thr_copy_a = tiled_copy_a.get_thread_slice(lane_id);

  if (warp_id < 8)
  {
#pragma unroll
    for (int32_t m_half = 0; m_half < 2; ++m_half)
    {
      auto source_like = cute::partition_fragment_C(score_mma, typename Traits::BlockMNShape{});
      auto source = cute::make_fragment_like<int8_t>(source_like);
#pragma unroll
      for (int32_t idx = 0; idx < cute::size(source); ++idx)
      {
        source(idx) = source_value(warp_id, m_half, lane_id, idx);
      }

      const int32_t n_pair = warp_id / 2;
      const int32_t n_half = warp_id & 1;
      auto reference_slot = cute::make_tensor(
        cute::make_smem_ptr(reference_raw + (2 * n_pair + m_half) * slot_offset), Layout{});
      auto reference = cute::recast<int8_t>(reference_slot);
      auto reference_ds = cute::local_tile(reference, dS_tile_shape, cute::make_coord(cute::_0{}, cute::_0{}));
      auto reference_half = cute::local_tile(reference_ds, score_store_shape, cute::make_coord(cute::_0{}, n_half));

      auto candidate_slot = cute::make_tensor(
        cute::make_smem_ptr(candidate_raw + n_pair * slot_offset), Layout{});
      auto candidate = cute::recast<int8_t>(candidate_slot);
      auto candidate_ds = cute::local_tile(candidate, dS_tile_shape, cute::make_coord(cute::_0{}, m_half));
      auto candidate_half = cute::local_tile(candidate_ds, score_store_shape, cute::make_coord(cute::_0{}, n_half));

      const auto source_view = thr_copy_c.retile_S(source);
      cute::copy(tiled_copy_c, source_view, thr_copy_c.partition_D(reference_half));
      cute::copy(tiled_copy_c, source_view, thr_copy_c.partition_D(candidate_half));
    }
  }
  __syncthreads();

  if (warp_id == 8)
  {
#pragma unroll
    for (int32_t n_pair = 0; n_pair < 4; ++n_pair)
    {
#pragma unroll
      for (int32_t m_half = 0; m_half < 2; ++m_half)
      {
        auto reference_slot = cute::make_tensor(
          cute::make_smem_ptr(reference_raw + (2 * n_pair + m_half) * slot_offset), Layout{});
        auto reference = cute::recast<int8_t>(reference_slot);
        auto reference_ds = cute::local_tile(reference, dS_tile_shape, cute::make_coord(cute::_0{}, cute::_0{}));

        auto candidate_slot = cute::make_tensor(
          cute::make_smem_ptr(candidate_raw + n_pair * slot_offset), Layout{});
        auto candidate = cute::recast<int8_t>(candidate_slot);
        auto candidate_ds = cute::local_tile(candidate, dS_tile_shape, cute::make_coord(cute::_0{}, m_half));

        auto reference_fragment = thr_mma.partition_fragment_A(reference_ds);
        auto candidate_fragment = thr_mma.partition_fragment_A(candidate_ds);
        cute::copy(tiled_copy_a, thr_copy_a.partition_S(reference_ds), thr_copy_a.retile_D(reference_fragment));
        cute::copy(tiled_copy_a, thr_copy_a.partition_S(candidate_ds), thr_copy_a.retile_D(candidate_fragment));

        auto reference_bytes = cute::recast<int8_t>(reference_fragment);
        auto candidate_bytes = cute::recast<int8_t>(candidate_fragment);
#pragma unroll
        for (int32_t idx = 0; idx < cute::size(reference_bytes); ++idx)
        {
          if (reference_bytes(idx) != candidate_bytes(idx))
          {
            const int32_t mismatch_index = atomicAdd(mismatch_count, 1);
            if (mismatch_index == 0)
            {
              first_mismatch[0] = n_pair;
              first_mismatch[1] = m_half;
              first_mismatch[2] = lane_id;
              first_mismatch[3] = idx;
              first_mismatch[4] = static_cast<int32_t>(reference_bytes(idx));
              first_mismatch[5] = static_cast<int32_t>(candidate_bytes(idx));
            }
          }
        }
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
  cudaMallocManaged(&first_mismatch, 6 * sizeof(int32_t));
  *mismatch_count = 0;

  sageattention::qattn_cutlass_bwd::test::test_kernel<<<1, dim3(32, 9)>>>(
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
      "mismatches=%d first=(pair=%d,m_half=%d,lane=%d,index=%d,reference=%d,candidate=%d)\n",
      *mismatch_count,
      first_mismatch[0],
      first_mismatch[1],
      first_mismatch[2],
      first_mismatch[3],
      first_mismatch[4],
      first_mismatch[5]);
    return 1;
  }

  cudaFree(first_mismatch);
  cudaFree(mismatch_count);
  std::printf("pair-interleaved dS mirror: 2048/2048 MMA-A bytes matched\n");
  return 0;
}

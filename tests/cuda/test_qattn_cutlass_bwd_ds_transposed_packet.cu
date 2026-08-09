#include <cuda_runtime.h>

#include "csrc/qattn_cutlass/qk_int8_sv_f16_bwd_kernel_cutlass_sm80.cuh"

#include <cstdint>
#include <cstdio>

namespace sageattention::qattn_cutlass_bwd::transposed_probe {

using ScoreAtom = cute::MMA_Atom<cute::SM80_16x8x32_S32S8S8S32_TN>;
using ScoreMMA = cute::TiledMMA<
  ScoreAtom,
  cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>,
  cute::Tile<cute::_16, cute::_16, cute::_32>>;
using CToBSourceValueLayout = decltype(ScoreMMA{}.get_layoutC_TV());
using CToBValueSelector = cute::Layout<
  cute::Shape<cute::_4, cute::_2>,
  cute::Stride<cute::_1, cute::_8>>;
using CToBDestValueLayout = decltype(cute::composition(
  ScoreMMA{}.get_layoutB_TV(),
  cute::make_tile(cute::_, CToBValueSelector{})));

struct SM80_U32x2_REG_C_TO_INT8_B
{
  using SRegisters = uint32_t[2];
  using DRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void copy(const uint32_t &src0,
                                    const uint32_t &src1,
                                    uint32_t &dst0,
                                    uint32_t &dst1)
  {
#if defined(__CUDA_ARCH__)
    const int32_t lane_id = static_cast<int32_t>(threadIdx.x) & 31;
    const int32_t source_lane_base = (lane_id & ~3) + ((lane_id & 1) << 1);
    const uint32_t word0_lane0 = __shfl_sync(0xffffffffu, src0, source_lane_base);
    const uint32_t word0_lane1 = __shfl_sync(0xffffffffu, src0, source_lane_base + 1);
    const uint32_t word1_lane0 = __shfl_sync(0xffffffffu, src1, source_lane_base);
    const uint32_t word1_lane1 = __shfl_sync(0xffffffffu, src1, source_lane_base + 1);
    const bool select_word1 = ((lane_id >> 1) & 1) != 0;
    const uint32_t source0 = select_word1 ? word1_lane0 : word0_lane0;
    const uint32_t source1 = select_word1 ? word1_lane1 : word0_lane1;
    dst0 = __byte_perm(source0, source1, 0x5410u);
    dst1 = __byte_perm(source0, source1, 0x7632u);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use the SM80 register C-to-B copy without CUDA device support.");
#endif
  }
};

} // namespace sageattention::qattn_cutlass_bwd::transposed_probe

namespace cute {

template <>
struct Copy_Traits<sageattention::qattn_cutlass_bwd::transposed_probe::SM80_U32x2_REG_C_TO_INT8_B>
{
  using ThrID = Layout<_32>;
  using SrcLayout = decltype(recast_layout<int8_t, uint1_t>(
    sageattention::qattn_cutlass_bwd::transposed_probe::CToBSourceValueLayout{}));
  using DstLayout = decltype(recast_layout<int8_t, uint1_t>(
    sageattention::qattn_cutlass_bwd::transposed_probe::CToBDestValueLayout{}));
  using RefLayout = DstLayout;
};

} // namespace cute

namespace sageattention::qattn_cutlass_bwd::test {

using Traits = BwdTileTraits<64, 64, 128, 8>;
using ScoreMMA = transposed_probe::ScoreMMA;
using RegCopyAtomInt8CToB = cute::Copy_Atom<transposed_probe::SM80_U32x2_REG_C_TO_INT8_B, int8_t>;
using TransposedDSPacketLayout = cute::Layout<
  cute::Shape<cute::_4, cute::_2, cute::_32, cute::_2, cute::_2>,
  cute::Stride<cute::Int<256>, cute::Int<128>, cute::_4, cute::_2, cute::_1>>;

static_assert(cute::cosize_v<TransposedDSPacketLayout> == 1024);
static_assert(cute::size<0>(typename RegCopyAtomInt8CToB::ValLayoutSrc{}) == cute::_32{});
static_assert(cute::size<1>(typename RegCopyAtomInt8CToB::ValLayoutSrc{}) == cute::_8{});
static_assert(cute::size<0>(typename RegCopyAtomInt8CToB::ValLayoutDst{}) == cute::_32{});
static_assert(cute::size<1>(typename RegCopyAtomInt8CToB::ValLayoutDst{}) == cute::_8{});
static_assert(cute::cosize(transposed_probe::CToBSourceValueLayout{}) ==
              cute::cosize(transposed_probe::CToBDestValueLayout{}));

template <typename Storage, typename Fragment>
__device__ __forceinline__ void store_transposed_dS_packet(Storage &storage,
                                                           const Fragment &score_c_fragment,
                                                           const int32_t pair,
                                                           const int32_t m_half,
                                                           const int32_t source_half,
                                                           const int32_t lane_id)
{
  auto packet = cute::make_tensor(
    cute::make_smem_ptr(reinterpret_cast<uint32_t *>(storage.begin())),
    TransposedDSPacketLayout{});
  auto destination = packet(pair, m_half, lane_id, source_half, cute::_);
  auto converted = cute::make_tensor<int8_t>(cute::Int<8>{});
  cute::copy(RegCopyAtomInt8CToB{}, cute::coalesce(score_c_fragment), converted);
  cute::copy(cute::recast<uint64_t>(converted), cute::recast<uint64_t>(destination));
}

template <typename Storage, typename Fragment>
__device__ __forceinline__ void load_transposed_dS_packet(const Storage &storage,
                                                           Fragment &fragment,
                                                           const int32_t pair,
                                                           const int32_t m_half,
                                                           const int32_t lane_id)
{
  auto packet = cute::make_tensor(
    cute::make_smem_ptr(reinterpret_cast<const uint32_t *>(storage.begin())),
    TransposedDSPacketLayout{});
  const auto packet_slice = packet(pair, m_half, lane_id, cute::_, cute::_);
  const auto contiguous_packet = cute::make_tensor(packet_slice.data(), cute::make_layout(cute::_4{}));
  auto loaded = cute::make_tensor<uint32_t>(cute::_4{});
  cute::copy(cute::recast<cute::uint128_t>(contiguous_packet), cute::recast<cute::uint128_t>(loaded));
  const auto b_words = cute::make_tensor(
    loaded.data(),
    cute::Layout<cute::Shape<cute::_2, cute::_2>, cute::Stride<cute::_2, cute::_1>>{});
  auto fragment_words = cute::recast<uint32_t>(fragment);
  static_assert(cute::size(fragment_words) == 4, "INT8 MMA B fragments must contain four words");
  cute::copy(b_words, fragment_words);
}

struct PacketStorage
{
  alignas(16) cute::ArrayEngine<cute::uint128_t, 8 * cute::cosize_v<typename Traits::SmemLayoutdSdKV>> dS_dKV;
};

__device__ __forceinline__ int8_t source_value(const int32_t row, const int32_t col)
{
  return static_cast<int8_t>((row * 17 + col * 29) % 127 - 63);
}

__global__ void test_kernel(int32_t *mismatch_count, int32_t *first_mismatch)
{
  using Storage = PacketStorage;
  using Layout = typename Traits::SmemLayoutScorePair;
  __shared__ __align__(16) char shared_raw[sizeof(Storage)];
  __shared__ __align__(16) cute::uint128_t canonical_raw[cute::cosize_v<Layout>];
  auto &shared = *reinterpret_cast<Storage *>(shared_raw);

  const int32_t lane_id = static_cast<int32_t>(threadIdx.x);
  const int32_t warp_id = static_cast<int32_t>(threadIdx.y);
  ScoreMMA score_mma;
  const auto tiled_copy_score_b = cute::make_tiled_copy_B(typename Traits::SmemCopyAtomScoreB{}, score_mma);
  const auto thr_copy_score_b = tiled_copy_score_b.get_thread_slice(lane_id);
  constexpr auto dS_shape = cute::make_shape(cute::Int<16>{}, cute::Int<32>{});
  constexpr auto b_tile_shape = cute::make_shape(cute::Int<16>{}, cute::Int<32>{});

  if (warp_id < 8)
  {
    auto source_like = cute::partition_fragment_C(score_mma, cute::Shape<cute::_16, cute::_16>{});
    auto source_0 = cute::make_fragment_like<int8_t>(source_like);
    auto source_1 = cute::make_fragment_like<int8_t>(source_like);
    const auto c_layout = score_mma.get_layoutC_TV();
#pragma unroll
    for (int32_t idx = 0; idx < cute::size(source_0); ++idx)
    {
      const int32_t c_linear = static_cast<int32_t>(c_layout(cute::make_coord(lane_id, idx)));
      const int32_t row = c_linear & 15;
      const int32_t local_col = c_linear >> 4;
      const int32_t global_col = (warp_id & 1) * 16 + (warp_id / 2) * 32 + local_col;
      source_0(idx) = source_value(row, global_col);
      source_1(idx) = source_value(row + 16, global_col);
    }

    store_transposed_dS_packet(
      shared.dS_dKV,
      source_0,
      warp_id / 2,
      0,
      warp_id & 1,
      lane_id);
    store_transposed_dS_packet(
      shared.dS_dKV,
      source_1,
      warp_id / 2,
      1,
      warp_id & 1,
      lane_id);
  }
  __syncthreads();

  if (warp_id == 8)
  {
    auto canonical_storage = cute::make_tensor(cute::make_smem_ptr(canonical_raw), Layout{});
    auto canonical_bytes = cute::recast<int8_t>(canonical_storage);
    auto canonical_tile = cute::local_tile(canonical_bytes, b_tile_shape, cute::make_coord(cute::_0{}, cute::_0{}));
    auto loaded = cute::make_tensor<typename ScoreMMA::FrgTypeB>(
      cute::partition_shape_B(score_mma, dS_shape));
    auto canonical = cute::make_tensor<typename ScoreMMA::FrgTypeB>(
      cute::partition_shape_B(score_mma, dS_shape));
    auto loaded_words = cute::recast<int8_t>(loaded);
    auto canonical_words = cute::recast<int8_t>(canonical);

#pragma unroll
    for (int32_t n_pair = 0; n_pair < 4; ++n_pair)
    {
#pragma unroll
      for (int32_t m_half = 0; m_half < 2; ++m_half)
      {
        if (lane_id == 0)
        {
#pragma unroll
          for (int32_t row = 0; row < 16; ++row)
          {
#pragma unroll
            for (int32_t col = 0; col < 32; ++col)
            {
              canonical_tile(row, col) = source_value(
                row + m_half * 16,
                n_pair * 32 + col);
            }
          }
        }
        __syncthreads();
        load_transposed_dS_packet(
          shared.dS_dKV,
          loaded,
          n_pair,
          m_half,
          lane_id);
        cute::copy(
          tiled_copy_score_b,
          thr_copy_score_b.partition_S(canonical_tile),
          thr_copy_score_b.retile_D(canonical));
#pragma unroll
        for (int32_t idx = 0; idx < cute::size(loaded_words); ++idx)
        {
          if (loaded_words(idx) != canonical_words(idx))
          {
            const int32_t mismatch_index = atomicAdd(mismatch_count, 1);
            if (mismatch_index == 0)
            {
              first_mismatch[0] = n_pair;
              first_mismatch[1] = m_half;
              first_mismatch[2] = lane_id;
              first_mismatch[3] = idx;
              first_mismatch[4] = static_cast<int32_t>(loaded_words(idx));
              first_mismatch[5] = static_cast<int32_t>(canonical_words(idx));
            }
          }
        }
      }
      __syncthreads();
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

  sageattention::qattn_cutlass_bwd::test::test_kernel<<<1, dim3(32, 9)>>> (
    mismatch_count,
    first_mismatch);
  const cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess)
  {
    std::printf("CUDA error: %s\n", cudaGetErrorString(error));
    return 1;
  }
  if (*mismatch_count != 0)
  {
    std::printf(
      "mismatches=%d first=(pair=%d,m_half=%d,lane=%d,index=%d,loaded=%d,canonical=%d)\n",
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
  std::printf("transposed dS packet: all 2048 fragment bytes matched\n");
  return 0;
}

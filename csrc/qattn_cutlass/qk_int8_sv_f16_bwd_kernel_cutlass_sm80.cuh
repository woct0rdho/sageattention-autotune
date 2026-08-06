#pragma once

#include "int8_transposed_ldsm_cute.cuh"

#include <cuda_fp16.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/algorithm/clear.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

#include <cstdint>

namespace sageattention::qattn_cutlass_bwd {

template <typename Element>
constexpr int32_t packed_elements()
{
  static_assert(sizeof(cute::uint128_t) % sizeof(Element) == 0, "Packed element type must divide uint128_t");
  return static_cast<int32_t>(sizeof(cute::uint128_t) / sizeof(Element));
}

template <int32_t Extent>
CUTE_HOST_DEVICE constexpr auto make_smem_swizzle()
{
  if constexpr (Extent == 4)
  {
    return cute::Swizzle<2, 0, 3>{};
  }
  else
  {
    static_assert(Extent % 8 == 0, "Backward smem stride-one extent must be 4 for 64B swizzle or a multiple of 8 for 128B swizzle");
    constexpr int32_t row_bit_shift = static_cast<int32_t>(cute::log_2(static_cast<uint32_t>(Extent)));
    return cute::Swizzle<3, 0, row_bit_shift>{};
  }
}

template <int32_t Extent, typename Layout>
CUTE_HOST_DEVICE constexpr auto make_smem_layout(const Layout layout)
{
  return cute::composition(make_smem_swizzle<Extent>(), layout);
}

template <int32_t Rows, int32_t Cols>
CUTE_HOST_DEVICE constexpr auto make_plain_smem_matrix_layout()
{
  return cute::make_layout(cute::make_shape(cute::Int<Rows>{}, cute::Int<Cols>{}), cute::LayoutRight{});
}

template <int32_t Rows, int32_t Cols>
CUTE_HOST_DEVICE constexpr auto make_smem_matrix_layout()
{
  return make_smem_layout<Cols>(make_plain_smem_matrix_layout<Rows, Cols>());
}

template <int32_t Rows, int32_t Cols, typename CopyAtom>
struct GmemTiledCopyContract
{
  static constexpr int32_t kLineLanes = Cols == 4 ? 4 : 8;
  static_assert(Cols % kLineLanes == 0, "Bwd gmem head copy columns must divide the lane layout");
  static constexpr int32_t kRowsPerIter = 32 / kLineLanes;
  static_assert(Rows % kRowsPerIter == 0, "Bwd gmem head copy rows must divide the warp layout");
  using TiledCopy = decltype(cute::make_tiled_copy(
    CopyAtom{},
    cute::make_layout(cute::make_shape(cute::Int<kRowsPerIter>{}, cute::Int<kLineLanes>{}), cute::LayoutRight{}),
    cute::make_layout(cute::make_shape(cute::Int<Rows / kRowsPerIter>{}, cute::Int<Cols / kLineLanes>{}), cute::LayoutRight{})));
};

template <int32_t HeadDim, int32_t CtaM = 16, int32_t CtaN = 16, int32_t NumWarps = 1>
struct BwdTileTraits
{
  static_assert(HeadDim == 64 || HeadDim == 128, "CUTLASS qattn bwd currently supports head_dim 64 or 128");
  static_assert(CtaM > 0 && CtaN > 0 && NumWarps > 0, "CUTLASS qattn bwd CTA shape must be positive");

  using ScoreMMAAtom = cute::MMA_Atom<cute::SM80_16x8x32_S32S8S8S32_TN>;
  using HalfMMAAtom = cute::MMA_Atom<cute::SM80_16x8x16_F32F16F16F32_TN>;
  using ScoreAtomShapeMNK = typename ScoreMMAAtom::Shape_MNK;
  using HalfAtomShapeMNK = typename HalfMMAAtom::Shape_MNK;

  static constexpr int32_t kHeadDim = HeadDim;
  static constexpr int32_t kAtomM = cute::size<0>(HalfAtomShapeMNK{});
  static constexpr int32_t kAtomN = cute::size<1>(HalfAtomShapeMNK{});
  static constexpr int32_t kAtomK = cute::size<2>(HalfAtomShapeMNK{});
  static constexpr int32_t kScoreAtomK = cute::size<2>(ScoreAtomShapeMNK{});
  static constexpr int32_t kBlockM = kAtomM;
  static constexpr int32_t kBlockN = 2 * kAtomN;
  static constexpr int32_t kBlockK = kAtomK;
  static constexpr int32_t kCtaM = CtaM;
  static constexpr int32_t kCtaN = CtaN;
  static constexpr int32_t kNumWarps = NumWarps;
  static constexpr int32_t kCtaMMicroTiles = kCtaM / kBlockM;
  static constexpr int32_t kCtaNMicroTiles = kCtaN / kBlockN;
  static constexpr int32_t kSmemWarps = kNumWarps < kCtaNMicroTiles ? kNumWarps : kCtaNMicroTiles;
  static constexpr int32_t kMStageLoadRows = NumWarps == 1 ? kBlockM : (HeadDim == 64 ? 8 : 4);
  static constexpr int32_t kMStageLoadWarps = kBlockM / kMStageLoadRows;
  static constexpr int32_t kCtaNLoadRows = kCtaN / kNumWarps;
  static_assert(kCtaNMicroTiles % kSmemWarps == 0, "Backward N microtiles must divide active warp waves");

  static_assert(cute::size<0>(ScoreAtomShapeMNK{}) == kAtomM && cute::size<1>(ScoreAtomShapeMNK{}) == kAtomN,
                "Backward score MMA atom must match logical M/N atom shape");
  static_assert(HeadDim % kBlockK == 0, "head_dim must be divisible by the MMA K tile");
  static_assert(HeadDim % kScoreAtomK == 0, "head_dim must be divisible by the score MMA K tile");
  static_assert(kCtaM % kBlockM == 0 && kCtaN % kBlockN == 0, "Backward CTA tile must be a multiple of the micro MMA tile");
  static_assert(kCtaM % kNumWarps == 0 && kCtaN % kNumWarps == 0, "Backward CTA tile rows must divide the warp copy schedule");
  static_assert(kBlockM % kMStageLoadRows == 0, "Backward microtile rows must divide the cooperative Q/dO warp schedule");
  static_assert(kNumWarps >= kMStageLoadWarps, "Backward CTA must have enough warps to stage Q/dO cooperatively");

  using ScoreMMA = cute::TiledMMA<ScoreMMAAtom,
                                  cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>,
                                  cute::Tile<cute::Int<kBlockM>, cute::Int<kBlockN>, cute::Int<kScoreAtomK>>>;
  using HalfMMA = cute::TiledMMA<HalfMMAAtom,
                                 cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>,
                                 cute::Tile<cute::Int<kBlockM>, cute::Int<kBlockN>, cute::Int<kBlockK>>>;
  static constexpr int32_t kHalfGmemVecElems = packed_elements<half>();
  static constexpr int32_t kdKVGmemThreadsPerRow = kBlockK / kHalfGmemVecElems;
  static_assert(kBlockK % kHalfGmemVecElems == 0, "dKV gmem rows must be 128-bit aligned");
  using GmemCopyAtomAsync = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, cute::uint128_t>;
  using GmemCopyAtomdKV = cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, cute::uint128_t>;
  using GmemTiledCopydKV = decltype(cute::make_tiled_copy(
    GmemCopyAtomdKV{},
    cute::Layout<cute::Shape<cute::Int<32 / kdKVGmemThreadsPerRow>, cute::Int<kdKVGmemThreadsPerRow>>, cute::Stride<cute::Int<kdKVGmemThreadsPerRow>, cute::_1>>{},
    cute::Layout<cute::Shape<cute::_1, cute::_1>>{}));
  using SmemCopyAtomScoreA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, int8_t>;
  using SmemCopyAtomScoreB = cute::Copy_Atom<cute::SM75_U32x2_LDSM_N, int8_t>;
  using SmemCopyAtomScoreC = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<64>, int8_t>;
  using SmemCopyAtomdKVC = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, half>;
  using SmemCopyAtomTransposedB = qattn_cutlass::SmemCopyAtomInt8TransposedB;
  using SmemCopyAtomdPA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, half>;
  using SmemCopyAtomdPB = cute::Copy_Atom<cute::SM75_U32x2_LDSM_N, half>;
  using BlockMNShape = cute::Shape<cute::Int<kBlockM>, cute::Int<kBlockN>>;

  static constexpr int32_t kInt8LoadVecCols = kHeadDim / packed_elements<int8_t>();
  static constexpr int32_t kHalfLoadVecCols = kHeadDim / packed_elements<half>();
  static constexpr int32_t kFloatLoadVecCols = kHeadDim / packed_elements<float>();
  static constexpr int32_t kScoreVecCols = kScoreAtomK / packed_elements<int8_t>();
  using GmemCopyAtomdQAccum = cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, cute::uint128_t>;
  using GmemCopyAtomdQ = cute::Copy_Atom<cute::UniversalCopy<uint64_t>, uint64_t>;
  using GmemTiledCopydQZero = typename GmemTiledCopyContract<8, kFloatLoadVecCols, GmemCopyAtomdQAccum>::TiledCopy;
  using GmemTiledCopydQAccum = typename GmemTiledCopyContract<4, kFloatLoadVecCols, GmemCopyAtomdQAccum>::TiledCopy;
  using GmemTiledCopydQ = typename GmemTiledCopyContract<4, kFloatLoadVecCols, GmemCopyAtomdQ>::TiledCopy;
  using GmemTiledCopyQ = typename GmemTiledCopyContract<kMStageLoadRows, kInt8LoadVecCols, GmemCopyAtomAsync>::TiledCopy;
  using GmemTiledCopydO = typename GmemTiledCopyContract<kMStageLoadRows, kInt8LoadVecCols, GmemCopyAtomAsync>::TiledCopy;
  using GmemTiledCopyK = typename GmemTiledCopyContract<kCtaNLoadRows, kInt8LoadVecCols, GmemCopyAtomAsync>::TiledCopy;
  using GmemTiledCopyV = typename GmemTiledCopyContract<kCtaNLoadRows, kHalfLoadVecCols, GmemCopyAtomAsync>::TiledCopy;
  using GmemTiledCopydOFp16 = typename GmemTiledCopyContract<kMStageLoadRows, kHalfLoadVecCols, GmemCopyAtomAsync>::TiledCopy;

  using SmemLayoutQ = decltype(make_smem_matrix_layout<2 * kBlockM, kInt8LoadVecCols>());
  using SmemLayoutdO = decltype(make_smem_matrix_layout<2 * kBlockM, kInt8LoadVecCols>());
  using SmemLayoutK = decltype(make_smem_matrix_layout<kCtaN, kInt8LoadVecCols>());
  using SmemLayoutdOFp16 = decltype(make_smem_matrix_layout<kBlockM, kHalfLoadVecCols>());
  using SmemLayoutV = decltype(make_smem_matrix_layout<kCtaN, kHalfLoadVecCols>());
  using SmemLayoutScorePair = decltype(make_smem_matrix_layout<kBlockN, 2 * kScoreVecCols>());
  using SmemLayoutdSdKV = decltype(make_smem_matrix_layout<kBlockM, 2 * kScoreVecCols>());
  using SmemLayoutdKV = decltype(cute::composition(cute::Swizzle<1, 3, 3>{}, make_plain_smem_matrix_layout<kBlockN, kBlockK>()));
  using SmemLayoutM = decltype(cute::make_layout(cute::make_shape(cute::Int<kBlockM>{}), cute::make_stride(cute::_1{})));
  using SmemLayoutN = decltype(cute::make_layout(cute::make_shape(cute::Int<kBlockN>{}), cute::make_stride(cute::_1{})));
};

template <typename Traits>
struct QdOMmaBPacketStorage
{
};

template <typename Traits>
  requires (Traits::kCtaN == 128)
struct QdOMmaBPacketStorage<Traits>
{
  alignas(16) cute::ArrayEngine<cute::uint128_t, cute::cosize_v<typename Traits::SmemLayoutQ>> q_i8_mma_b;
  alignas(16) cute::ArrayEngine<cute::uint128_t, cute::cosize_v<typename Traits::SmemLayoutdO>> dO_i8_mma_b;
};

template <typename Traits>
struct VOrQdOMmaBStorage
{
  alignas(16) cute::ArrayEngine<cute::uint128_t, cute::cosize_v<typename Traits::SmemLayoutV>> v;
};

template <typename Traits>
  requires (Traits::kCtaN == 128)
struct VOrQdOMmaBStorage<Traits>
{
  union
  {
    alignas(16) cute::ArrayEngine<cute::uint128_t, cute::cosize_v<typename Traits::SmemLayoutV>> v;
    QdOMmaBPacketStorage<Traits> QdO_mma_b;
  };
};

template <typename Traits>
struct SharedStorage2DWarp
  : VOrQdOMmaBStorage<Traits>
{
  static_assert(
    Traits::kCtaM == 64 &&
      ((Traits::kNumWarps == 8 && (Traits::kCtaN == 64 || Traits::kCtaN == 128)) ||
       (Traits::kNumWarps == 16 && Traits::kCtaN == 128)),
    "2D warp storage is specialized for the 64-row, two-M-half schedule");
  using SmemLayoutdOFp16Pair = decltype(make_smem_matrix_layout<2 * Traits::kBlockM, Traits::kHalfLoadVecCols>());
  using SmemLayoutMPair = decltype(cute::make_layout(
    cute::make_shape(cute::Int<2 * Traits::kBlockM>{}), cute::make_stride(cute::_1{})));

  struct WarpScratchStorage
  {
    alignas(16) cute::ArrayEngine<cute::uint128_t, Traits::kSmemWarps * cute::cosize_v<typename Traits::SmemLayoutdSdKV>> dS_dKV;
  };

  alignas(16) cute::ArrayEngine<cute::uint128_t, cute::cosize_v<typename Traits::SmemLayoutQ>> q_i8;
  alignas(16) cute::ArrayEngine<cute::uint128_t, cute::cosize_v<typename Traits::SmemLayoutdO>> dO_i8;
  alignas(16) cute::ArrayEngine<cute::uint128_t, cute::cosize_v<typename Traits::SmemLayoutK>> k_i8;
  alignas(16) cute::ArrayEngine<cute::uint128_t, cute::cosize_v<typename Traits::SmemLayoutK>> k_i8_mma_b;
  alignas(16) cute::ArrayEngine<cute::uint128_t, cute::cosize_v<SmemLayoutdOFp16Pair>> dO_fp16_pair;
  WarpScratchStorage warp_scratch;
  alignas(16) cute::ArrayEngine<cute::uint128_t, Traits::kSmemWarps * cute::cosize_v<typename Traits::SmemLayoutScorePair>> score_pair_i8;
  cute::ArrayEngine<float, cute::cosize_v<SmemLayoutMPair>> lse;
  cute::ArrayEngine<float, cute::cosize_v<SmemLayoutMPair>> delta;
  float p_scale[Traits::kNumWarps];
};

template <typename Storage, typename Layout>
__device__ __forceinline__ auto make_smem_tensor(Storage& storage, const Layout layout)
{
  return cute::make_tensor(cute::make_smem_ptr(storage.begin()), layout);
}

template <typename Storage, typename Layout>
__device__ __forceinline__ auto make_smem_tensor(Storage& storage, const Layout layout, const int32_t offset)
{
  return cute::make_tensor(cute::make_smem_ptr(storage.begin() + offset), layout);
}

struct Params {
  int32_t head_dim;
  int32_t batch_size;
  int32_t seq_len;
  int32_t num_heads;
  int32_t bwd_block_m;
  int32_t bwd_block_n;
  int32_t blk_q;
  int32_t blk_k;
  int32_t stride_bz_q;
  int32_t stride_seq_q;
  int32_t stride_h_q;
  int32_t stride_bz_k;
  int32_t stride_seq_k;
  int32_t stride_h_k;
  int32_t stride_bz_v;
  int32_t stride_seq_v;
  int32_t stride_h_v;
  int32_t stride_bz_o;
  int32_t stride_seq_o;
  int32_t stride_h_o;
  int32_t stride_bz_dO;
  int32_t stride_seq_dO;
  int32_t stride_h_dO;
  int32_t stride_bz_dQ;
  int32_t stride_seq_dQ;
  int32_t stride_h_dQ;
  int32_t stride_bz_dK;
  int32_t stride_seq_dK;
  int32_t stride_h_dK;
  int32_t stride_bz_dV;
  int32_t stride_seq_dV;
  int32_t stride_h_dV;
  int32_t stride_bz_q_scale;
  int32_t stride_h_q_scale;
  int32_t stride_bz_k_scale;
  int32_t stride_h_k_scale;
};

template <typename ScaleView>
__device__ __forceinline__ float query_scale_for_row(const ScaleView scale, const Params &params, const int32_t row)
{
  return scale(row / params.blk_q);
}

template <typename ScaleView>
__device__ __forceinline__ float key_scale_for_row(const ScaleView scale, const Params &params, const int32_t row)
{
  return scale(row / params.blk_k);
}

template <int32_t HeadDim, typename T>
__device__ __forceinline__ auto make_head_matrix_view(T *const base_ptr,
                                                      const Params &params,
                                                      const int32_t batch,
                                                      const int32_t head,
                                                      const int32_t stride_bz,
                                                      const int32_t stride_seq,
                                                      const int32_t stride_h)
{
  const auto layout = cute::make_layout(cute::make_shape(params.seq_len, cute::Int<HeadDim>{}),
                                        cute::make_stride(stride_seq, cute::_1{}));
  return cute::make_tensor(cute::make_gmem_ptr(base_ptr + batch * stride_bz + head * stride_h), layout);
}

template <int32_t HeadDim, typename T>
__device__ __forceinline__ auto make_workspace_matrix_view(T *const base_ptr,
                                                           const Params &params,
                                                           const int32_t batch,
                                                           const int32_t head)
{
  const auto layout = cute::make_layout(cute::make_shape(params.seq_len, cute::Int<HeadDim>{}),
                                        cute::make_stride(cute::Int<HeadDim>{}, cute::_1{}));
  return cute::make_tensor(cute::make_gmem_ptr(base_ptr + (batch * params.num_heads + head) * params.seq_len * HeadDim), layout);
}

template <typename T>
__device__ __forceinline__ auto make_head_vector_view(T *const base_ptr,
                                                      const Params &params,
                                                      const int32_t batch,
                                                      const int32_t head)
{
  return cute::make_tensor(cute::make_gmem_ptr(base_ptr + (batch * params.num_heads + head) * params.seq_len),
                           cute::make_layout(params.seq_len));
}

template <typename T>
__device__ __forceinline__ auto make_head_strided_vector_view(T *const base_ptr,
                                                              const int32_t batch,
                                                              const int32_t head,
                                                              const int32_t stride_bz,
                                                              const int32_t stride_h,
                                                              const int32_t extent)
{
  return cute::make_tensor(cute::make_gmem_ptr(base_ptr + batch * stride_bz + head * stride_h),
                           cute::make_layout(extent));
}

__device__ __forceinline__ float warp_reduce_max(float value)
{
  const uint32_t max_bits = __reduce_max_sync(0xffffffffu, __float_as_uint(value));
  return __uint_as_float(max_bits);
}

__device__ __forceinline__ int8_t round_to_int8(const float value)
{
  int32_t result;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(result) : "f"(value));
  return static_cast<int8_t>(result);
}

// 127.5 - 2**-12 leaves a round-to-nearest margin without changing the scale materially.
inline constexpr float kInt8ScaleInv = 0x1.010122p-7f;
inline constexpr float kInt8ScaleFloor = 0x1.0p-126f;
inline constexpr float kdSPredictorGuard = 0x1.8p0f;

struct ThreadCoord
{
  int32_t lane_id;
  int32_t warp_id;
};

__device__ __forceinline__ ThreadCoord make_thread_coord()
{
  return {static_cast<int32_t>(threadIdx.x), static_cast<int32_t>(threadIdx.y)};
}

template <int32_t MmaN, typename CopyAtom, typename TiledMMA>
CUTE_HOST_DEVICE constexpr auto make_tiled_copy_C_warpcontiguousN(const CopyAtom copy_atom, const TiledMMA tiled_mma)
{
  constexpr int32_t tile_m = decltype(tiled_mma.template tile_size_mnk<0>())::value;
  constexpr int32_t tile_n = decltype(tiled_mma.template tile_size_mnk<1>())::value;
  using AtomShapeMNK = typename TiledMMA::AtomShape_MNK;
  constexpr int32_t atom_n = decltype(cute::size<1>(AtomShapeMNK{}))::value;
  constexpr int32_t n_warps_n = tile_n / atom_n / 2;
  static_assert(n_warps_n > 0, "Bwd warp-contiguous C copy needs at least two N atom groups");
  constexpr int32_t mma_stride_n = MmaN * atom_n * 2;
  const auto c_tile = cute::make_tile(
    cute::make_layout(cute::Int<tile_m>{}),
    cute::Layout<cute::Shape<cute::Int<atom_n>, cute::Int<n_warps_n>, cute::_2>, cute::Stride<cute::_1, cute::Int<mma_stride_n>, cute::_8>>{});
  return cute::make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutC_TV(), c_tile);
}

template <int32_t Rows, int32_t VecCols, typename TiledCopy, typename Gmem, typename Smem>
__device__ __forceinline__ void load_head_tile_with_contract(const TiledCopy tiled_copy,
                                                              const Gmem gmem_vec,
                                                              Smem smem,
                                                              const int32_t row_base,
                                                              const int32_t lane_id)
{
  constexpr auto vec_tile_shape = cute::make_shape(cute::Int<Rows>{}, cute::Int<VecCols>{});
  const auto tile_coord = cute::make_coord(row_base / Rows, cute::_0{});
  const auto gmem_tile = cute::local_tile(gmem_vec, vec_tile_shape, tile_coord);
  const auto coord_tile = cute::local_tile(cute::make_identity_tensor(cute::shape(gmem_vec)), vec_tile_shape, tile_coord);
  const auto thr_copy = tiled_copy.get_thread_slice(lane_id);
  const auto thr_gmem = thr_copy.partition_S(gmem_tile);
  const auto thr_coord = thr_copy.partition_S(coord_tile);
  auto thr_smem = thr_copy.partition_D(smem);
  const auto thr_pred = cute::lazy::transform(thr_coord, [&](const auto &coord) {
    return cute::get<0>(coord) < cute::size<0>(gmem_vec) && cute::get<1>(coord) < cute::size<1>(gmem_vec);
  });
  cute::copy_if(tiled_copy, thr_pred, thr_gmem, thr_smem);
}

template <typename Traits, typename Gmem, typename Smem>
__device__ __forceinline__ void load_q_tile(const Gmem gmem, Smem smem, const int32_t row_base, const int32_t lane_id)
{
  typename Traits::GmemTiledCopyQ tiled_copy;
  load_head_tile_with_contract<Traits::kMStageLoadRows, Traits::kInt8LoadVecCols>(
    tiled_copy, cute::recast<cute::uint128_t>(gmem), smem, row_base, lane_id);
}

template <typename Traits, typename Gmem, typename Smem>
__device__ __forceinline__ void load_k_tile(const Gmem gmem, Smem smem, const int32_t row_base, const int32_t lane_id)
{
  typename Traits::GmemTiledCopyK tiled_copy;
  load_head_tile_with_contract<Traits::kCtaNLoadRows, Traits::kInt8LoadVecCols>(
    tiled_copy, cute::recast<cute::uint128_t>(gmem), smem, row_base, lane_id);
}

template <typename Traits, typename Gmem, typename Smem>
__device__ __forceinline__ void load_v_tile(const Gmem gmem, Smem smem, const int32_t row_base, const int32_t lane_id)
{
  typename Traits::GmemTiledCopyV tiled_copy;
  load_head_tile_with_contract<Traits::kCtaNLoadRows, Traits::kHalfLoadVecCols>(
    tiled_copy, cute::recast<cute::uint128_t>(gmem), smem, row_base, lane_id);
}

template <typename Traits, typename Gmem, typename Smem>
__device__ __forceinline__ void load_dO_int8_tile(const Gmem gmem, Smem smem, const int32_t row_base, const int32_t lane_id)
{
  typename Traits::GmemTiledCopydO tiled_copy;
  load_head_tile_with_contract<Traits::kMStageLoadRows, Traits::kInt8LoadVecCols>(
    tiled_copy, cute::recast<cute::uint128_t>(gmem), smem, row_base, lane_id);
}

template <typename Traits, typename Gmem, typename Smem>
__device__ __forceinline__ void load_dO_fp16_tile(const Gmem gmem, Smem smem, const int32_t row_base, const int32_t lane_id)
{
  typename Traits::GmemTiledCopydOFp16 tiled_copy;
  load_head_tile_with_contract<Traits::kMStageLoadRows, Traits::kHalfLoadVecCols>(
    tiled_copy, cute::recast<cute::uint128_t>(gmem), smem, row_base, lane_id);
}

template <typename Traits,
          typename GQ,
          typename GdOInt8,
          typename GdO,
          typename SQStorage,
          typename SdOInt8Storage,
          typename SdOFp16Storage>
__device__ __forceinline__ void load_q_dO_pair(const GQ gQ,
                                                const GdOInt8 gdOInt8,
                                                const GdO gdO,
                                                SQStorage sQStorage,
                                                SdOInt8Storage sdOInt8Storage,
                                                SdOFp16Storage sdOFp16Storage,
                                                const int32_t m_pair_base,
                                                const int32_t loader_warp,
                                                const int32_t lane_id)
{
  constexpr auto int8_pair_load_shape = cute::make_shape(
    cute::Int<Traits::kMStageLoadRows>{}, cute::Int<Traits::kInt8LoadVecCols>{});
  constexpr auto half_pair_load_shape = cute::make_shape(
    cute::Int<Traits::kMStageLoadRows>{}, cute::Int<Traits::kHalfLoadVecCols>{});
  const int32_t m_load_base = m_pair_base + loader_warp * Traits::kMStageLoadRows;
#pragma unroll
  for (int32_t pair_half = 0; pair_half < 2; ++pair_half)
  {
    const int32_t load_warp = loader_warp + pair_half * Traits::kMStageLoadWarps;
    auto sQLoad = cute::local_tile(sQStorage, int8_pair_load_shape, cute::make_coord(load_warp, cute::_0{}));
    auto sdOInt8Load = cute::local_tile(sdOInt8Storage, int8_pair_load_shape, cute::make_coord(load_warp, cute::_0{}));
    auto sdOFp16Load = cute::local_tile(sdOFp16Storage, half_pair_load_shape, cute::make_coord(load_warp, cute::_0{}));
    const int32_t row_base = m_load_base + pair_half * Traits::kBlockM;
    load_q_tile<Traits>(gQ, sQLoad, row_base, lane_id);
    load_dO_int8_tile<Traits>(gdOInt8, sdOInt8Load, row_base, lane_id);
    load_dO_fp16_tile<Traits>(gdO, sdOFp16Load, row_base, lane_id);
  }
  cute::cp_async_fence();
}

template <typename GLse, typename GDelta, typename SLse, typename SDelta>
__device__ __forceinline__ void load_row_state_pair(const GLse gLse,
                                                     const GDelta gDelta,
                                                     SLse sLse,
                                                     SDelta sDelta,
                                                     const int32_t m_pair_base,
                                                     const int32_t seq_len,
                                                     const int32_t lane_id)
{
  const int32_t row = m_pair_base + lane_id;
  if (row < seq_len)
  {
    sLse(lane_id) = gLse(row);
    sDelta(lane_id) = gDelta(row);
  }
  else
  {
    sLse(lane_id) = 0.0f;
    sDelta(lane_id) = 0.0f;
  }
}

template <typename Storage, typename Fragment>
__device__ __forceinline__ void store_packed_mma_b_fragment(Storage &storage,
                                                            const Fragment &fragment,
                                                            const int32_t tile,
                                                            const int32_t lane_id)
{
  auto words = cute::recast<uint32_t>(fragment);
  static_assert(cute::size(words) == 4, "INT8 MMA B fragments must contain four words");
  auto *const packed = reinterpret_cast<uint32_t *>(storage.begin());
#pragma unroll
  for (int32_t word = 0; word < 4; ++word)
  {
    packed[(tile * 4 + word) * warpSize + lane_id] = words(word);
  }
}

template <typename Storage, typename Fragment>
__device__ __forceinline__ void load_packed_mma_b_fragment(const Storage &storage,
                                                           Fragment &fragment,
                                                           const int32_t tile,
                                                           const int32_t lane_id)
{
  auto words = cute::recast<uint32_t>(fragment);
  static_assert(cute::size(words) == 4, "INT8 MMA B fragments must contain four words");
  const auto *const packed = reinterpret_cast<const uint32_t *>(storage.begin());
#pragma unroll
  for (int32_t word = 0; word < 4; ++word)
  {
    words(word) = packed[(tile * 4 + word) * warpSize + lane_id];
  }
}

template <typename Tensor>
__device__ __forceinline__ auto make_transposed_tensor(Tensor tensor)
{
  CUTE_STATIC_ASSERT_V(cute::rank(tensor) == cute::_2{});
  const auto transpose = cute::make_layout(cute::make_shape(cute::size<1>(tensor), cute::size<0>(tensor)), cute::GenRowMajor{});
  return cute::make_tensor(tensor.data(), cute::composition(tensor.layout(), transpose));
}

template <typename Traits, typename Accum, typename Gmem>
__device__ __forceinline__ void store_dKV_fragment_coalesced(const Accum &accum_frag,
                                                             const int32_t dim_block,
                                                             Gmem dst,
                                                             half *const smem_stage,
                                                             const int32_t row_base,
                                                             const int32_t seq_len,
                                                             const int32_t lane_id)
{
  static_assert(Traits::kBlockK == 16, "dKV epilogue stage swizzle assumes 16-column MMA K tiles");
  typename Traits::ScoreMMA score_mma;
  auto acc_like = cute::partition_fragment_C(score_mma, typename Traits::BlockMNShape{});
  auto half_frag = cute::make_fragment_like<half>(acc_like);
#pragma unroll
  for (int32_t idx = 0; idx < cute::size(half_frag); ++idx)
  {
    half_frag(idx) = __float2half_rn(accum_frag(dim_block, idx));
  }

  auto sStage = cute::make_tensor(cute::make_smem_ptr(smem_stage), typename Traits::SmemLayoutdKV{});
  const auto tiled_copy_dKV = make_tiled_copy_C_warpcontiguousN<2>(typename Traits::SmemCopyAtomdKVC{}, score_mma);
  const auto thr_copy_dKV = tiled_copy_dKV.get_thread_slice(lane_id);
  auto tAcc = thr_copy_dKV.retile_S(half_frag);
  auto tStage = thr_copy_dKV.partition_D(sStage);
  cute::copy(tiled_copy_dKV, tAcc, tStage);
  __syncwarp();

  const auto dst_tile = cute::local_tile(dst, cute::make_shape(cute::Int<Traits::kBlockN>{}, cute::Int<Traits::kBlockK>{}), cute::make_coord(row_base / Traits::kBlockN, dim_block));
  const auto sStageVec = cute::recast<cute::uint128_t>(sStage);
  auto dstVec = cute::recast<cute::uint128_t>(dst_tile);
  const auto coordVec = cute::make_identity_tensor(cute::shape(dstVec));
  typename Traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
  const auto gmem_thr_copy = gmem_tiled_copy_dKV.get_thread_slice(lane_id);
  const auto tStageSrc = gmem_thr_copy.partition_S(sStageVec);
  auto tDst = gmem_thr_copy.partition_D(dstVec);
  const auto tCoord = gmem_thr_copy.partition_D(coordVec);
  const auto pred = cute::lazy::transform(tCoord, [&](const auto &coord) {
    return row_base + cute::get<0>(coord) < seq_len;
  });
  cute::copy_if(gmem_tiled_copy_dKV, pred, tStageSrc, tDst);
  __syncwarp();
}

template <typename Traits, typename Accum, typename Gmem>
__device__ __forceinline__ void accumulate_dQ_fragment_warp_shuffle(const Accum &accum_frag,
                                                                     const float scale,
                                                                     Gmem dst,
                                                                     const int32_t row_base,
                                                                     const int32_t dim_base,
                                                                     const int32_t seq_len,
                                                                     const int32_t lane_id)
{
  static_assert(Traits::kBlockM == 16 && Traits::kBlockK == 16, "dQ warp permutation assumes 16x16 MMA C tiles");
#pragma unroll
  for (int32_t target_idx = 0; target_idx < 8; ++target_idx)
  {
    const int32_t source_group = (target_idx >= 4 ? 2 : 0) + ((target_idx & 1) != 0 ? 4 : 0);
    const int32_t row_local = target_idx / 2 * 4 + lane_id / 8;
    const int32_t dim_local = (target_idx & 1) * 8 + lane_id % 8;
    const int32_t source_lane = (row_local & 7) * 4 + (dim_local & 7) / 2;
    const float source_even = __shfl_sync(0xffffffffu, static_cast<float>(accum_frag(source_group)), source_lane);
    const float source_odd = __shfl_sync(0xffffffffu, static_cast<float>(accum_frag(source_group + 1)), source_lane);
    const float value = (dim_local & 1) == 0 ? source_even : source_odd;
    const int32_t row = row_base + row_local;
    if (row < seq_len)
    {
      atomicAdd(&dst(row, dim_base + dim_local), value * scale);
    }
  }
}

__device__ __forceinline__ int32_t dQ_stage_offset(const int32_t row, const int32_t col)
{
  const int32_t deinterleaved_col = ((col & 1) << 3) | (col >> 1);
  const int32_t physical_col = deinterleaved_col ^ (((row >> 1) & 3) << 2);
  return row * 16 + physical_col;
}

template <bool IsAligned, typename Traits, typename Accum>
__device__ __forceinline__ void accumulate_dQ_fragment_shared_contiguous(const Accum &accum_frag,
                                                                          const float scale,
                                                                          float *const smem_stage,
                                                                          float *const dst,
                                                                          const int32_t row_base,
                                                                          const int32_t dim_base,
                                                                          const int32_t seq_len,
                                                                          const int32_t lane_id)
{
  static_assert(Traits::kBlockM == 16 && Traits::kBlockK == 16, "dQ warp permutation assumes 16x16 MMA C tiles");
#pragma unroll
  for (int32_t idx = 0; idx < cute::size(accum_frag); ++idx)
  {
    const int32_t row_local = lane_id / 4 + ((idx & 2) != 0 ? 8 : 0);
    const int32_t dim_local = (lane_id & 3) * 2 + (idx & 1) + ((idx & 4) != 0 ? 8 : 0);
    smem_stage[dQ_stage_offset(row_local, dim_local)] = static_cast<float>(accum_frag(idx)) * scale;
  }
  __syncwarp();

  const int32_t lane_row = row_base + lane_id / 8;
  float *const lane_dst = dst + lane_row * Traits::kHeadDim + dim_base + lane_id % 8;
#pragma unroll
  for (int32_t target_idx = 0; target_idx < 8; ++target_idx)
  {
    constexpr int32_t kRowsPerTargetPair = 4;
    const int32_t target_pair = target_idx / 2;
    const int32_t row_local = target_pair * kRowsPerTargetPair + lane_id / 8;
    const int32_t dim_local = (target_idx & 1) * 8 + lane_id % 8;
    const float value = smem_stage[dQ_stage_offset(row_local, dim_local)];
    const int32_t row = lane_row + target_pair * kRowsPerTargetPair;
    if (IsAligned || row < seq_len)
    {
      constexpr int32_t kPairColumnOffset = 8;
      atomicAdd(
        lane_dst + target_pair * kRowsPerTargetPair * Traits::kHeadDim + (target_idx & 1) * kPairColumnOffset,
        value);
    }
  }
  __syncwarp();
}

template <bool IsAligned, typename Traits, typename Accum>
__device__ __forceinline__ void accumulate_dQ_float_fragment_shared_contiguous(const Accum &accum_frag,
                                                                                float *const smem_stage,
                                                                                float *const dst,
                                                                                const int32_t row_base,
                                                                                const int32_t dim_base,
                                                                                const int32_t seq_len,
                                                                                const int32_t lane_id)
{
  static_assert(Traits::kBlockM == 16 && Traits::kBlockK == 16, "dQ warp permutation assumes 16x16 MMA C tiles");
#pragma unroll
  for (int32_t idx = 0; idx < cute::size(accum_frag); ++idx)
  {
    const int32_t row_local = lane_id / 4 + ((idx & 2) != 0 ? 8 : 0);
    const int32_t dim_local = (lane_id & 3) * 2 + (idx & 1) + ((idx & 4) != 0 ? 8 : 0);
    smem_stage[dQ_stage_offset(row_local, dim_local)] = accum_frag(idx);
  }
  __syncwarp();

  const int32_t lane_row = row_base + lane_id / 8;
  float *const lane_dst = dst + lane_row * Traits::kHeadDim + dim_base + lane_id % 8;
#pragma unroll
  for (int32_t target_idx = 0; target_idx < 8; ++target_idx)
  {
    constexpr int32_t kRowsPerTargetPair = 4;
    const int32_t target_pair = target_idx / 2;
    const int32_t row_local = target_pair * kRowsPerTargetPair + lane_id / 8;
    const int32_t dim_local = (target_idx & 1) * 8 + lane_id % 8;
    const float value = smem_stage[dQ_stage_offset(row_local, dim_local)];
    const int32_t row = lane_row + target_pair * kRowsPerTargetPair;
    if (IsAligned || row < seq_len)
    {
      constexpr int32_t kPairColumnOffset = 8;
      atomicAdd(
        lane_dst + target_pair * kRowsPerTargetPair * Traits::kHeadDim + (target_idx & 1) * kPairColumnOffset,
        value);
    }
  }
  __syncwarp();
}

template <int32_t HeadDim,
          int32_t CtaM,
          int32_t CtaN,
          int32_t NumWarps,
          int32_t QuantBlockQ,
          int32_t QuantBlockK,
          bool IsAligned>
__global__ __launch_bounds__(32 * NumWarps, 1) void fused_mma_kernel_k128_8warp(const int8_t *__restrict__ const Q,
                                                                               const int8_t *__restrict__ const K,
                                                                               const float *__restrict__ const QScale,
                                                                               const float *__restrict__ const KScale,
                                                                               const half *__restrict__ const V,
                                                                               const half *__restrict__ const dO,
                                                                               const int8_t *__restrict__ const dOInt8,
                                                                               const float *__restrict__ const dOScale,
                                                                               const float *__restrict__ const dSQFactors,
                                                                               const float *__restrict__ const dSKFactors,
                                                                               const float *__restrict__ const Lse,
                                                                               const float *__restrict__ const Delta,
                                                                               float *__restrict__ const dQAccum,
                                                                               half *__restrict__ const dK,
                                                                               half *__restrict__ const dV,
                                                                               const Params params,
                                                                               const float sm_scale)
{
  using Traits = BwdTileTraits<HeadDim, CtaM, CtaN, NumWarps>;
  using Storage = SharedStorage2DWarp<Traits>;
  static_assert(
    HeadDim == 64 && CtaM == 64 && CtaN == 128 && NumWarps == 8,
    "K128 eight-warp kernel is specialized for the head64 M64xN128 schedule");
  static_assert(
    QuantBlockQ == 32 && QuantBlockK == 64,
    "K128 eight-warp kernel currently implements the selected QBlock=32/KBlock=64 quantization format");
  static_assert(
    Traits::kCtaNMicroTiles == NumWarps && Traits::kCtaMMicroTiles == 4,
    "K128 ownership requires one physical warp per N16 tile and two temporal M halves");

  const int32_t n_block = blockIdx.x;
  const int32_t head = blockIdx.y;
  const int32_t batch = blockIdx.z;
  const ThreadCoord thread_coord = make_thread_coord();
  const int32_t lane_id = thread_coord.lane_id;
  const int32_t warp_id = thread_coord.warp_id;
  const int32_t n_tile = warp_id;
  const int32_t n_pair = n_tile / 2;
  const int32_t n_domain = n_tile / 4;
  const int32_t n_cta_base = n_block * Traits::kCtaN;
  const int32_t n_base = n_cta_base + n_tile * Traits::kBlockN;
  const bool n_valid = IsAligned || n_base < params.seq_len;

  if (n_cta_base >= params.seq_len)
  {
    return;
  }

  const auto gQ = make_head_matrix_view<HeadDim>(Q, params, batch, head, params.stride_bz_q, params.stride_seq_q, params.stride_h_q);
  const auto gK = make_head_matrix_view<HeadDim>(K, params, batch, head, params.stride_bz_k, params.stride_seq_k, params.stride_h_k);
  const auto gV = make_head_matrix_view<HeadDim>(V, params, batch, head, params.stride_bz_v, params.stride_seq_v, params.stride_h_v);
  const auto gdO = make_head_matrix_view<HeadDim>(dO, params, batch, head, params.stride_bz_dO, params.stride_seq_dO, params.stride_h_dO);
  const auto gdOInt8 = make_workspace_matrix_view<HeadDim>(dOInt8, params, batch, head);
  const auto gdK = make_head_matrix_view<HeadDim>(dK, params, batch, head, params.stride_bz_dK, params.stride_seq_dK, params.stride_h_dK);
  const auto gdV = make_head_matrix_view<HeadDim>(dV, params, batch, head, params.stride_bz_dV, params.stride_seq_dV, params.stride_h_dV);
  const auto gLse = make_head_vector_view(Lse, params, batch, head);
  const auto gDelta = make_head_vector_view(Delta, params, batch, head);
  constexpr int32_t kDimBlocks = HeadDim / Traits::kBlockK;
  constexpr int32_t kdQNPairs = Traits::kCtaN / (2 * Traits::kBlockN);
  constexpr int32_t kQuantDomains = Traits::kCtaN / QuantBlockK;
  static_assert(kDimBlocks == kdQNPairs && kQuantDomains == 2, "K128 dQ ownership expects four N32 pairs and two KBlock=64 domains");
  const int32_t pair_blocks = (params.seq_len + 2 * Traits::kBlockM - 1) / (2 * Traits::kBlockM);
  const float *const gdOScale = dOScale + (batch * params.num_heads + head) * pair_blocks * kDimBlocks;
  const int32_t dS_q_extent = (params.seq_len + 32 - 1) / 32;
  const int32_t dS_k_extent = (params.seq_len + 64 - 1) / 64;
  const float *const gdSQFactors = dSQFactors + (batch * params.num_heads + head) * dS_q_extent * 2;
  const float *const gdSKFactors = dSKFactors + (batch * params.num_heads + head) * dS_k_extent;
  const int32_t q_scale_extent = (params.seq_len + QuantBlockQ - 1) / QuantBlockQ;
  const int32_t k_scale_extent = (params.seq_len + QuantBlockK - 1) / QuantBlockK;
  const auto gQScale = make_head_strided_vector_view(QScale, batch, head, params.stride_bz_q_scale, params.stride_h_q_scale, q_scale_extent);
  const auto gKScale = make_head_strided_vector_view(KScale, batch, head, params.stride_bz_k_scale, params.stride_h_k_scale, k_scale_extent);

  extern __shared__ char shared_storage[];
  auto &shared = *reinterpret_cast<Storage *>(shared_storage);

  constexpr int32_t score_pair_offset = cute::cosize_v<typename Traits::SmemLayoutScorePair>;
  constexpr int32_t warp_scratch_offset = cute::cosize_v<typename Traits::SmemLayoutdSdKV>;
  constexpr int32_t kCtaNLoadRows = Traits::kCtaNLoadRows;

  auto sQStorage = make_smem_tensor(shared.q_i8, typename Traits::SmemLayoutQ{});
  auto sdOInt8Storage = make_smem_tensor(shared.dO_i8, typename Traits::SmemLayoutdO{});
  auto sKStorage = make_smem_tensor(shared.k_i8, typename Traits::SmemLayoutK{});
  auto sVStorage = make_smem_tensor(shared.v, typename Traits::SmemLayoutV{});
  auto sdOFp16PairStorage = make_smem_tensor(shared.dO_fp16_pair, typename Storage::SmemLayoutdOFp16Pair{});
  auto sQ = cute::recast<int8_t>(sQStorage);
  auto sdOInt8 = cute::recast<int8_t>(sdOInt8Storage);
  auto sK = cute::recast<int8_t>(sKStorage);
  auto sV = cute::recast<half>(sVStorage);
  auto sdOFp16Pair = cute::recast<half>(sdOFp16PairStorage);
  auto sLse = make_smem_tensor(shared.lse, typename Storage::SmemLayoutMPair{});
  auto sDelta = make_smem_tensor(shared.delta, typename Storage::SmemLayoutMPair{});

  auto sScorePairStorage = make_smem_tensor(
    shared.score_pair_i8, typename Traits::SmemLayoutScorePair{}, n_tile * score_pair_offset);
  auto sScorePair = cute::recast<int8_t>(sScorePairStorage);
  constexpr auto score_pair_tile_shape = cute::make_shape(
    cute::Int<Traits::kBlockN>{}, cute::Int<Traits::kScoreAtomK>{});
  auto sPPair = cute::local_tile(sScorePair, score_pair_tile_shape, cute::make_coord(cute::_0{}, cute::_0{}));
  auto sdSPair = cute::local_tile(sScorePair, score_pair_tile_shape, cute::make_coord(cute::_0{}, cute::_1{}));

  using BlockMNShape = typename Traits::BlockMNShape;
  typename Traits::ScoreMMA score_mma;
  typename Traits::HalfMMA half_mma;
  const auto thr_mma_score = score_mma.get_thread_slice(lane_id);
  const auto thr_mma_half = half_mma.get_thread_slice(lane_id);

  const auto tiled_copy_score_a = cute::make_tiled_copy_A(typename Traits::SmemCopyAtomScoreA{}, score_mma);
  const auto tiled_copy_score_b = cute::make_tiled_copy_B(typename Traits::SmemCopyAtomScoreB{}, score_mma);
  const auto tiled_copy_score_c = make_tiled_copy_C_warpcontiguousN<2>(typename Traits::SmemCopyAtomScoreC{}, score_mma);
  const auto thr_copy_score_a = tiled_copy_score_a.get_thread_slice(lane_id);
  const auto thr_copy_score_b = tiled_copy_score_b.get_thread_slice(lane_id);
  const auto thr_copy_score_c = tiled_copy_score_c.get_thread_slice(lane_id);
  const auto tiled_copy_transposed_b = cute::make_tiled_copy_B(typename Traits::SmemCopyAtomTransposedB{}, score_mma);
  const auto thr_copy_transposed_b = tiled_copy_transposed_b.get_thread_slice(lane_id);
  const auto tiled_copy_half_a = cute::make_tiled_copy_A(typename Traits::SmemCopyAtomdPA{}, half_mma);
  const auto tiled_copy_half_b = cute::make_tiled_copy_B(typename Traits::SmemCopyAtomdPB{}, half_mma);
  const auto thr_copy_half_a = tiled_copy_half_a.get_thread_slice(lane_id);
  const auto thr_copy_half_b = tiled_copy_half_b.get_thread_slice(lane_id);
  constexpr int32_t kAccumulatorElements = 8;

  constexpr auto cta_k_load_shape = cute::make_shape(
    cute::Int<kCtaNLoadRows>{}, cute::Int<Traits::kInt8LoadVecCols>{});
  constexpr auto cta_v_load_shape = cute::make_shape(
    cute::Int<kCtaNLoadRows>{}, cute::Int<Traits::kHalfLoadVecCols>{});
  auto sKLoad = cute::local_tile(sKStorage, cta_k_load_shape, cute::make_coord(warp_id, cute::_0{}));
  auto sVLoad = cute::local_tile(sVStorage, cta_v_load_shape, cute::make_coord(warp_id, cute::_0{}));
  const int32_t n_load_base = n_cta_base + warp_id * kCtaNLoadRows;
  load_k_tile<Traits>(gK, sKLoad, n_load_base, lane_id);
  load_v_tile<Traits>(gV, sVLoad, n_load_base, lane_id);
  constexpr int32_t kKPairRows = 2 * Traits::kBlockN;
  constexpr int32_t kKPairCount = Traits::kCtaN / kKPairRows;
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  if (warp_id < kKPairCount)
  {
#pragma unroll
    for (int32_t dim_base = 0; dim_base < HeadDim; dim_base += Traits::kBlockK)
    {
      constexpr auto k_pair_shape = cute::make_shape(
        cute::Int<kKPairRows>{}, cute::Int<Traits::kBlockK>{});
      const int32_t dim_block = dim_base / Traits::kBlockK;
      const auto gKPair = cute::local_tile(
        gK,
        k_pair_shape,
        cute::make_coord(n_cta_base / kKPairRows + warp_id, dim_block));
      const auto gKdQ = qattn_cutlass::make_int8_transposed_b_view(gKPair);
      auto fragment = thr_mma_score.partition_fragment_B(gKdQ);
      const auto gKCoordPair = cute::local_tile(
        cute::make_identity_tensor(cute::shape(gK)),
        k_pair_shape,
        cute::make_coord(n_cta_base / kKPairRows + warp_id, dim_block));
      const auto gKCoorddQ = qattn_cutlass::make_int8_transposed_b_view(gKCoordPair);
      const auto predicate = cute::lazy::transform(
        thr_mma_score.partition_B(gKCoorddQ),
        [&](auto coord) { return cute::get<0>(coord) < params.seq_len; });
      cute::clear(fragment);
      cute::copy_if(predicate, thr_mma_score.partition_B(gKdQ), fragment);
      store_packed_mma_b_fragment(
        shared.k_i8_mma_b, fragment, warp_id * kDimBlocks + dim_block, lane_id);
    }
  }
  __syncthreads();

  auto dK_accum_frag = cute::make_tensor<float>(
    cute::make_shape(cute::Int<kDimBlocks>{}, cute::Int<kAccumulatorElements>{}), cute::LayoutRight{});
  auto dV_accum_frag = cute::make_tensor<float>(
    cute::make_shape(cute::Int<kDimBlocks>{}, cute::Int<kAccumulatorElements>{}), cute::LayoutRight{});
  cute::clear(dK_accum_frag);
  cute::clear(dV_accum_frag);

  constexpr auto v_subtile_shape = cute::make_shape(
    cute::Int<Traits::kBlockN>{}, cute::Int<Traits::kBlockK>{});
  const auto sVTile0 = cute::local_tile(sV, v_subtile_shape, cute::make_coord(n_tile, cute::_0{}));
  const auto sVTile1 = cute::local_tile(sV, v_subtile_shape, cute::make_coord(n_tile, cute::_1{}));
  const auto sVTile2 = cute::local_tile(sV, v_subtile_shape, cute::make_coord(n_tile, cute::Int<2>{}));
  const auto sVTile3 = cute::local_tile(sV, v_subtile_shape, cute::make_coord(n_tile, cute::Int<3>{}));
  const auto sVHalf0 = cute::recast<cute::half_t>(sVTile0);
  const auto sVHalf1 = cute::recast<cute::half_t>(sVTile1);
  const auto sVHalf2 = cute::recast<cute::half_t>(sVTile2);
  const auto sVHalf3 = cute::recast<cute::half_t>(sVTile3);
  auto tdPrV0 = thr_mma_half.partition_fragment_B(sVHalf0);
  auto tdPrV1 = thr_mma_half.partition_fragment_B(sVHalf1);
  auto tdPrV2 = thr_mma_half.partition_fragment_B(sVHalf2);
  auto tdPrV3 = thr_mma_half.partition_fragment_B(sVHalf3);
  cute::clear(tdPrV0);
  cute::clear(tdPrV1);
  cute::clear(tdPrV2);
  cute::clear(tdPrV3);
  if (n_valid)
  {
    cute::copy(tiled_copy_half_b, thr_copy_half_b.partition_S(sVHalf0), thr_copy_half_b.retile_D(tdPrV0));
    cute::copy(tiled_copy_half_b, thr_copy_half_b.partition_S(sVHalf1), thr_copy_half_b.retile_D(tdPrV1));
    cute::copy(tiled_copy_half_b, thr_copy_half_b.partition_S(sVHalf2), thr_copy_half_b.retile_D(tdPrV2));
    cute::copy(tiled_copy_half_b, thr_copy_half_b.partition_S(sVHalf3), thr_copy_half_b.retile_D(tdPrV3));
  }

  constexpr int32_t kTailLoaderWarp0 = 1;
  constexpr int32_t kTailLoaderWarp1 = 3;
  constexpr int32_t kRowStateLoaderWarp = 7;
  const bool is_tail_loader = warp_id == kTailLoaderWarp0 || warp_id == kTailLoaderWarp1;
  const int32_t tail_loader_id = warp_id == kTailLoaderWarp1;
  if (is_tail_loader)
  {
    load_q_dO_pair<Traits>(
      gQ,
      gdOInt8,
      gdO,
      sQStorage,
      sdOInt8Storage,
      sdOFp16PairStorage,
      0,
      tail_loader_id,
      lane_id);
  }
  if (warp_id == kRowStateLoaderWarp)
  {
    load_row_state_pair(gLse, gDelta, sLse, sDelta, 0, params.seq_len, lane_id);
  }
  if (is_tail_loader)
  {
    cute::cp_async_wait<0>();
  }
  __syncthreads();

  const int32_t k_domain_base = n_cta_base / QuantBlockK;
  const auto load_k_block_scale = [&]() {
    const int32_t scale_index_unclamped = k_domain_base + n_domain;
    const int32_t scale_index =
      scale_index_unclamped < k_scale_extent ? scale_index_unclamped : k_scale_extent - 1;
    return gKScale(scale_index);
  };
  float aligned_k_block_scale = 0.0f;
  if constexpr (IsAligned)
  {
    aligned_k_block_scale = load_k_block_scale();
  }

  for (int32_t m_pair_base = 0; m_pair_base < params.seq_len; m_pair_base += 2 * Traits::kBlockM)
  {
    const int32_t q_block_index = m_pair_base / QuantBlockQ;
    const float q_block_scale = gQScale(q_block_index);
    float k_block_scale = aligned_k_block_scale;
    if constexpr (!IsAligned)
    {
      k_block_scale = load_k_block_scale();
    }
    constexpr auto QdO_pair_shape = cute::make_shape(
      cute::Int<2 * Traits::kBlockM>{}, cute::Int<Traits::kBlockK>{});
    if (warp_id < kDimBlocks)
    {
      const auto sQPair = cute::local_tile(
        sQ, QdO_pair_shape, cute::make_coord(cute::_0{}, warp_id));
      const auto sQdK = qattn_cutlass::make_int8_transposed_b_view(sQPair);
      auto fragment = thr_mma_score.partition_fragment_B(sQdK);
      cute::copy(
        tiled_copy_transposed_b,
        thr_copy_transposed_b.partition_S(sQdK),
        thr_copy_transposed_b.retile_D(fragment));
      store_packed_mma_b_fragment(shared.QdO_mma_b.q_i8_mma_b, fragment, warp_id, lane_id);
    }
    else if (warp_id < 2 * kDimBlocks)
    {
      const int32_t dim_block = warp_id - kDimBlocks;
      const auto sdOPair = cute::local_tile(
        sdOInt8, QdO_pair_shape, cute::make_coord(cute::_0{}, dim_block));
      const auto sdOdV = qattn_cutlass::make_int8_transposed_b_view(sdOPair);
      auto fragment = thr_mma_score.partition_fragment_B(sdOdV);
      cute::copy(
        tiled_copy_transposed_b,
        thr_copy_transposed_b.partition_S(sdOdV),
        thr_copy_transposed_b.retile_D(fragment));
      store_packed_mma_b_fragment(shared.QdO_mma_b.dO_i8_mma_b, fragment, dim_block, lane_id);
    }
    auto acc_score_0 = cute::partition_fragment_C(score_mma, BlockMNShape{});
    auto acc_score_1 = cute::partition_fragment_C(score_mma, BlockMNShape{});
    auto acc_dp_0 = cute::partition_fragment_C(half_mma, BlockMNShape{});
    auto acc_dp_1 = cute::partition_fragment_C(half_mma, BlockMNShape{});
    cute::clear(acc_score_0);
    cute::clear(acc_score_1);
    cute::clear(acc_dp_0);
    cute::clear(acc_dp_1);

    if (n_valid)
    {
#pragma unroll
      for (int32_t dim_base = 0; dim_base < HeadDim; dim_base += Traits::kScoreAtomK)
      {
        constexpr auto score_subtile_shape = cute::make_shape(
          cute::Int<Traits::kBlockM>{}, cute::Int<Traits::kScoreAtomK>{});
        const auto sQTile0 = cute::local_tile(
          sQ, score_subtile_shape, cute::make_coord(cute::_0{}, dim_base / Traits::kScoreAtomK));
        const auto sQTile1 = cute::local_tile(
          sQ, score_subtile_shape, cute::make_coord(cute::_1{}, dim_base / Traits::kScoreAtomK));
        const auto sKTile = cute::local_tile(
          sK,
          cute::make_shape(cute::Int<Traits::kBlockN>{}, cute::Int<Traits::kScoreAtomK>{}),
          cute::make_coord(n_tile, dim_base / Traits::kScoreAtomK));
        auto tSrQ = thr_mma_score.partition_fragment_A(sQTile0);
        auto tSrK = thr_mma_score.partition_fragment_B(sKTile);
        cute::copy(tiled_copy_score_b, thr_copy_score_b.partition_S(sKTile), thr_copy_score_b.retile_D(tSrK));
        cute::copy(tiled_copy_score_a, thr_copy_score_a.partition_S(sQTile0), thr_copy_score_a.retile_D(tSrQ));
        cute::gemm(thr_mma_score, tSrQ, tSrK, acc_score_0);
        cute::copy(tiled_copy_score_a, thr_copy_score_a.partition_S(sQTile1), thr_copy_score_a.retile_D(tSrQ));
        cute::gemm(thr_mma_score, tSrQ, tSrK, acc_score_1);
      }

#pragma unroll
      for (int32_t dim_base = 0; dim_base < HeadDim; dim_base += Traits::kBlockK)
      {
        constexpr auto half_subtile_shape = cute::make_shape(
          cute::Int<Traits::kBlockM>{}, cute::Int<Traits::kBlockK>{});
        const auto sdOTile0 = cute::local_tile(
          sdOFp16Pair, half_subtile_shape, cute::make_coord(cute::_0{}, dim_base / Traits::kBlockK));
        const auto sdOTile1 = cute::local_tile(
          sdOFp16Pair, half_subtile_shape, cute::make_coord(cute::_1{}, dim_base / Traits::kBlockK));
        const auto sdOHalf0 = cute::recast<cute::half_t>(sdOTile0);
        const auto sdOHalf1 = cute::recast<cute::half_t>(sdOTile1);
        auto tdPrdO = thr_mma_half.partition_fragment_A(sdOHalf0);
        cute::copy(tiled_copy_half_a, thr_copy_half_a.partition_S(sdOHalf0), thr_copy_half_a.retile_D(tdPrdO));
        if (dim_base == 0)
        {
          cute::gemm(thr_mma_half, tdPrdO, tdPrV0, acc_dp_0);
        }
        else if (dim_base == Traits::kBlockK)
        {
          cute::gemm(thr_mma_half, tdPrdO, tdPrV1, acc_dp_0);
        }
        else if (dim_base == 2 * Traits::kBlockK)
        {
          cute::gemm(thr_mma_half, tdPrdO, tdPrV2, acc_dp_0);
        }
        else
        {
          cute::gemm(thr_mma_half, tdPrdO, tdPrV3, acc_dp_0);
        }
        cute::copy(tiled_copy_half_a, thr_copy_half_a.partition_S(sdOHalf1), thr_copy_half_a.retile_D(tdPrdO));
        if (dim_base == 0)
        {
          cute::gemm(thr_mma_half, tdPrdO, tdPrV0, acc_dp_1);
        }
        else if (dim_base == Traits::kBlockK)
        {
          cute::gemm(thr_mma_half, tdPrdO, tdPrV1, acc_dp_1);
        }
        else if (dim_base == 2 * Traits::kBlockK)
        {
          cute::gemm(thr_mma_half, tdPrdO, tdPrV2, acc_dp_1);
        }
        else
        {
          cute::gemm(thr_mma_half, tdPrdO, tdPrV3, acc_dp_1);
        }
      }
    }

    auto rPFloat0 = cute::recast<float>(acc_score_0);
    auto rPFloat1 = cute::recast<float>(acc_score_1);
    float p_max_abs = 0.0f;
    const float score_scale = q_block_scale * k_block_scale;
    {
      const int32_t row_state_0 = lane_id / 4;
      const int32_t row_state_1 = row_state_0 + 8;
      const float lse_0 = sLse(row_state_0);
      const float lse_1 = sLse(row_state_1);
      const float delta_0 = sDelta(row_state_0);
      const float delta_1 = sDelta(row_state_1);
#pragma unroll
      for (int32_t idx = 0; idx < cute::size(acc_score_0); ++idx)
      {
        const int32_t m_local = lane_id / 4 + ((idx & 2) != 0 ? 8 : 0);
        const int32_t n_local = (lane_id & 3) * 2 + (idx & 1) + ((idx & 4) != 0 ? 8 : 0);
        const int32_t row = m_pair_base + m_local;
        const int32_t col = n_base + n_local;
        float p = 0.0f;
        float dS = 0.0f;
        if (IsAligned || (row < params.seq_len && col < params.seq_len))
        {
          const bool upper_row = (idx & 2) != 0;
          const float score = static_cast<float>(acc_score_0(idx)) * score_scale;
          p = expf(score * sm_scale - (upper_row ? lse_1 : lse_0));
          dS = p * (acc_dp_0(idx) - (upper_row ? delta_1 : delta_0)) * sm_scale;
          p_max_abs = fmaxf(p_max_abs, p);
        }
        rPFloat0(idx) = p;
        acc_dp_0(idx) = dS;
      }
    }
    {
      const int32_t row_state_0 = Traits::kBlockM + lane_id / 4;
      const int32_t row_state_1 = row_state_0 + 8;
      const float lse_0 = sLse(row_state_0);
      const float lse_1 = sLse(row_state_1);
      const float delta_0 = sDelta(row_state_0);
      const float delta_1 = sDelta(row_state_1);
#pragma unroll
      for (int32_t idx = 0; idx < cute::size(acc_score_1); ++idx)
      {
        const int32_t m_local = lane_id / 4 + ((idx & 2) != 0 ? 8 : 0);
        const int32_t n_local = (lane_id & 3) * 2 + (idx & 1) + ((idx & 4) != 0 ? 8 : 0);
        const int32_t row = m_pair_base + Traits::kBlockM + m_local;
        const int32_t col = n_base + n_local;
        float p = 0.0f;
        float dS = 0.0f;
        if (IsAligned || (row < params.seq_len && col < params.seq_len))
        {
          const bool upper_row = (idx & 2) != 0;
          const float score = static_cast<float>(acc_score_1(idx)) * score_scale;
          p = expf(score * sm_scale - (upper_row ? lse_1 : lse_0));
          dS = p * (acc_dp_1(idx) - (upper_row ? delta_1 : delta_0)) * sm_scale;
          p_max_abs = fmaxf(p_max_abs, p);
        }
        rPFloat1(idx) = p;
        acc_dp_1(idx) = dS;
      }
    }
    p_max_abs = warp_reduce_max(p_max_abs);
    const float p_scale = p_max_abs * kInt8ScaleInv + kInt8ScaleFloor;
    const int32_t q_factor_index = q_block_index;
    const int32_t q_factor_next = q_factor_index + 1 < dS_q_extent ? q_factor_index + 1 : q_factor_index;
    const float dO_l2_max = fmaxf(gdSQFactors[2 * q_factor_index], gdSQFactors[2 * q_factor_next]);
    const float delta_abs_max = fmaxf(gdSQFactors[2 * q_factor_index + 1], gdSQFactors[2 * q_factor_next + 1]);
    const int32_t k_factor_unclamped = k_domain_base + n_domain;
    const int32_t k_factor_index = k_factor_unclamped < dS_k_extent ? k_factor_unclamped : dS_k_extent - 1;
    const float predicted_dS_max = kdSPredictorGuard * sm_scale / static_cast<float>(params.seq_len) *
      (dO_l2_max * gdSKFactors[k_factor_index] + delta_abs_max);
    const float dS_scale = predicted_dS_max * kInt8ScaleInv + kInt8ScaleFloor;
    const float inv_p_scale = 1.0f / p_scale;
    const float inv_dS_scale = 1.0f / dS_scale;
    constexpr auto transposed_store_shape = cute::make_shape(
      cute::Int<Traits::kBlockN>{}, cute::Int<Traits::kBlockM>{});
    constexpr auto score_store_shape = cute::make_shape(
      cute::Int<Traits::kBlockM>{}, cute::Int<Traits::kBlockN>{});
    constexpr auto dS_tile_shape = cute::make_shape(
      cute::Int<Traits::kBlockM>{}, cute::Int<Traits::kScoreAtomK>{});
    {
      auto rP = cute::make_fragment_like<int8_t>(acc_score_0);
      auto rdS = cute::make_fragment_like<int8_t>(acc_score_0);
#pragma unroll
      for (int32_t idx = 0; idx < cute::size(acc_score_0); ++idx)
      {
        const int32_t m_local = lane_id / 4 + ((idx & 2) != 0 ? 8 : 0);
        const int32_t n_local = (lane_id & 3) * 2 + (idx & 1) + ((idx & 4) != 0 ? 8 : 0);
        const int32_t row = m_pair_base + m_local;
        const int32_t col = n_base + n_local;
        int8_t p_i8 = 0;
        int8_t dS_i8 = 0;
        if (IsAligned || (row < params.seq_len && col < params.seq_len))
        {
          p_i8 = round_to_int8(rPFloat0(idx) * inv_p_scale);
          dS_i8 = round_to_int8(acc_dp_0(idx) * inv_dS_scale);
        }
        rP(idx) = p_i8;
        rdS(idx) = dS_i8;
      }
      auto sPTile = cute::local_tile(sPPair, transposed_store_shape, cute::make_coord(cute::_0{}, cute::_0{}));
      auto sdSTile = cute::local_tile(sdSPair, transposed_store_shape, cute::make_coord(cute::_0{}, cute::_0{}));
      auto sP = make_transposed_tensor(sPTile);
      auto sdS = make_transposed_tensor(sdSTile);
      auto tCrP = thr_copy_score_c.retile_S(rP);
      auto tCrdS = thr_copy_score_c.retile_S(rdS);
      cute::copy(tiled_copy_score_c, tCrP, thr_copy_score_c.partition_D(sP));
      cute::copy(tiled_copy_score_c, tCrdS, thr_copy_score_c.partition_D(sdS));
      const int32_t dS_slot = 2 * n_pair;
      auto sWarpScratchStorage = make_smem_tensor(
        shared.warp_scratch.dS_dKV, typename Traits::SmemLayoutdSdKV{}, dS_slot * warp_scratch_offset);
      auto sWarpScratch = cute::recast<int8_t>(sWarpScratchStorage);
      auto sdSMirror = cute::local_tile(sWarpScratch, dS_tile_shape, cute::make_coord(cute::_0{}, cute::_0{}));
      auto sdSMirrorHalf = cute::local_tile(sdSMirror, score_store_shape, cute::make_coord(cute::_0{}, n_tile & 1));
      cute::copy(tiled_copy_score_c, tCrdS, thr_copy_score_c.partition_D(sdSMirrorHalf));
    }
    {
      auto rP = cute::make_fragment_like<int8_t>(acc_score_1);
      auto rdS = cute::make_fragment_like<int8_t>(acc_score_1);
#pragma unroll
      for (int32_t idx = 0; idx < cute::size(acc_score_1); ++idx)
      {
        const int32_t m_local = lane_id / 4 + ((idx & 2) != 0 ? 8 : 0);
        const int32_t n_local = (lane_id & 3) * 2 + (idx & 1) + ((idx & 4) != 0 ? 8 : 0);
        const int32_t row = m_pair_base + Traits::kBlockM + m_local;
        const int32_t col = n_base + n_local;
        int8_t p_i8 = 0;
        int8_t dS_i8 = 0;
        if (IsAligned || (row < params.seq_len && col < params.seq_len))
        {
          p_i8 = round_to_int8(rPFloat1(idx) * inv_p_scale);
          dS_i8 = round_to_int8(acc_dp_1(idx) * inv_dS_scale);
        }
        rP(idx) = p_i8;
        rdS(idx) = dS_i8;
      }
      auto sPTile = cute::local_tile(sPPair, transposed_store_shape, cute::make_coord(cute::_0{}, cute::_1{}));
      auto sdSTile = cute::local_tile(sdSPair, transposed_store_shape, cute::make_coord(cute::_0{}, cute::_1{}));
      auto sP = make_transposed_tensor(sPTile);
      auto sdS = make_transposed_tensor(sdSTile);
      auto tCrP = thr_copy_score_c.retile_S(rP);
      auto tCrdS = thr_copy_score_c.retile_S(rdS);
      cute::copy(tiled_copy_score_c, tCrP, thr_copy_score_c.partition_D(sP));
      cute::copy(tiled_copy_score_c, tCrdS, thr_copy_score_c.partition_D(sdS));
      const int32_t dS_slot = 2 * n_pair + 1;
      auto sWarpScratchStorage = make_smem_tensor(
        shared.warp_scratch.dS_dKV, typename Traits::SmemLayoutdSdKV{}, dS_slot * warp_scratch_offset);
      auto sWarpScratch = cute::recast<int8_t>(sWarpScratchStorage);
      auto sdSMirror = cute::local_tile(sWarpScratch, dS_tile_shape, cute::make_coord(cute::_0{}, cute::_0{}));
      auto sdSMirrorHalf = cute::local_tile(sdSMirror, score_store_shape, cute::make_coord(cute::_0{}, n_tile & 1));
      cute::copy(tiled_copy_score_c, tCrdS, thr_copy_score_c.partition_D(sdSMirrorHalf));
    }
    __syncthreads();

    if (n_valid)
    {
      const float dK_scale = dS_scale * q_block_scale;
      const int32_t dO_scale_base = (m_pair_base / (2 * Traits::kBlockM)) * kDimBlocks;
      auto tdVP = thr_mma_score.partition_fragment_A(sPPair);
      auto tdKdS = thr_mma_score.partition_fragment_A(sdSPair);
      cute::copy(tiled_copy_score_a, thr_copy_score_a.partition_S(sPPair), thr_copy_score_a.retile_D(tdVP));
      cute::copy(tiled_copy_score_a, thr_copy_score_a.partition_S(sdSPair), thr_copy_score_a.retile_D(tdKdS));
#pragma unroll
      for (int32_t dim_base = 0; dim_base < HeadDim; dim_base += Traits::kBlockK)
      {
        constexpr auto dO_pair_shape = cute::make_shape(
          cute::Int<2 * Traits::kBlockM>{}, cute::Int<Traits::kBlockK>{});
        const auto sdOPair = cute::local_tile(
          sdOInt8, dO_pair_shape, cute::make_coord(cute::_0{}, dim_base / Traits::kBlockK));
        const auto sdOdV = qattn_cutlass::make_int8_transposed_b_view(sdOPair);
        auto dV_acc = cute::partition_fragment_C(score_mma, BlockMNShape{});
        cute::clear(dV_acc);
        auto tdVdO = thr_mma_score.partition_fragment_B(sdOdV);
        load_packed_mma_b_fragment(shared.QdO_mma_b.dO_i8_mma_b, tdVdO, dim_base / Traits::kBlockK, lane_id);
        cute::gemm(thr_mma_score, tdVP, tdVdO, dV_acc);
        const int32_t dim_block = dim_base / Traits::kBlockK;
        const float dO_scale = gdOScale[dO_scale_base + dim_block];
        const float dV_scale = p_scale * dO_scale;
#pragma unroll
        for (int32_t idx = 0; idx < cute::size(dV_acc); ++idx)
        {
          dV_accum_frag(dim_block, idx) += static_cast<float>(dV_acc(idx)) * dV_scale;
        }

        constexpr auto q_pair_shape = cute::make_shape(
          cute::Int<2 * Traits::kBlockM>{}, cute::Int<Traits::kBlockK>{});
        const auto sQPair = cute::local_tile(
          sQ, q_pair_shape, cute::make_coord(cute::_0{}, dim_block));
        const auto sQdK = qattn_cutlass::make_int8_transposed_b_view(sQPair);
        auto dK_acc = cute::partition_fragment_C(score_mma, BlockMNShape{});
        cute::clear(dK_acc);
        auto tdKQ = thr_mma_score.partition_fragment_B(sQdK);
        load_packed_mma_b_fragment(shared.QdO_mma_b.q_i8_mma_b, tdKQ, dim_block, lane_id);
        cute::gemm(thr_mma_score, tdKdS, tdKQ, dK_acc);
#pragma unroll
        for (int32_t idx = 0; idx < cute::size(dK_acc); ++idx)
        {
          dK_accum_frag(dim_block, idx) += static_cast<float>(dK_acc(idx)) * dK_scale;
        }
      }
    }

    __syncthreads();
    const int32_t next_m_pair_base = m_pair_base + 2 * Traits::kBlockM;
    const bool has_next_m_pair = next_m_pair_base < params.seq_len;
    if (has_next_m_pair && is_tail_loader)
    {
      load_q_dO_pair<Traits>(
        gQ,
        gdOInt8,
        gdO,
        sQStorage,
        sdOInt8Storage,
        sdOFp16PairStorage,
        next_m_pair_base,
        tail_loader_id,
        lane_id);
    }
    if (has_next_m_pair && warp_id == kRowStateLoaderWarp)
    {
      load_row_state_pair(gLse, gDelta, sLse, sDelta, next_m_pair_base, params.seq_len, lane_id);
    }

    if ((n_tile & 1) == 0)
    {
      const int32_t dim_base = n_pair * Traits::kBlockK;
#pragma unroll
      for (int32_t m_half = 0; m_half < 2; ++m_half)
      {
        auto dQ_acc_like = cute::partition_fragment_C(score_mma, BlockMNShape{});
        auto dQ_sum = cute::make_fragment_like<float>(dQ_acc_like);
        cute::clear(dQ_sum);
#pragma unroll
        for (int32_t domain = 0; domain < kQuantDomains; ++domain)
        {
          auto dQ_acc = cute::partition_fragment_C(score_mma, BlockMNShape{});
          cute::clear(dQ_acc);
#pragma unroll
          for (int32_t pair_in_domain = 0; pair_in_domain < 2; ++pair_in_domain)
          {
            const int32_t dQ_pair = 2 * domain + pair_in_domain;
            const int32_t dQ_dS_slot = 2 * dQ_pair + m_half;
            auto sdSScratchStorage = make_smem_tensor(
              shared.warp_scratch.dS_dKV,
              typename Traits::SmemLayoutdSdKV{},
              dQ_dS_slot * warp_scratch_offset);
            auto sdSScratch = cute::recast<int8_t>(sdSScratchStorage);
            auto sdSdQ = cute::local_tile(sdSScratch, dS_tile_shape, cute::make_coord(cute::_0{}, cute::_0{}));
            auto tdQdS = thr_mma_score.partition_fragment_A(sdSdQ);
            constexpr auto k_pair_shape = cute::make_shape(
              cute::Int<2 * Traits::kBlockN>{}, cute::Int<Traits::kBlockK>{});
            const auto sKPair = cute::local_tile(
              sK, k_pair_shape, cute::make_coord(dQ_pair, dim_base / Traits::kBlockK));
            const auto sKdQ = qattn_cutlass::make_int8_transposed_b_view(sKPair);
            auto tdQK = thr_mma_score.partition_fragment_B(sKdQ);
            cute::copy(tiled_copy_score_a, thr_copy_score_a.partition_S(sdSdQ), thr_copy_score_a.retile_D(tdQdS));
            load_packed_mma_b_fragment(
              shared.k_i8_mma_b,
              tdQK,
              dQ_pair * kDimBlocks + dim_base / Traits::kBlockK,
              lane_id);
            cute::gemm(thr_mma_score, tdQdS, tdQK, dQ_acc);
          }
          const int32_t domain_k_factor_unclamped = k_domain_base + domain;
          const int32_t domain_k_factor_index =
            domain_k_factor_unclamped < dS_k_extent ? domain_k_factor_unclamped : dS_k_extent - 1;
          const float dQ_predicted_dS_max = kdSPredictorGuard * sm_scale / static_cast<float>(params.seq_len) *
            (dO_l2_max * gdSKFactors[domain_k_factor_index] + delta_abs_max);
          const float dQ_dS_scale = dQ_predicted_dS_max * kInt8ScaleInv + kInt8ScaleFloor;
          const int32_t domain_k_scale_index_unclamped = k_domain_base + domain;
          const int32_t domain_k_scale_index =
            domain_k_scale_index_unclamped < k_scale_extent ? domain_k_scale_index_unclamped : k_scale_extent - 1;
          const float dQ_scale = dQ_dS_scale * gKScale(domain_k_scale_index);
#pragma unroll
          for (int32_t idx = 0; idx < cute::size(dQ_acc); ++idx)
          {
            dQ_sum(idx) += static_cast<float>(dQ_acc(idx)) * dQ_scale;
          }
        }
        static_assert(
          score_pair_offset * sizeof(cute::uint128_t) >= 16 * 16 * sizeof(float),
          "Each dead score-pair slot must fit one warp-local dQ transpose");
        auto *const dQ_stage = reinterpret_cast<float *>(
          shared.score_pair_i8.begin() + n_tile * score_pair_offset);
        accumulate_dQ_float_fragment_shared_contiguous<IsAligned, Traits>(
          dQ_sum,
          dQ_stage,
          dQAccum + (batch * params.num_heads + head) * params.seq_len * HeadDim,
          m_pair_base + m_half * Traits::kBlockM,
          dim_base,
          params.seq_len,
          lane_id);
      }
    }

    if (has_next_m_pair && is_tail_loader)
    {
      cute::cp_async_wait<0>();
    }
    __syncthreads();
  }

  half *const dKV_stage = reinterpret_cast<half *>(
    shared.warp_scratch.dS_dKV.begin() + n_tile * warp_scratch_offset);
  if (n_valid)
  {
#pragma unroll
    for (int32_t dim_block = 0; dim_block < kDimBlocks; ++dim_block)
    {
      store_dKV_fragment_coalesced<Traits>(
        dV_accum_frag, dim_block, gdV, dKV_stage, n_base, params.seq_len, lane_id);
      store_dKV_fragment_coalesced<Traits>(
        dK_accum_frag, dim_block, gdK, dKV_stage, n_base, params.seq_len, lane_id);
    }
  }
}

} // namespace sageattention::qattn_cutlass_bwd

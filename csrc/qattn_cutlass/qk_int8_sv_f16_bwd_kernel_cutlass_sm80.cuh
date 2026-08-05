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
  static constexpr int32_t kDkvGmemThreadsPerRow = kBlockK / kHalfGmemVecElems;
  static_assert(kBlockK % kHalfGmemVecElems == 0, "dKV gmem rows must be 128-bit aligned");
  using GmemCopyAtomAsync = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, cute::uint128_t>;
  using GmemCopyAtomdKV = cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, cute::uint128_t>;
  using GmemTiledCopydKV = decltype(cute::make_tiled_copy(
    GmemCopyAtomdKV{},
    cute::Layout<cute::Shape<cute::Int<32 / kDkvGmemThreadsPerRow>, cute::Int<kDkvGmemThreadsPerRow>>, cute::Stride<cute::Int<kDkvGmemThreadsPerRow>, cute::_1>>{},
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
  using GmemCopyAtomDQAccum = cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, cute::uint128_t>;
  using GmemCopyAtomDQ = cute::Copy_Atom<cute::UniversalCopy<uint64_t>, uint64_t>;
  using GmemTiledCopyDQZero = typename GmemTiledCopyContract<8, kFloatLoadVecCols, GmemCopyAtomDQAccum>::TiledCopy;
  using GmemTiledCopyDQAccum = typename GmemTiledCopyContract<4, kFloatLoadVecCols, GmemCopyAtomDQAccum>::TiledCopy;
  using GmemTiledCopyDQ = typename GmemTiledCopyContract<4, kFloatLoadVecCols, GmemCopyAtomDQ>::TiledCopy;
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
  using SmemLayoutdSkdKV = decltype(make_smem_matrix_layout<kBlockM, 2 * kScoreVecCols>());
  using SmemLayoutdKV = decltype(cute::composition(cute::Swizzle<1, 3, 3>{}, make_plain_smem_matrix_layout<kBlockN, kBlockK>()));
  using SmemLayoutM = decltype(cute::make_layout(cute::make_shape(cute::Int<kBlockM>{}), cute::make_stride(cute::_1{})));
  using SmemLayoutN = decltype(cute::make_layout(cute::make_shape(cute::Int<kBlockN>{}), cute::make_stride(cute::_1{})));
};

template <typename Traits>
struct SharedStorage
{
  struct WarpScratchStorage
  {
    alignas(16) cute::ArrayEngine<cute::uint128_t, Traits::kSmemWarps * cute::cosize_v<typename Traits::SmemLayoutdSkdKV>> dsk_dkv;
  };

  alignas(16) cute::ArrayEngine<cute::uint128_t, cute::cosize_v<typename Traits::SmemLayoutQ>> q_i8;
  alignas(16) cute::ArrayEngine<cute::uint128_t, cute::cosize_v<typename Traits::SmemLayoutdO>> do_i8;
  alignas(16) cute::ArrayEngine<cute::uint128_t, cute::cosize_v<typename Traits::SmemLayoutK>> k_i8;
  union
  {
    alignas(16) cute::ArrayEngine<cute::uint128_t, cute::cosize_v<typename Traits::SmemLayoutdOFp16>> do_fp16;
    WarpScratchStorage warp_scratch;
  };
  alignas(16) cute::ArrayEngine<cute::uint128_t, Traits::kSmemWarps * cute::cosize_v<typename Traits::SmemLayoutScorePair>> score_pair_i8;
  alignas(16) cute::ArrayEngine<cute::uint128_t, cute::cosize_v<typename Traits::SmemLayoutV>> v;
  cute::ArrayEngine<float, cute::cosize_v<typename Traits::SmemLayoutM>> q_scale;
  cute::ArrayEngine<float, Traits::kSmemWarps * cute::cosize_v<typename Traits::SmemLayoutN>> k_scale;
  float dsk_scale[Traits::kSmemWarps];
  cute::ArrayEngine<float, cute::cosize_v<typename Traits::SmemLayoutM>> lse;
  cute::ArrayEngine<float, cute::cosize_v<typename Traits::SmemLayoutM>> delta;
};

template <typename Traits>
struct SharedStorage2DWarp
{
  static_assert(
    Traits::kCtaM == 64 &&
      ((Traits::kNumWarps == 8 && Traits::kCtaN == 64) ||
       (Traits::kNumWarps == 16 && Traits::kCtaN == 128)),
    "2D warp storage is specialized for the 64-row, two-M-half schedule");
  using SmemLayoutdOFp16Pair = decltype(make_smem_matrix_layout<2 * Traits::kBlockM, Traits::kHalfLoadVecCols>());
  using SmemLayoutMPair = decltype(cute::make_layout(
    cute::make_shape(cute::Int<2 * Traits::kBlockM>{}), cute::make_stride(cute::_1{})));

  struct WarpScratchStorage
  {
    alignas(16) cute::ArrayEngine<cute::uint128_t, Traits::kSmemWarps * cute::cosize_v<typename Traits::SmemLayoutdSkdKV>> dsk_dkv;
  };

  alignas(16) cute::ArrayEngine<cute::uint128_t, cute::cosize_v<typename Traits::SmemLayoutQ>> q_i8;
  alignas(16) cute::ArrayEngine<cute::uint128_t, cute::cosize_v<typename Traits::SmemLayoutdO>> do_i8;
  alignas(16) cute::ArrayEngine<cute::uint128_t, cute::cosize_v<typename Traits::SmemLayoutK>> k_i8;
  alignas(16) cute::ArrayEngine<cute::uint128_t, cute::cosize_v<typename Traits::SmemLayoutK>> k_i8_mma_b;
  alignas(16) cute::ArrayEngine<cute::uint128_t, cute::cosize_v<SmemLayoutdOFp16Pair>> do_fp16_pair;
  WarpScratchStorage warp_scratch;
  alignas(16) cute::ArrayEngine<cute::uint128_t, Traits::kSmemWarps * cute::cosize_v<typename Traits::SmemLayoutScorePair>> score_pair_i8;
  alignas(16) cute::ArrayEngine<cute::uint128_t, cute::cosize_v<typename Traits::SmemLayoutV>> v;
  cute::ArrayEngine<float, cute::cosize_v<SmemLayoutMPair>> lse;
  cute::ArrayEngine<float, cute::cosize_v<SmemLayoutMPair>> delta;
  float p_scale[Traits::kNumWarps];
  float ds_scale[Traits::kNumWarps];
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
  int32_t warp_q;
  int32_t warp_k;
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
  int32_t stride_bz_do;
  int32_t stride_seq_do;
  int32_t stride_h_do;
  int32_t stride_bz_dq;
  int32_t stride_seq_dq;
  int32_t stride_h_dq;
  int32_t stride_bz_dk;
  int32_t stride_seq_dk;
  int32_t stride_h_dk;
  int32_t stride_bz_dv;
  int32_t stride_seq_dv;
  int32_t stride_h_dv;
  int32_t stride_bz_q_scale;
  int32_t stride_h_q_scale;
  int32_t stride_bz_k_scale;
  int32_t stride_h_k_scale;
};

template <typename ScaleView>
__device__ __forceinline__ float query_scale_for_row(const ScaleView scale, const Params &params, const int32_t row)
{
  const int32_t scale_idx = (row / params.warp_q) * 8 + (row & 7);
  return scale(scale_idx);
}

template <typename ScaleView>
__device__ __forceinline__ float key_scale_for_row(const ScaleView scale, const Params &params, const int32_t row)
{
  const int32_t scale_idx = (row / params.warp_k) * 4 + ((row & 7) >> 1);
  return scale(scale_idx);
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
  return static_cast<int8_t>(__float2int_rn(value));
}

struct LaneCoord
{
  const int32_t id;
  const int32_t quad;
  const int32_t quad_lane;
  const int32_t quad_pair;
  const int32_t quad_pair_lane;
};

__device__ __forceinline__ LaneCoord make_lane_coord(const int32_t lane_id)
{
  constexpr auto quad_layout = cute::make_layout(cute::make_shape(cute::_8{}, cute::_4{}), cute::LayoutRight{});
  const auto quad_coord = quad_layout.get_flat_coord(lane_id);
  const int32_t quad_lane = cute::get<1>(quad_coord);

  constexpr auto quad_pair_layout = cute::make_layout(cute::make_shape(cute::_2{}, cute::_2{}), cute::LayoutRight{});
  const auto quad_pair_coord = quad_pair_layout.get_flat_coord(quad_lane);

  return {lane_id, cute::get<0>(quad_coord), quad_lane, cute::get<0>(quad_pair_coord), cute::get<1>(quad_pair_coord)};
}

struct ThreadCoord
{
  const int32_t lane_id;
  const int32_t warp_id;
  const LaneCoord lane;
};

__device__ __forceinline__ ThreadCoord make_thread_coord()
{
  const int32_t lane_id = threadIdx.x;
  return {lane_id, static_cast<int32_t>(threadIdx.y), make_lane_coord(lane_id)};
}

template <int32_t HeadDim>
__global__ void preprocess_delta_zero_dq_kernel(const half *__restrict__ const O,
                                                const half *__restrict__ const DO,
                                                float *__restrict__ const Delta,
                                                float *__restrict__ const DQAccum,
                                                int8_t *__restrict__ const DOInt8,
                                                float *__restrict__ const DOScale,
                                                const Params params)
{
  using Traits = BwdTileTraits<HeadDim>;
  constexpr int32_t kPairRows = 2 * Traits::kBlockM;
  constexpr int32_t kWarps = 4;
  const int32_t pair_block = blockIdx.x;
  const int32_t head = blockIdx.y;
  const int32_t batch = blockIdx.z;
  const int32_t base_row = pair_block * kPairRows;
  const int32_t tid = threadIdx.x;
  const int32_t warp_id = tid / warpSize;
  const int32_t lane_id = tid % warpSize;

  const auto gO = make_head_matrix_view<HeadDim>(O, params, batch, head, params.stride_bz_o, params.stride_seq_o, params.stride_h_o);
  const auto gDO = make_head_matrix_view<HeadDim>(DO, params, batch, head, params.stride_bz_do, params.stride_seq_do, params.stride_h_do);
  const auto gDelta = make_head_vector_view(Delta, params, batch, head);
  const auto gDQAccum = make_workspace_matrix_view<HeadDim>(DQAccum, params, batch, head);
  const auto gDOInt8 = make_workspace_matrix_view<HeadDim>(DOInt8, params, batch, head);
  constexpr int32_t kDimBlocks = HeadDim / Traits::kBlockK;
  const int32_t pair_blocks = (params.seq_len + kPairRows - 1) / kPairRows;
  float *const gDOScale = DOScale + ((batch * params.num_heads + head) * pair_blocks + pair_block) * kDimBlocks;

  const auto gOVec = cute::recast<cute::uint128_t>(gO);
  const auto gDOVec = cute::recast<cute::uint128_t>(gDO);
  const auto gDQAccumVec = cute::recast<cute::uint128_t>(gDQAccum);
  const auto gDOInt8Vec = cute::recast<cute::uint128_t>(gDOInt8);

  constexpr int32_t kDeltaRowLanes = 4;
  constexpr int32_t kDeltaRowsPerWarp = 32 / kDeltaRowLanes;
  constexpr int32_t kDeltaVecsPerThread = Traits::kHalfLoadVecCols / kDeltaRowLanes;
  const int32_t delta_row = base_row + warp_id * kDeltaRowsPerWarp + lane_id / kDeltaRowLanes;
  const int32_t delta_vec_base = lane_id % kDeltaRowLanes;
  auto rO = cute::make_tensor<cute::uint128_t>(cute::make_shape(cute::Int<kDeltaVecsPerThread>{}));
  auto rdO = cute::make_tensor<cute::uint128_t>(cute::make_shape(cute::Int<kDeltaVecsPerThread>{}));
  cute::clear(rO);
  cute::clear(rdO);
  if (delta_row < params.seq_len)
  {
#pragma unroll
    for (int32_t vec = 0; vec < kDeltaVecsPerThread; ++vec)
    {
      const int32_t vec_col = delta_vec_base + vec * kDeltaRowLanes;
      rO(vec) = gOVec(delta_row, vec_col);
      rdO(vec) = gDOVec(delta_row, vec_col);
    }
  }

  const auto rOHalf = cute::recast<half>(rO);
  const auto rdOHalf = cute::recast<half>(rdO);
  float delta = 0.0f;
#pragma unroll
  for (int32_t idx = 0; idx < cute::size(rOHalf); ++idx)
  {
    delta += __half2float(rOHalf(idx)) * __half2float(rdOHalf(idx));
  }
#pragma unroll
  for (int32_t offset = kDeltaRowLanes / 2; offset > 0; offset >>= 1)
  {
    delta += __shfl_down_sync(0xffffffff, delta, offset, kDeltaRowLanes);
  }
  if ((lane_id % kDeltaRowLanes) == 0 && delta_row < params.seq_len)
  {
    gDelta(delta_row) = delta;
  }

  constexpr auto zero_tile_shape = cute::make_shape(cute::_8{}, cute::Int<Traits::kFloatLoadVecCols>{});
  const auto zero_tile_coord = cute::make_coord(base_row / 8 + warp_id, cute::_0{});
  auto gDQZeroTile = cute::local_tile(gDQAccumVec, zero_tile_shape, zero_tile_coord);
  const auto zero_coord_tile = cute::local_tile(cute::make_identity_tensor(cute::shape(gDQAccumVec)), zero_tile_shape, zero_tile_coord);
  typename Traits::GmemTiledCopyDQZero tiled_copy_zero;
  const auto thr_copy_zero = tiled_copy_zero.get_thread_slice(lane_id);
  auto tZgDQ = thr_copy_zero.partition_D(gDQZeroTile);
  const auto tZCoord = thr_copy_zero.partition_D(zero_coord_tile);
  auto rZero = cute::make_fragment_like(tZgDQ);
  cute::clear(rZero);
  const auto tZPred = cute::lazy::transform(tZCoord, [&](const auto &coord) {
    return cute::get<0>(coord) < params.seq_len;
  });
  cute::copy_if(tiled_copy_zero, tZPred, rZero, tZgDQ);

#pragma unroll
  for (int32_t dim_block = warp_id; dim_block < kDimBlocks; dim_block += kWarps)
  {
    const int32_t row = base_row + lane_id;
    auto rdOPacked = cute::make_tensor<cute::uint128_t>(cute::make_shape(cute::_2{}));
    cute::clear(rdOPacked);
    if (row < params.seq_len)
    {
      rdOPacked(0) = gDOVec(row, 2 * dim_block);
      rdOPacked(1) = gDOVec(row, 2 * dim_block + 1);
    }
    const auto rdOBlock = cute::recast<half>(rdOPacked);
    float max_abs = 0.0f;
#pragma unroll
    for (int32_t elem = 0; elem < Traits::kBlockK; ++elem)
    {
      max_abs = fmaxf(max_abs, fabsf(__half2float(rdOBlock(elem))));
    }
    max_abs = warp_reduce_max(max_abs);
    const float scale = __shfl_sync(0xffffffff, max_abs / 127.5f + 1.0e-7f, 0);
    if (lane_id == 0)
    {
      gDOScale[dim_block] = scale;
    }
    if (row < params.seq_len)
    {
      auto rdOQuant = cute::make_fragment_like<int8_t>(rdOBlock);
      const float inv_scale = 1.0f / scale;
#pragma unroll
      for (int32_t elem = 0; elem < Traits::kBlockK; ++elem)
      {
        rdOQuant(elem) = round_to_int8(__half2float(rdOBlock(elem)) * inv_scale);
      }
      const auto rdOQuantPacked = cute::recast<cute::uint128_t>(rdOQuant);
      gDOInt8Vec(row, dim_block) = rdOQuantPacked(0);
    }
  }
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
__device__ __forceinline__ void load_do_int8_tile(const Gmem gmem, Smem smem, const int32_t row_base, const int32_t lane_id)
{
  typename Traits::GmemTiledCopydO tiled_copy;
  load_head_tile_with_contract<Traits::kMStageLoadRows, Traits::kInt8LoadVecCols>(
    tiled_copy, cute::recast<cute::uint128_t>(gmem), smem, row_base, lane_id);
}

template <typename Traits, typename Gmem, typename Smem>
__device__ __forceinline__ void load_do_fp16_tile(const Gmem gmem, Smem smem, const int32_t row_base, const int32_t lane_id)
{
  typename Traits::GmemTiledCopydOFp16 tiled_copy;
  load_head_tile_with_contract<Traits::kMStageLoadRows, Traits::kHalfLoadVecCols>(
    tiled_copy, cute::recast<cute::uint128_t>(gmem), smem, row_base, lane_id);
}

template <typename Traits,
          typename GQ,
          typename GDOInt8,
          typename GDO,
          typename SQStorage,
          typename SDOInt8Storage,
          typename SDOFp16Storage>
__device__ __forceinline__ void load_q_do_pair(const GQ gQ,
                                                const GDOInt8 gDOInt8,
                                                const GDO gDO,
                                                SQStorage sQStorage,
                                                SDOInt8Storage sdOInt8Storage,
                                                SDOFp16Storage sdOFp16Storage,
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
    load_do_int8_tile<Traits>(gDOInt8, sdOInt8Load, row_base, lane_id);
    load_do_fp16_tile<Traits>(gDO, sdOFp16Load, row_base, lane_id);
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
__device__ __forceinline__ void store_dkv_fragment_coalesced(const Accum &accum_frag,
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
  const auto tiled_copy_dkv = make_tiled_copy_C_warpcontiguousN<2>(typename Traits::SmemCopyAtomdKVC{}, score_mma);
  const auto thr_copy_dkv = tiled_copy_dkv.get_thread_slice(lane_id);
  auto tAcc = thr_copy_dkv.retile_S(half_frag);
  auto tStage = thr_copy_dkv.partition_D(sStage);
  cute::copy(tiled_copy_dkv, tAcc, tStage);
  __syncwarp();

  const auto dst_tile = cute::local_tile(dst, cute::make_shape(cute::Int<Traits::kBlockN>{}, cute::Int<Traits::kBlockK>{}), cute::make_coord(row_base / Traits::kBlockN, dim_block));
  const auto sStageVec = cute::recast<cute::uint128_t>(sStage);
  auto dstVec = cute::recast<cute::uint128_t>(dst_tile);
  const auto coordVec = cute::make_identity_tensor(cute::shape(dstVec));
  typename Traits::GmemTiledCopydKV gmem_tiled_copy_dkv;
  const auto gmem_thr_copy = gmem_tiled_copy_dkv.get_thread_slice(lane_id);
  const auto tStageSrc = gmem_thr_copy.partition_S(sStageVec);
  auto tDst = gmem_thr_copy.partition_D(dstVec);
  const auto tCoord = gmem_thr_copy.partition_D(coordVec);
  const auto pred = cute::lazy::transform(tCoord, [&](const auto &coord) {
    return row_base + cute::get<0>(coord) < seq_len;
  });
  cute::copy_if(gmem_tiled_copy_dkv, pred, tStageSrc, tDst);
  __syncwarp();
}

template <typename Traits, typename Accum, typename Gmem>
__device__ __forceinline__ void accumulate_dq_fragment_warp_shuffle(const Accum &accum_frag,
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

__device__ __forceinline__ int32_t dq_stage_offset(const int32_t row, const int32_t col)
{
  const int32_t deinterleaved_col = ((col & 1) << 3) | (col >> 1);
  const int32_t physical_col = deinterleaved_col ^ (((row >> 1) & 3) << 2);
  return row * 16 + physical_col;
}

template <bool IsAligned, typename Traits, typename Accum>
__device__ __forceinline__ void accumulate_dq_fragment_shared_contiguous(const Accum &accum_frag,
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
    smem_stage[dq_stage_offset(row_local, dim_local)] = static_cast<float>(accum_frag(idx)) * scale;
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
    const float value = smem_stage[dq_stage_offset(row_local, dim_local)];
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
__global__ __launch_bounds__(32 * NumWarps, NumWarps == 8 ? 2 : 1) void fused_mma_kernel_2d_warp(const int8_t *__restrict__ const Q,
                                        const int8_t *__restrict__ const K,
                                        const float *__restrict__ const QScale,
                                        const float *__restrict__ const KScale,
                                        const half *__restrict__ const V,
                                        const half *__restrict__ const DO,
                                        const int8_t *__restrict__ const DOInt8,
                                        const float *__restrict__ const DOScale,
                                        const float *__restrict__ const Lse,
                                        const float *__restrict__ const Delta,
                                        float *__restrict__ const DQAccum,
                                        half *__restrict__ const DK,
                                        half *__restrict__ const DV,
                                        const Params params,
                                        const float sm_scale)
{
  using Traits = BwdTileTraits<HeadDim, CtaM, CtaN, NumWarps>;
  using Storage = SharedStorage2DWarp<Traits>;
  static_assert(
    HeadDim == 64 && CtaM == 64 &&
      ((CtaN == 64 && NumWarps == 8) || (CtaN == 128 && NumWarps == 16)),
    "2D warp kernel is specialized for the head64 two-M-half schedule");
  static_assert(
    Traits::kCtaNMicroTiles == NumWarps / 2 && Traits::kCtaMMicroTiles == 4,
    "2D warp ownership requires one warp per (M-half, N-microtile)");
  static_assert(
    QuantBlockQ >= 2 * Traits::kBlockM && QuantBlockQ % (2 * Traits::kBlockM) == 0,
    "Q quantization blocks must contain complete 32-row mainloop pairs");
  static_assert(
    QuantBlockK >= Traits::kCtaN && QuantBlockK % Traits::kCtaN == 0,
    "K quantization blocks must contain complete resident CTA tiles");

  const int32_t n_block = blockIdx.x;
  const int32_t head = blockIdx.y;
  const int32_t batch = blockIdx.z;
  const ThreadCoord thread_coord = make_thread_coord();
  const int32_t lane_id = thread_coord.lane_id;
  const int32_t warp_id = thread_coord.warp_id;
  const int32_t n_tile = warp_id / 2;
  const int32_t m_half = warp_id & 1;
  const int32_t n_pair = n_tile / 2;
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
  const auto gDO = make_head_matrix_view<HeadDim>(DO, params, batch, head, params.stride_bz_do, params.stride_seq_do, params.stride_h_do);
  const auto gDOInt8 = make_workspace_matrix_view<HeadDim>(DOInt8, params, batch, head);
  const auto gDK = make_head_matrix_view<HeadDim>(DK, params, batch, head, params.stride_bz_dk, params.stride_seq_dk, params.stride_h_dk);
  const auto gDV = make_head_matrix_view<HeadDim>(DV, params, batch, head, params.stride_bz_dv, params.stride_seq_dv, params.stride_h_dv);
  const auto gLse = make_head_vector_view(Lse, params, batch, head);
  const auto gDelta = make_head_vector_view(Delta, params, batch, head);
  constexpr int32_t kDimBlocks = HeadDim / Traits::kBlockK;
  constexpr int32_t kDqNPairs = Traits::kCtaN / (2 * Traits::kBlockN);
  static_assert(kDimBlocks % kDqNPairs == 0, "dQ dimension blocks must divide N-pair owners");
  constexpr int32_t kDqDimBlocksPerOwner = kDimBlocks / kDqNPairs;
  const int32_t pair_blocks = (params.seq_len + 2 * Traits::kBlockM - 1) / (2 * Traits::kBlockM);
  const float *const gDOScale = DOScale + (batch * params.num_heads + head) * pair_blocks * kDimBlocks;
  const int32_t q_scale_extent = (params.seq_len + QuantBlockQ - 1) / QuantBlockQ;
  const int32_t k_scale_extent = (params.seq_len + QuantBlockK - 1) / QuantBlockK;
  const auto gQScale = make_head_strided_vector_view(QScale, batch, head, params.stride_bz_q_scale, params.stride_h_q_scale, q_scale_extent);
  const auto gKScale = make_head_strided_vector_view(KScale, batch, head, params.stride_bz_k_scale, params.stride_h_k_scale, k_scale_extent);

  extern __shared__ char shared_storage[];
  auto &shared = *reinterpret_cast<Storage *>(shared_storage);

  constexpr int32_t score_pair_offset = cute::cosize_v<typename Traits::SmemLayoutScorePair>;
  constexpr int32_t warp_scratch_offset = cute::cosize_v<typename Traits::SmemLayoutdSkdKV>;
  constexpr int32_t kBlockMLoadRows = Traits::kMStageLoadRows;
  constexpr int32_t kBlockMLoadWarps = Traits::kMStageLoadWarps;
  constexpr int32_t kCtaNLoadRows = Traits::kCtaNLoadRows;

  auto sQStorage = make_smem_tensor(shared.q_i8, typename Traits::SmemLayoutQ{});
  auto sdOInt8Storage = make_smem_tensor(shared.do_i8, typename Traits::SmemLayoutdO{});
  auto sKStorage = make_smem_tensor(shared.k_i8, typename Traits::SmemLayoutK{});
  auto sVStorage = make_smem_tensor(shared.v, typename Traits::SmemLayoutV{});
  auto sdOFp16PairStorage = make_smem_tensor(shared.do_fp16_pair, typename Storage::SmemLayoutdOFp16Pair{});
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

  const int32_t ds_slot = 2 * n_pair + m_half;
  auto sWarpScratchStorage = make_smem_tensor(
    shared.warp_scratch.dsk_dkv, typename Traits::SmemLayoutdSkdKV{}, ds_slot * warp_scratch_offset);
  auto sWarpScratch = cute::recast<int8_t>(sWarpScratchStorage);
  constexpr auto ds_tile_shape = cute::make_shape(
    cute::Int<Traits::kBlockM>{}, cute::Int<Traits::kScoreAtomK>{});
  auto sdSMirror = cute::local_tile(sWarpScratch, ds_tile_shape, cute::make_coord(cute::_0{}, cute::_0{}));

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
      const auto gKCoordDq = qattn_cutlass::make_int8_transposed_b_view(gKCoordPair);
      const auto predicate = cute::lazy::transform(
        thr_mma_score.partition_B(gKCoordDq),
        [&](auto coord) { return cute::get<0>(coord) < params.seq_len; });
      cute::clear(fragment);
      cute::copy_if(predicate, thr_mma_score.partition_B(gKdQ), fragment);
      store_packed_mma_b_fragment(
        shared.k_i8_mma_b, fragment, warp_id * kDimBlocks + dim_block, lane_id);
    }
  }
  __syncthreads();
  const float k_block_scale = gKScale(n_cta_base / QuantBlockK);

  auto dkv_accum_frag = cute::make_tensor<float>(
    cute::make_shape(cute::Int<kDimBlocks>{}, cute::Int<kAccumulatorElements>{}), cute::LayoutRight{});
  cute::clear(dkv_accum_frag);

  constexpr int32_t kTailLoaderWarpBase = 2;
  constexpr int32_t kRowStateLoaderWarp = 6;
  const bool is_tail_loader =
    warp_id >= kTailLoaderWarpBase && warp_id < kTailLoaderWarpBase + kBlockMLoadWarps;
  if (is_tail_loader)
  {
    load_q_do_pair<Traits>(
      gQ,
      gDOInt8,
      gDO,
      sQStorage,
      sdOInt8Storage,
      sdOFp16PairStorage,
      0,
      warp_id - kTailLoaderWarpBase,
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

  for (int32_t m_pair_base = 0; m_pair_base < params.seq_len; m_pair_base += 2 * Traits::kBlockM)
  {
    const float q_block_scale = gQScale(m_pair_base / QuantBlockQ);
    const int32_t m_base = m_pair_base + m_half * Traits::kBlockM;
    auto acc_score = cute::partition_fragment_C(score_mma, BlockMNShape{});
    auto acc_dp = cute::partition_fragment_C(half_mma, BlockMNShape{});
    cute::clear(acc_score);
    cute::clear(acc_dp);

    if (n_valid)
    {
#pragma unroll
      for (int32_t dim_base = 0; dim_base < HeadDim; dim_base += Traits::kScoreAtomK)
      {
        constexpr auto score_subtile_shape = cute::make_shape(
          cute::Int<Traits::kBlockM>{}, cute::Int<Traits::kScoreAtomK>{});
        const auto sQTile = cute::local_tile(
          sQ, score_subtile_shape, cute::make_coord(m_half, dim_base / Traits::kScoreAtomK));
        const auto sKTile = cute::local_tile(
          sK,
          cute::make_shape(cute::Int<Traits::kBlockN>{}, cute::Int<Traits::kScoreAtomK>{}),
          cute::make_coord(n_tile, dim_base / Traits::kScoreAtomK));
        auto tSrQ = thr_mma_score.partition_fragment_A(sQTile);
        auto tSrK = thr_mma_score.partition_fragment_B(sKTile);
        cute::copy(tiled_copy_score_a, thr_copy_score_a.partition_S(sQTile), thr_copy_score_a.retile_D(tSrQ));
        cute::copy(tiled_copy_score_b, thr_copy_score_b.partition_S(sKTile), thr_copy_score_b.retile_D(tSrK));
        cute::gemm(thr_mma_score, tSrQ, tSrK, acc_score);
      }

#pragma unroll
      for (int32_t dim_base = 0; dim_base < HeadDim; dim_base += Traits::kBlockK)
      {
        constexpr auto half_subtile_shape = cute::make_shape(
          cute::Int<Traits::kBlockM>{}, cute::Int<Traits::kBlockK>{});
        const auto sdO = cute::local_tile(
          sdOFp16Pair, half_subtile_shape, cute::make_coord(m_half, dim_base / Traits::kBlockK));
        const auto sVTile = cute::local_tile(
          sV,
          cute::make_shape(cute::Int<Traits::kBlockN>{}, cute::Int<Traits::kBlockK>{}),
          cute::make_coord(n_tile, dim_base / Traits::kBlockK));
        const auto sdOHalf = cute::recast<cute::half_t>(sdO);
        const auto sVHalf = cute::recast<cute::half_t>(sVTile);
        auto tdPrdO = thr_mma_half.partition_fragment_A(sdOHalf);
        auto tdPrV = thr_mma_half.partition_fragment_B(sVHalf);
        cute::copy(tiled_copy_half_a, thr_copy_half_a.partition_S(sdOHalf), thr_copy_half_a.retile_D(tdPrdO));
        cute::copy(tiled_copy_half_b, thr_copy_half_b.partition_S(sVHalf), thr_copy_half_b.retile_D(tdPrV));
        cute::gemm(thr_mma_half, tdPrdO, tdPrV, acc_dp);
      }
    }

    auto rPFloat = cute::recast<float>(acc_score);
    float p_max_abs = 0.0f;
    float ds_max_abs = 0.0f;
    const float score_scale = q_block_scale * k_block_scale;
    // BwdTileTraits<64,64,64,8> maps lane bits and fragment bits to the eight C coordinates.
    {
      const int32_t row_state_0 = m_half * Traits::kBlockM + lane_id / 4;
      const int32_t row_state_1 = row_state_0 + 8;
      const float lse_0 = sLse(row_state_0);
      const float lse_1 = sLse(row_state_1);
      const float delta_0 = sDelta(row_state_0);
      const float delta_1 = sDelta(row_state_1);
#pragma unroll
      for (int32_t idx = 0; idx < cute::size(acc_score); ++idx)
      {
        const int32_t m_local = lane_id / 4 + ((idx & 2) != 0 ? 8 : 0);
        const int32_t n_local = (lane_id & 3) * 2 + (idx & 1) + ((idx & 4) != 0 ? 8 : 0);
        const int32_t row = m_base + m_local;
        const int32_t col = n_base + n_local;
        float p = 0.0f;
        float ds = 0.0f;
        if (IsAligned || (row < params.seq_len && col < params.seq_len))
        {
          const bool upper_row = (idx & 2) != 0;
          const float score = static_cast<float>(acc_score(idx)) * score_scale;
          p = expf(score * sm_scale - (upper_row ? lse_1 : lse_0));
          ds = p * (acc_dp(idx) - (upper_row ? delta_1 : delta_0)) * sm_scale;
          p_max_abs = fmaxf(p_max_abs, fabsf(p));
          ds_max_abs = fmaxf(ds_max_abs, fabsf(ds));
        }
        rPFloat(idx) = p;
        acc_dp(idx) = ds;
      }
    }
    p_max_abs = warp_reduce_max(p_max_abs);
    ds_max_abs = warp_reduce_max(ds_max_abs);
    if (lane_id == 0)
    {
      shared.p_scale[warp_id] = p_max_abs / 127.5f + 1.0e-7f;
      shared.ds_scale[warp_id] = ds_max_abs / 127.5f + 1.0e-7f;
    }
    __syncthreads();

    const float p_scale = fmaxf(shared.p_scale[warp_id], shared.p_scale[warp_id ^ 1]);
    float ds_scale = shared.ds_scale[0];
#pragma unroll
    for (int32_t scale_warp = 1; scale_warp < Traits::kNumWarps; ++scale_warp)
    {
      ds_scale = fmaxf(ds_scale, shared.ds_scale[scale_warp]);
    }
    const float inv_p_scale = 1.0f / p_scale;
    const float inv_ds_scale = 1.0f / ds_scale;
    constexpr auto transposed_store_shape = cute::make_shape(
      cute::Int<Traits::kBlockN>{}, cute::Int<Traits::kBlockM>{});
    auto sPTile = cute::local_tile(sPPair, transposed_store_shape, cute::make_coord(cute::_0{}, m_half));
    auto sdSTile = cute::local_tile(sdSPair, transposed_store_shape, cute::make_coord(cute::_0{}, m_half));
    const auto sP = make_transposed_tensor(sPTile);
    const auto sdS = make_transposed_tensor(sdSTile);
    {
      auto rP = cute::make_fragment_like<int8_t>(acc_score);
      auto rdS = cute::make_fragment_like<int8_t>(acc_score);
#pragma unroll
      for (int32_t idx = 0; idx < cute::size(acc_score); ++idx)
      {
        const int32_t m_local = lane_id / 4 + ((idx & 2) != 0 ? 8 : 0);
        const int32_t n_local = (lane_id & 3) * 2 + (idx & 1) + ((idx & 4) != 0 ? 8 : 0);
        const int32_t row = m_base + m_local;
        const int32_t col = n_base + n_local;
        int8_t p_i8 = 0;
        int8_t ds_i8 = 0;
        if (IsAligned || (row < params.seq_len && col < params.seq_len))
        {
          p_i8 = round_to_int8(rPFloat(idx) * inv_p_scale);
          ds_i8 = round_to_int8(acc_dp(idx) * inv_ds_scale);
        }
        rP(idx) = p_i8;
        rdS(idx) = ds_i8;
      }
      auto tCrP = thr_copy_score_c.retile_S(rP);
      auto tCrdS = thr_copy_score_c.retile_S(rdS);
      auto tCsP = thr_copy_score_c.partition_D(sP);
      auto tCsdS = thr_copy_score_c.partition_D(sdS);
      cute::copy(tiled_copy_score_c, tCrP, tCsP);
      cute::copy(tiled_copy_score_c, tCrdS, tCsdS);
      constexpr auto score_store_shape = cute::make_shape(
        cute::Int<Traits::kBlockM>{}, cute::Int<Traits::kBlockN>{});
      auto sdSMirrorHalf = cute::local_tile(sdSMirror, score_store_shape, cute::make_coord(cute::_0{}, n_tile & 1));
      auto tCsdSMirror = thr_copy_score_c.partition_D(sdSMirrorHalf);
      cute::copy(tiled_copy_score_c, tCrdS, tCsdSMirror);
    }
    __syncthreads();

    if (n_valid && m_half == 0)
    {
#pragma unroll
      for (int32_t dim_base = 0; dim_base < HeadDim; dim_base += Traits::kBlockK)
      {
        constexpr auto do_pair_shape = cute::make_shape(
          cute::Int<2 * Traits::kBlockM>{}, cute::Int<Traits::kBlockK>{});
        const auto sdOPair = cute::local_tile(
          sdOInt8, do_pair_shape, cute::make_coord(cute::_0{}, dim_base / Traits::kBlockK));
        const auto sdOdV = qattn_cutlass::make_int8_transposed_b_view(sdOPair);
        auto dkv_acc = cute::partition_fragment_C(score_mma, BlockMNShape{});
        cute::clear(dkv_acc);
        auto tDvP = thr_mma_score.partition_fragment_A(sPPair);
        auto tDvdO = thr_mma_score.partition_fragment_B(sdOdV);
        cute::copy(tiled_copy_score_a, thr_copy_score_a.partition_S(sPPair), thr_copy_score_a.retile_D(tDvP));
        cute::copy(tiled_copy_transposed_b, thr_copy_transposed_b.partition_S(sdOdV), thr_copy_transposed_b.retile_D(tDvdO));
        cute::gemm(thr_mma_score, tDvP, tDvdO, dkv_acc);
        const float do_scale = gDOScale[
          (m_pair_base / (2 * Traits::kBlockM)) * kDimBlocks + dim_base / Traits::kBlockK];
        const float dv_scale = p_scale * do_scale;
#pragma unroll
        for (int32_t idx = 0; idx < cute::size(dkv_acc); ++idx)
        {
          const int32_t dim_block = dim_base / Traits::kBlockK;
          const float contribution = static_cast<float>(dkv_acc(idx)) * dv_scale;
          dkv_accum_frag(dim_block, idx) += contribution;
        }
      }
    }

    if (n_valid && m_half == 1)
    {
#pragma unroll
      for (int32_t dim_base = 0; dim_base < HeadDim; dim_base += Traits::kBlockK)
      {
        constexpr auto q_pair_shape = cute::make_shape(
          cute::Int<2 * Traits::kBlockM>{}, cute::Int<Traits::kBlockK>{});
        const auto sQPair = cute::local_tile(
          sQ, q_pair_shape, cute::make_coord(cute::_0{}, dim_base / Traits::kBlockK));
        const auto sQdK = qattn_cutlass::make_int8_transposed_b_view(sQPair);
        auto dkv_acc = cute::partition_fragment_C(score_mma, BlockMNShape{});
        cute::clear(dkv_acc);
        auto tDkdS = thr_mma_score.partition_fragment_A(sdSPair);
        auto tDkQ = thr_mma_score.partition_fragment_B(sQdK);
        cute::copy(tiled_copy_score_a, thr_copy_score_a.partition_S(sdSPair), thr_copy_score_a.retile_D(tDkdS));
        cute::copy(tiled_copy_transposed_b, thr_copy_transposed_b.partition_S(sQdK), thr_copy_transposed_b.retile_D(tDkQ));
        cute::gemm(thr_mma_score, tDkdS, tDkQ, dkv_acc);
#pragma unroll
        for (int32_t idx = 0; idx < cute::size(dkv_acc); ++idx)
        {
          const int32_t dim_block = dim_base / Traits::kBlockK;
          const float contribution = static_cast<float>(dkv_acc(idx)) * (ds_scale * q_block_scale);
          dkv_accum_frag(dim_block, idx) += contribution;
        }
      }
    }

    __syncthreads();
    const int32_t next_m_pair_base = m_pair_base + 2 * Traits::kBlockM;
    const bool has_next_m_pair = next_m_pair_base < params.seq_len;
    if (has_next_m_pair && is_tail_loader)
    {
      load_q_do_pair<Traits>(
        gQ,
        gDOInt8,
        gDO,
        sQStorage,
        sdOInt8Storage,
        sdOFp16PairStorage,
        next_m_pair_base,
        warp_id - kTailLoaderWarpBase,
        lane_id);
    }
    if (has_next_m_pair && warp_id == kRowStateLoaderWarp)
    {
      load_row_state_pair(
        gLse, gDelta, sLse, sDelta, next_m_pair_base, params.seq_len, lane_id);
    }

    if ((n_tile & 1) == 0)
    {
      const int32_t dq_dim_begin = n_pair * kDqDimBlocksPerOwner * Traits::kBlockK;
#pragma unroll
      for (int32_t dim_offset = 0;
           dim_offset < kDqDimBlocksPerOwner * Traits::kBlockK;
           dim_offset += Traits::kBlockK)
      {
        const int32_t dim_base = dq_dim_begin + dim_offset;
        auto dq_acc = cute::partition_fragment_C(score_mma, BlockMNShape{});
        cute::clear(dq_acc);
#pragma unroll
        for (int32_t dq_pair = 0; dq_pair < kDqNPairs; ++dq_pair)
        {
          const int32_t dq_ds_slot = 2 * dq_pair + m_half;
          auto sDSScratchStorage = make_smem_tensor(
            shared.warp_scratch.dsk_dkv,
            typename Traits::SmemLayoutdSkdKV{},
            dq_ds_slot * warp_scratch_offset);
          auto sDSScratch = cute::recast<int8_t>(sDSScratchStorage);
          auto sdSDQ = cute::local_tile(sDSScratch, ds_tile_shape, cute::make_coord(cute::_0{}, cute::_0{}));
          constexpr auto k_pair_shape = cute::make_shape(
            cute::Int<2 * Traits::kBlockN>{}, cute::Int<Traits::kBlockK>{});
          const auto sKPair = cute::local_tile(
            sK, k_pair_shape, cute::make_coord(dq_pair, dim_base / Traits::kBlockK));
          const auto sKdQ = qattn_cutlass::make_int8_transposed_b_view(sKPair);
          auto tDqdS = thr_mma_score.partition_fragment_A(sdSDQ);
          auto tDqK = thr_mma_score.partition_fragment_B(sKdQ);
          cute::copy(tiled_copy_score_a, thr_copy_score_a.partition_S(sdSDQ), thr_copy_score_a.retile_D(tDqdS));
          load_packed_mma_b_fragment(
            shared.k_i8_mma_b,
            tDqK,
            dq_pair * kDimBlocks + dim_base / Traits::kBlockK,
            lane_id);
          cute::gemm(thr_mma_score, tDqdS, tDqK, dq_acc);
        }
        static_assert(
          score_pair_offset * sizeof(cute::uint128_t) >= 16 * 16 * sizeof(float),
          "Each dead score-pair slot must fit one warp-local dQ transpose");
        const int32_t dq_stage_slot = n_tile + m_half;
        auto *const dq_stage = reinterpret_cast<float *>(
          shared.score_pair_i8.begin() + dq_stage_slot * score_pair_offset);
        accumulate_dq_fragment_shared_contiguous<IsAligned, Traits>(
          dq_acc,
          ds_scale * k_block_scale,
          dq_stage,
          DQAccum + (batch * params.num_heads + head) * params.seq_len * HeadDim,
          m_base,
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

  half *const dkv_stage = reinterpret_cast<half *>(
    shared.warp_scratch.dsk_dkv.begin() + n_tile * warp_scratch_offset);
  if (n_valid && m_half == 0)
  {
#pragma unroll
    for (int32_t dim_block = 0; dim_block < kDimBlocks; ++dim_block)
    {
      store_dkv_fragment_coalesced<Traits>(
        dkv_accum_frag, dim_block, gDV, dkv_stage, n_base, params.seq_len, lane_id);
    }
  }
  __syncthreads();
  if (n_valid && m_half == 1)
  {
#pragma unroll
    for (int32_t dim_block = 0; dim_block < kDimBlocks; ++dim_block)
    {
      store_dkv_fragment_coalesced<Traits>(
        dkv_accum_frag, dim_block, gDK, dkv_stage, n_base, params.seq_len, lane_id);
    }
  }
}

template <int32_t HeadDim, int32_t CtaM = 16, int32_t CtaN = 16, int32_t NumWarps = 1>
__global__ void fused_mma_kernel(const int8_t *__restrict__ const Q,
                                 const int8_t *__restrict__ const K,
                                 const float *__restrict__ const QScale,
                                 const float *__restrict__ const KScale,
                                 const half *__restrict__ const V,
                                 const half *__restrict__ const DO,
                                 const int8_t *__restrict__ const DOInt8,
                                 const float *__restrict__ const DOScale,
                                 const float *__restrict__ const Lse,
                                 const float *__restrict__ const Delta,
                                 float *__restrict__ const DQAccum,
                                 half *__restrict__ const DK,
                                 half *__restrict__ const DV,
                                 const Params params,
                                 const float sm_scale)
{
  using Traits = BwdTileTraits<HeadDim, CtaM, CtaN, NumWarps>;
  constexpr bool kDirectDskPairQuantization = HeadDim == 64 && CtaM == 64 && CtaN == 64 && NumWarps == 4;
  static_assert(Traits::kCtaMMicroTiles % 2 == 0, "Backward CTA M microtiles must pair cleanly for dV/dK k32");
  static_assert(Traits::kNumWarps % 2 == 0 && Traits::kCtaNMicroTiles % 2 == 0, "Backward dQ ownership requires adjacent N-warp pairs");
  const int32_t n_block = blockIdx.x;
  const int32_t head = blockIdx.y;
  const int32_t batch = blockIdx.z;
  const ThreadCoord thread_coord = make_thread_coord();
  const int32_t lane_id = thread_coord.lane_id;
  const int32_t warp_id = thread_coord.warp_id;
  const int32_t n_cta_base = n_block * Traits::kCtaN;
  const int32_t n_smem_slot = warp_id % Traits::kSmemWarps;
  const int32_t dq_pair_smem_slot = n_smem_slot & ~1;

  if (n_cta_base >= params.seq_len)
  {
    return;
  }

  const auto gQ = make_head_matrix_view<HeadDim>(Q, params, batch, head, params.stride_bz_q, params.stride_seq_q, params.stride_h_q);
  const auto gK = make_head_matrix_view<HeadDim>(K, params, batch, head, params.stride_bz_k, params.stride_seq_k, params.stride_h_k);
  const auto gV = make_head_matrix_view<HeadDim>(V, params, batch, head, params.stride_bz_v, params.stride_seq_v, params.stride_h_v);
  const auto gDO = make_head_matrix_view<HeadDim>(DO, params, batch, head, params.stride_bz_do, params.stride_seq_do, params.stride_h_do);
  const auto gDOInt8 = make_workspace_matrix_view<HeadDim>(DOInt8, params, batch, head);
  const auto gDQAccum = make_workspace_matrix_view<HeadDim>(DQAccum, params, batch, head);
  const auto gDK = make_head_matrix_view<HeadDim>(DK, params, batch, head, params.stride_bz_dk, params.stride_seq_dk, params.stride_h_dk);
  const auto gDV = make_head_matrix_view<HeadDim>(DV, params, batch, head, params.stride_bz_dv, params.stride_seq_dv, params.stride_h_dv);
  const auto gLse = make_head_vector_view(Lse, params, batch, head);
  const auto gDelta = make_head_vector_view(Delta, params, batch, head);
  constexpr int32_t kDimBlocks = HeadDim / Traits::kBlockK;
  const int32_t pair_blocks = (params.seq_len + 2 * Traits::kBlockM - 1) / (2 * Traits::kBlockM);
  const float *const gDOScale = DOScale + (batch * params.num_heads + head) * pair_blocks * kDimBlocks;
  const int32_t q_scale_extent = (params.seq_len + params.blk_q - 1) / params.blk_q;
  const int32_t k_scale_extent = (params.seq_len + params.blk_k - 1) / params.blk_k;
  const auto gQScale = make_head_strided_vector_view(QScale, batch, head, params.stride_bz_q_scale, params.stride_h_q_scale, q_scale_extent);
  const auto gKScale = make_head_strided_vector_view(KScale, batch, head, params.stride_bz_k_scale, params.stride_h_k_scale, k_scale_extent);

  extern __shared__ char shared_storage[];
  auto &shared = *reinterpret_cast<SharedStorage<Traits> *>(shared_storage);

  constexpr int32_t score_pair_offset = cute::cosize_v<typename Traits::SmemLayoutScorePair>;
  constexpr int32_t warp_scratch_offset = cute::cosize_v<typename Traits::SmemLayoutdSkdKV>;
  constexpr int32_t k_scale_offset = cute::cosize_v<typename Traits::SmemLayoutN>;
  constexpr int32_t kBlockMLoadRows = Traits::kMStageLoadRows;
  constexpr int32_t kBlockMLoadWarps = Traits::kMStageLoadWarps;
  constexpr int32_t kCtaNLoadRows = Traits::kCtaNLoadRows;

  auto sQStorage = make_smem_tensor(shared.q_i8, typename Traits::SmemLayoutQ{});
  auto sdOInt8Storage = make_smem_tensor(shared.do_i8, typename Traits::SmemLayoutdO{});
  auto sKStorage = make_smem_tensor(shared.k_i8, typename Traits::SmemLayoutK{});
  auto sQ = cute::recast<int8_t>(sQStorage);
  auto sdOInt8 = cute::recast<int8_t>(sdOInt8Storage);
  auto sK = cute::recast<int8_t>(sKStorage);
  auto sWarpScratchStorage = make_smem_tensor(shared.warp_scratch.dsk_dkv, typename Traits::SmemLayoutdSkdKV{}, dq_pair_smem_slot * warp_scratch_offset);
  auto sWarpScratch = cute::recast<int8_t>(sWarpScratchStorage);
  constexpr auto dsk_tile_shape = cute::make_shape(cute::Int<Traits::kBlockM>{}, cute::Int<Traits::kScoreAtomK>{});
  auto sdSk = cute::local_tile(sWarpScratch, dsk_tile_shape, cute::make_coord(cute::_0{}, cute::_0{}));
  auto sScorePairStorage = make_smem_tensor(shared.score_pair_i8, typename Traits::SmemLayoutScorePair{}, n_smem_slot * score_pair_offset);
  auto sScorePair = cute::recast<int8_t>(sScorePairStorage);
  constexpr auto score_pair_tile_shape = cute::make_shape(cute::Int<Traits::kBlockN>{}, cute::Int<Traits::kScoreAtomK>{});
  auto sPPair = cute::local_tile(sScorePair, score_pair_tile_shape, cute::make_coord(cute::_0{}, cute::_0{}));
  auto sdSqPair = cute::local_tile(sScorePair, score_pair_tile_shape, cute::make_coord(cute::_0{}, cute::_1{}));
  auto sVStorage = make_smem_tensor(shared.v, typename Traits::SmemLayoutV{});
  auto sdOFp16Storage = make_smem_tensor(shared.do_fp16, typename Traits::SmemLayoutdOFp16{});
  auto sV = cute::recast<half>(sVStorage);
  auto sdOFp16 = cute::recast<half>(sdOFp16Storage);
  auto sQScale = make_smem_tensor(shared.q_scale, typename Traits::SmemLayoutM{});
  auto sKScale = make_smem_tensor(shared.k_scale, typename Traits::SmemLayoutN{}, n_smem_slot * k_scale_offset);
  auto sLse = make_smem_tensor(shared.lse, typename Traits::SmemLayoutM{});
  auto sDelta = make_smem_tensor(shared.delta, typename Traits::SmemLayoutM{});

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

  const auto acc_coord = thr_mma_score.partition_C(cute::make_identity_tensor(BlockMNShape{}));

  constexpr auto int8_pair_load_shape = cute::make_shape(cute::Int<kBlockMLoadRows>{}, cute::Int<Traits::kInt8LoadVecCols>{});
  constexpr auto do_load_shape = cute::make_shape(cute::Int<kBlockMLoadRows>{}, cute::Int<Traits::kHalfLoadVecCols>{});
  constexpr auto cta_k_load_shape = cute::make_shape(cute::Int<kCtaNLoadRows>{}, cute::Int<Traits::kInt8LoadVecCols>{});
  constexpr auto cta_v_load_shape = cute::make_shape(cute::Int<kCtaNLoadRows>{}, cute::Int<Traits::kHalfLoadVecCols>{});
  auto sKLoad = cute::local_tile(sKStorage, cta_k_load_shape, cute::make_coord(warp_id, cute::_0{}));
  auto sVLoad = cute::local_tile(sVStorage, cta_v_load_shape, cute::make_coord(warp_id, cute::_0{}));
  const int32_t n_load_base = n_cta_base + warp_id * kCtaNLoadRows;
  load_k_tile<Traits>(gK, sKLoad, n_load_base, lane_id);
  load_v_tile<Traits>(gV, sVLoad, n_load_base, lane_id);
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();

  for (int32_t n_wave_base = 0; n_wave_base < Traits::kCtaNMicroTiles; n_wave_base += Traits::kNumWarps)
  {
    if (n_cta_base + n_wave_base * Traits::kBlockN >= params.seq_len)
    {
      break;
    }
    const int32_t n_tile = n_wave_base + warp_id;
    const int32_t n_base = n_cta_base + n_tile * Traits::kBlockN;
    const bool n_valid = n_tile < Traits::kCtaNMicroTiles && n_base < params.seq_len;

    auto dk_accum_frag = cute::make_tensor<float>(cute::make_shape(cute::Int<kDimBlocks>{}, cute::size(acc_coord)), cute::LayoutRight{});
    auto dv_accum_frag = cute::make_tensor<float>(cute::make_shape(cute::Int<kDimBlocks>{}, cute::size(acc_coord)), cute::LayoutRight{});
    cute::clear(dk_accum_frag);
    cute::clear(dv_accum_frag);

    if (n_valid)
    {
      for (int32_t n_local = lane_id; n_local < Traits::kBlockN; n_local += warpSize)
      {
        const int32_t col = n_base + n_local;
        sKScale(n_local) = col < params.seq_len ? key_scale_for_row(gKScale, params, col) : 0.0f;
      }
    }
    __syncwarp();

    for (int32_t m_cta_base = 0; m_cta_base < params.seq_len; m_cta_base += Traits::kCtaM)
    {
      float pending_dv_p_scale = 0.0f;
      float pending_dk_dsq_scale = 0.0f;
      int32_t pending_dv_m_base = m_cta_base;
#pragma unroll
      for (int32_t m_tile = 0; m_tile < Traits::kCtaMMicroTiles; ++m_tile)
      {
        const int32_t m_base = m_cta_base + m_tile * Traits::kBlockM;
        if (m_base >= params.seq_len)
        {
          break;
        }
        __syncthreads();
        auto acc_score = cute::partition_fragment_C(score_mma, BlockMNShape{});
        auto acc_dp = cute::partition_fragment_C(half_mma, BlockMNShape{});
        cute::clear(acc_score);
        cute::clear(acc_dp);

        if (warp_id < kBlockMLoadWarps)
        {
          auto sdOLoad = cute::local_tile(sdOFp16Storage, do_load_shape, cute::make_coord(warp_id, cute::_0{}));
          const int32_t m_load_base = m_base + warp_id * kBlockMLoadRows;
          if ((m_tile & 1) == 0)
          {
#pragma unroll
            for (int32_t pair_half = 0; pair_half < 2; ++pair_half)
            {
              auto sQLoad = cute::local_tile(sQStorage, int8_pair_load_shape, cute::make_coord(warp_id + pair_half * kBlockMLoadWarps, cute::_0{}));
              auto sdOInt8Load = cute::local_tile(sdOInt8Storage, int8_pair_load_shape, cute::make_coord(warp_id + pair_half * kBlockMLoadWarps, cute::_0{}));
              load_q_tile<Traits>(gQ, sQLoad, m_load_base + pair_half * Traits::kBlockM, lane_id);
              load_do_int8_tile<Traits>(gDOInt8, sdOInt8Load, m_load_base + pair_half * Traits::kBlockM, lane_id);
            }
          }
          load_do_fp16_tile<Traits>(gDO, sdOLoad, m_load_base, lane_id);
          cute::cp_async_fence();
          for (int32_t row_offset = lane_id; row_offset < kBlockMLoadRows; row_offset += warpSize)
          {
            const int32_t m_local = warp_id * kBlockMLoadRows + row_offset;
            const int32_t row = m_base + m_local;
            if (row < params.seq_len)
            {
              sQScale(m_local) = query_scale_for_row(gQScale, params, row);
              sLse(m_local) = gLse(row);
              sDelta(m_local) = gDelta(row);
            }
            else
            {
              sQScale(m_local) = 0.0f;
              sLse(m_local) = 0.0f;
              sDelta(m_local) = 0.0f;
            }
          }
          cute::cp_async_wait<0>();
        }
        __syncthreads();

        const bool dv_upper_half = (m_tile & 1) != 0;
        const bool dv_lower_tail = !dv_upper_half && (m_base + Traits::kBlockM >= params.seq_len);
        const bool dv_ready = dv_upper_half || dv_lower_tail;
        const int32_t dv_col_offset = dv_upper_half ? Traits::kBlockM : 0;
        const int32_t dv_pair_m_base = dv_upper_half ? pending_dv_m_base : m_base;
        const bool dk_upper_half = dv_upper_half;
        const bool dk_ready = dv_ready;
        const int32_t dk_col_offset = dk_upper_half ? Traits::kBlockM : 0;
        const bool dq_pair_owner = n_valid && ((n_tile & 1) == 0);
        float p_scale = 0.0f;
        float dv_p_scale = 0.0f;
        float dsq_scale = 0.0f;
        float dk_dsq_scale = 0.0f;
        float dsk_scale = 0.0f;
        if (n_valid)
        {
#pragma unroll
          for (int32_t dim_base = 0; dim_base < HeadDim; dim_base += Traits::kScoreAtomK)
          {
            constexpr auto score_subtile_shape = cute::make_shape(cute::Int<Traits::kBlockM>{}, cute::Int<Traits::kScoreAtomK>{});
            const auto sQTile = cute::local_tile(sQ, score_subtile_shape, cute::make_coord(m_tile & 1, dim_base / Traits::kScoreAtomK));
            const auto sKTile = cute::local_tile(sK, cute::make_shape(cute::Int<Traits::kBlockN>{}, cute::Int<Traits::kScoreAtomK>{}), cute::make_coord(n_tile, dim_base / Traits::kScoreAtomK));

            auto tSrQ = thr_mma_score.partition_fragment_A(sQTile);
            auto tSrK = thr_mma_score.partition_fragment_B(sKTile);
            cute::copy(tiled_copy_score_a, thr_copy_score_a.partition_S(sQTile), thr_copy_score_a.retile_D(tSrQ));
            cute::copy(tiled_copy_score_b, thr_copy_score_b.partition_S(sKTile), thr_copy_score_b.retile_D(tSrK));
            cute::gemm(thr_mma_score, tSrQ, tSrK, acc_score);
          }

#pragma unroll
          for (int32_t dim_base = 0; dim_base < HeadDim; dim_base += Traits::kBlockK)
          {
            constexpr auto a_subtile_shape = cute::make_shape(cute::Int<Traits::kBlockM>{}, cute::Int<Traits::kBlockK>{});
            const auto sdO = cute::local_tile(sdOFp16, a_subtile_shape, cute::make_coord(cute::_0{}, dim_base / Traits::kBlockK));
            const auto sVTile = cute::local_tile(sV, cute::make_shape(cute::Int<Traits::kBlockN>{}, cute::Int<Traits::kBlockK>{}), cute::make_coord(n_tile, dim_base / Traits::kBlockK));
            const auto sVHalf = cute::recast<cute::half_t>(sVTile);
            const auto sdOHalf = cute::recast<cute::half_t>(sdO);

            auto tdPrdO = thr_mma_half.partition_fragment_A(sdOHalf);
            auto tdPrV = thr_mma_half.partition_fragment_B(sVHalf);
            cute::copy(tiled_copy_half_a, thr_copy_half_a.partition_S(sdOHalf), thr_copy_half_a.retile_D(tdPrdO));
            cute::copy(tiled_copy_half_b, thr_copy_half_b.partition_S(sVHalf), thr_copy_half_b.retile_D(tdPrV));
            cute::gemm(thr_mma_half, tdPrdO, tdPrV, acc_dp);
          }

        }
        __syncthreads();
        constexpr auto score_store_shape = cute::make_shape(cute::Int<Traits::kBlockM>{}, cute::Int<Traits::kBlockN>{});
        auto sdSkTile = cute::local_tile(sdSk, score_store_shape, cute::make_coord(cute::_0{}, n_tile & 1));
        auto rdSk = cute::make_fragment_like<int8_t>(acc_score);
        cute::clear(rdSk);
        if (n_valid)
        {
          auto rPFloat = cute::recast<float>(acc_score);
          float p_max_abs = 0.0f;
          float dsq_max_abs = 0.0f;
          float dsk_max_abs = 0.0f;
#pragma unroll
          for (int32_t idx = 0; idx < cute::size(acc_score); ++idx)
          {
            const auto coord = acc_coord(idx);
            const int32_t m_local = static_cast<int32_t>(cute::get<0>(coord));
            const int32_t n_local = static_cast<int32_t>(cute::get<1>(coord));
            const int32_t row = m_base + m_local;
            const int32_t col = n_base + n_local;
            float p = 0.0f;
            float ds = 0.0f;
            if (row < params.seq_len && col < params.seq_len)
            {
              const float q_scale = sQScale(m_local);
              const float k_scale = sKScale(n_local);
              const float score = static_cast<float>(acc_score(idx)) * q_scale * k_scale;
              p = expf(score * sm_scale - sLse(m_local));
              ds = p * (acc_dp(idx) - sDelta(m_local)) * sm_scale;
              p_max_abs = fmaxf(p_max_abs, fabsf(p));
              dsq_max_abs = fmaxf(dsq_max_abs, fabsf(ds * q_scale));
              dsk_max_abs = fmaxf(dsk_max_abs, fabsf(ds * k_scale));
            }
            rPFloat(idx) = p;
            acc_dp(idx) = ds;
          }
          p_max_abs = warp_reduce_max(p_max_abs);
          dsq_max_abs = warp_reduce_max(dsq_max_abs);
          dsk_max_abs = warp_reduce_max(dsk_max_abs);
          p_scale = __shfl_sync(0xffffffff, p_max_abs / 127.5f + 1.0e-7f, 0);
          dsq_scale = __shfl_sync(0xffffffff, dsq_max_abs / 127.5f + 1.0e-7f, 0);
          dsk_scale = __shfl_sync(0xffffffff, dsk_max_abs / 127.5f + 1.0e-7f, 0);

          dv_p_scale = dv_upper_half ? fmaxf(pending_dv_p_scale, p_scale) : p_scale;
          dk_dsq_scale = dk_upper_half ? fmaxf(pending_dk_dsq_scale, dsq_scale) : dsq_scale;
          const float inv_p_scale = 1.0f / dv_p_scale;
          const float inv_dsq_scale = 1.0f / dk_dsq_scale;
          const float inv_dsk_scale = 1.0f / dsk_scale;
          constexpr auto transposed_score_store_shape = cute::make_shape(cute::Int<Traits::kBlockN>{}, cute::Int<Traits::kBlockM>{});
          auto sPTile = cute::local_tile(sPPair, transposed_score_store_shape, cute::make_coord(cute::_0{}, dv_col_offset / Traits::kBlockM));
          auto sdSqTile = cute::local_tile(sdSqPair, transposed_score_store_shape, cute::make_coord(cute::_0{}, dk_col_offset / Traits::kBlockM));
          const auto sP = make_transposed_tensor(sPTile);
          const auto sdSq = make_transposed_tensor(sdSqTile);
          auto rP = cute::make_fragment_like<int8_t>(acc_score);
          auto rdSq = cute::make_fragment_like<int8_t>(acc_score);
#pragma unroll
          for (int32_t idx = 0; idx < cute::size(acc_score); ++idx)
          {
            const auto coord = acc_coord(idx);
            const int32_t m_local = static_cast<int32_t>(cute::get<0>(coord));
            const int32_t n_local = static_cast<int32_t>(cute::get<1>(coord));
            const int32_t row = m_base + m_local;
            const int32_t col = n_base + n_local;
            int8_t p_i8 = 0;
            int8_t dsq_i8 = 0;
            int8_t dsk_i8 = 0;
            if (row < params.seq_len && col < params.seq_len)
            {
              const float q_scale = sQScale(m_local);
              const float p = rPFloat(idx);
              const float ds = acc_dp(idx);
              p_i8 = round_to_int8(p * inv_p_scale);
              dsq_i8 = round_to_int8(ds * q_scale * inv_dsq_scale);
              if constexpr (!kDirectDskPairQuantization)
              {
                dsk_i8 = round_to_int8(ds * sKScale(n_local) * inv_dsk_scale);
              }
            }
            rP(idx) = p_i8;
            rdSq(idx) = dsq_i8;
            if constexpr (!kDirectDskPairQuantization)
            {
              rdSk(idx) = dsk_i8;
            }
          }
          auto tCrP = thr_copy_score_c.retile_S(rP);
          auto tCrdSq = thr_copy_score_c.retile_S(rdSq);
          auto tCsP = thr_copy_score_c.partition_D(sP);
          auto tCsdSq = thr_copy_score_c.partition_D(sdSq);
          cute::copy(tiled_copy_score_c, tCrP, tCsP);
          cute::copy(tiled_copy_score_c, tCrdSq, tCsdSq);
          if (dk_upper_half)
          {
            auto sdSqLowerTile = cute::local_tile(sdSqPair, transposed_score_store_shape, cute::make_coord(cute::_0{}, cute::_0{}));
            const auto sdSqLower = make_transposed_tensor(sdSqLowerTile);
            auto rdSqRescale = cute::make_fragment_like<int8_t>(acc_score);
            auto tCsdSqLower = thr_copy_score_c.partition_S(sdSqLower);
            auto tCrdSqRescale = thr_copy_score_c.retile_D(rdSqRescale);
            cute::copy(tiled_copy_score_c, tCsdSqLower, tCrdSqRescale);
            const float scale = pending_dk_dsq_scale / dk_dsq_scale;
#pragma unroll
            for (int32_t idx = 0; idx < cute::size(rdSqRescale); ++idx)
            {
              rdSqRescale(idx) = round_to_int8(static_cast<float>(rdSqRescale(idx)) * scale);
            }
            auto tCrdSqStore = thr_copy_score_c.retile_S(rdSqRescale);
            auto tCsdSqStore = thr_copy_score_c.partition_D(sdSqLower);
            cute::copy(tiled_copy_score_c, tCrdSqStore, tCsdSqStore);
          }
          else
          {
            pending_dk_dsq_scale = dsq_scale;
            if (dv_lower_tail)
            {
              auto rdSqZero = cute::make_fragment_like<int8_t>(acc_score);
              cute::clear(rdSqZero);
              auto sdSqUpperTile = cute::local_tile(sdSqPair, transposed_score_store_shape, cute::make_coord(cute::_0{}, cute::_1{}));
              const auto sdSqUpper = make_transposed_tensor(sdSqUpperTile);
              auto tCrdSqZero = thr_copy_score_c.retile_S(rdSqZero);
              auto tCsdSqZero = thr_copy_score_c.partition_D(sdSqUpper);
              cute::copy(tiled_copy_score_c, tCrdSqZero, tCsdSqZero);
            }
          }
          if (dv_upper_half)
          {
            auto sPLowerTile = cute::local_tile(sPPair, transposed_score_store_shape, cute::make_coord(cute::_0{}, cute::_0{}));
            const auto sPLower = make_transposed_tensor(sPLowerTile);
            auto rPRescale = cute::make_fragment_like<int8_t>(acc_score);
            auto tCsPLower = thr_copy_score_c.partition_S(sPLower);
            auto tCrPRescale = thr_copy_score_c.retile_D(rPRescale);
            cute::copy(tiled_copy_score_c, tCsPLower, tCrPRescale);
            const float scale = pending_dv_p_scale / dv_p_scale;
#pragma unroll
            for (int32_t idx = 0; idx < cute::size(rPRescale); ++idx)
            {
              rPRescale(idx) = round_to_int8(static_cast<float>(rPRescale(idx)) * scale);
            }
            auto tCrPStore = thr_copy_score_c.retile_S(rPRescale);
            auto tCsPStore = thr_copy_score_c.partition_D(sPLower);
            cute::copy(tiled_copy_score_c, tCrPStore, tCsPStore);
          }
          else
          {
            pending_dv_p_scale = p_scale;
            pending_dv_m_base = m_base;
            if (dv_lower_tail)
            {
              auto rPZero = cute::make_fragment_like<int8_t>(acc_score);
              cute::clear(rPZero);
              auto sPUpperTile = cute::local_tile(sPPair, transposed_score_store_shape, cute::make_coord(cute::_0{}, cute::_1{}));
              const auto sPUpper = make_transposed_tensor(sPUpperTile);
              auto tCrPZero = thr_copy_score_c.retile_S(rPZero);
              auto tCsPZero = thr_copy_score_c.partition_D(sPUpper);
              cute::copy(tiled_copy_score_c, tCrPZero, tCsPZero);
            }
          }
        }
        if constexpr (kDirectDskPairQuantization)
        {
          if (lane_id == 0)
          {
            shared.dsk_scale[n_smem_slot] = dsk_scale;
          }
          __syncthreads();

          if (n_valid)
          {
            dsk_scale = fmaxf(shared.dsk_scale[dq_pair_smem_slot], shared.dsk_scale[dq_pair_smem_slot + 1]);
            const float inv_dsk_pair_scale = 1.0f / dsk_scale;
#pragma unroll
            for (int32_t idx = 0; idx < cute::size(acc_score); ++idx)
            {
              const auto coord = acc_coord(idx);
              const int32_t m_local = static_cast<int32_t>(cute::get<0>(coord));
              const int32_t n_local = static_cast<int32_t>(cute::get<1>(coord));
              const int32_t row = m_base + m_local;
              const int32_t col = n_base + n_local;
              int8_t dsk_i8 = 0;
              if (row < params.seq_len && col < params.seq_len)
              {
                dsk_i8 = round_to_int8(acc_dp(idx) * sKScale(n_local) * inv_dsk_pair_scale);
              }
              rdSk(idx) = dsk_i8;
            }
            auto tCrdSk = thr_copy_score_c.retile_S(rdSk);
            auto tCsdSk = thr_copy_score_c.partition_D(sdSkTile);
            cute::copy(tiled_copy_score_c, tCrdSk, tCsdSk);
          }
          __syncthreads();
        }
        else
        {
          auto tCrdSk = thr_copy_score_c.retile_S(rdSk);
          auto tCsdSk = thr_copy_score_c.partition_D(sdSkTile);
          cute::copy(tiled_copy_score_c, tCrdSk, tCsdSk);
          if (lane_id == 0)
          {
            shared.dsk_scale[n_smem_slot] = dsk_scale;
          }
          __syncthreads();

          if (dq_pair_owner)
          {
            dsk_scale = fmaxf(shared.dsk_scale[dq_pair_smem_slot], shared.dsk_scale[dq_pair_smem_slot + 1]);
#pragma unroll
            for (int32_t pair_half = 0; pair_half < 2; ++pair_half)
            {
              auto sdSkHalf = cute::local_tile(sdSk, score_store_shape, cute::make_coord(cute::_0{}, pair_half));
              auto rdSkRescale = cute::make_fragment_like<int8_t>(acc_score);
              auto tCsdSkHalf = thr_copy_score_c.partition_S(sdSkHalf);
              auto tCrdSkRescale = thr_copy_score_c.retile_D(rdSkRescale);
              cute::copy(tiled_copy_score_c, tCsdSkHalf, tCrdSkRescale);
              const float scale = shared.dsk_scale[dq_pair_smem_slot + pair_half] / dsk_scale;
#pragma unroll
              for (int32_t idx = 0; idx < cute::size(rdSkRescale); ++idx)
              {
                rdSkRescale(idx) = round_to_int8(static_cast<float>(rdSkRescale(idx)) * scale);
              }
              auto tCrdSkStore = thr_copy_score_c.retile_S(rdSkRescale);
              auto tCsdSkStore = thr_copy_score_c.partition_D(sdSkHalf);
              cute::copy(tiled_copy_score_c, tCrdSkStore, tCsdSkStore);
            }
          }
          __syncwarp();
        }

        if (dv_ready)
        {
#pragma unroll
          for (int32_t dim_base = 0; dim_base < HeadDim; dim_base += Traits::kBlockK)
          {
            constexpr auto do_pair_shape = cute::make_shape(cute::Int<2 * Traits::kBlockM>{}, cute::Int<Traits::kBlockK>{});
            const auto sdOPair = cute::local_tile(sdOInt8, do_pair_shape, cute::make_coord(cute::_0{}, dim_base / Traits::kBlockK));
            const auto sdOdV = qattn_cutlass::make_int8_transposed_b_view(sdOPair);
            const float do_scale = gDOScale[(dv_pair_m_base / (2 * Traits::kBlockM)) * kDimBlocks + dim_base / Traits::kBlockK];

            if (n_valid)
            {
              auto dkv_acc = cute::partition_fragment_C(score_mma, BlockMNShape{});
              cute::clear(dkv_acc);
              auto tDvP = thr_mma_score.partition_fragment_A(sPPair);
              auto tDvdO = thr_mma_score.partition_fragment_B(sdOdV);
              cute::copy(tiled_copy_score_a, thr_copy_score_a.partition_S(sPPair), thr_copy_score_a.retile_D(tDvP));
              cute::copy(tiled_copy_transposed_b, thr_copy_transposed_b.partition_S(sdOdV), thr_copy_transposed_b.retile_D(tDvdO));
              cute::gemm(thr_mma_score, tDvP, tDvdO, dkv_acc);

              const float dv_scale = dv_p_scale * do_scale;
#pragma unroll
              for (int32_t idx = 0; idx < cute::size(dkv_acc); ++idx)
              {
                const int32_t dim_block = dim_base / Traits::kBlockK;
                dv_accum_frag(dim_block, idx) += static_cast<float>(dkv_acc(idx)) * dv_scale;
              }
            }
          }
        }
        if (!n_valid)
        {
          continue;
        }

#pragma unroll
        for (int32_t dim_base = 0; dim_base < HeadDim; dim_base += Traits::kBlockK)
        {
          if (dq_pair_owner)
          {
            constexpr auto k_pair_shape = cute::make_shape(cute::Int<2 * Traits::kBlockN>{}, cute::Int<Traits::kBlockK>{});
            const auto sKPair = cute::local_tile(sK, k_pair_shape, cute::make_coord(n_tile / 2, dim_base / Traits::kBlockK));
            const auto sKdQ = qattn_cutlass::make_int8_transposed_b_view(sKPair);

            auto dq_acc = cute::partition_fragment_C(score_mma, BlockMNShape{});
            cute::clear(dq_acc);
            auto tDqdS = thr_mma_score.partition_fragment_A(sdSk);
            auto tDqK = thr_mma_score.partition_fragment_B(sKdQ);
            cute::copy(tiled_copy_score_a, thr_copy_score_a.partition_S(sdSk), thr_copy_score_a.retile_D(tDqdS));
            cute::copy(tiled_copy_transposed_b, thr_copy_transposed_b.partition_S(sKdQ), thr_copy_transposed_b.retile_D(tDqK));
            cute::gemm(thr_mma_score, tDqdS, tDqK, dq_acc);
            accumulate_dq_fragment_warp_shuffle<Traits>(dq_acc,
                                                          dsk_scale,
                                                          gDQAccum,
                                                          m_base,
                                                          dim_base,
                                                          params.seq_len,
                                                          lane_id);
          }

          if (dk_ready)
          {
            constexpr auto q_pair_shape = cute::make_shape(cute::Int<2 * Traits::kBlockM>{}, cute::Int<Traits::kBlockK>{});
            const auto sQPair = cute::local_tile(sQ, q_pair_shape, cute::make_coord(cute::_0{}, dim_base / Traits::kBlockK));
            const auto sQdK = qattn_cutlass::make_int8_transposed_b_view(sQPair);

            auto dk_acc = cute::partition_fragment_C(score_mma, BlockMNShape{});
            cute::clear(dk_acc);
            auto tDkdS = thr_mma_score.partition_fragment_A(sdSqPair);
            auto tDkQ = thr_mma_score.partition_fragment_B(sQdK);
            cute::copy(tiled_copy_score_a, thr_copy_score_a.partition_S(sdSqPair), thr_copy_score_a.retile_D(tDkdS));
            cute::copy(tiled_copy_transposed_b, thr_copy_transposed_b.partition_S(sQdK), thr_copy_transposed_b.retile_D(tDkQ));
            cute::gemm(thr_mma_score, tDkdS, tDkQ, dk_acc);
#pragma unroll
            for (int32_t idx = 0; idx < cute::size(dk_acc); ++idx)
            {
              const int32_t dim_block = dim_base / Traits::kBlockK;
              dk_accum_frag(dim_block, idx) += static_cast<float>(dk_acc(idx)) * dk_dsq_scale;
            }
          }
        }
      }
    }

    if (n_valid)
    {
      auto *const dkv_stage = reinterpret_cast<half *>(shared.warp_scratch.dsk_dkv.begin() + n_smem_slot * warp_scratch_offset);
#pragma unroll
      for (int32_t dim_block = 0; dim_block < kDimBlocks; ++dim_block)
      {
        store_dkv_fragment_coalesced<Traits>(dk_accum_frag, dim_block, gDK, dkv_stage, n_base, params.seq_len, lane_id);
        store_dkv_fragment_coalesced<Traits>(dv_accum_frag, dim_block, gDV, dkv_stage, n_base, params.seq_len, lane_id);
      }
    }
  }
}

template <int32_t HeadDim>
__global__ void convert_dq_kernel(const float *__restrict__ const DQAccum,
                                  half *__restrict__ const DQ,
                                  const Params params)
{
  using Traits = BwdTileTraits<HeadDim>;
  const int32_t q_block = blockIdx.x;
  const int32_t head = blockIdx.y;
  const int32_t batch = blockIdx.z;
  const int32_t base_row = q_block * Traits::kBlockM;
  const int32_t warp_id = threadIdx.x / warpSize;
  const int32_t lane_id = threadIdx.x % warpSize;
  const auto gDQAccum = make_workspace_matrix_view<HeadDim>(DQAccum, params, batch, head);
  const auto gDQ = make_head_matrix_view<HeadDim>(DQ, params, batch, head, params.stride_bz_dq, params.stride_seq_dq, params.stride_h_dq);
  const auto gDQAccumVec = cute::recast<cute::uint128_t>(gDQAccum);
  auto gDQVec = cute::recast<uint64_t>(gDQ);

  constexpr auto tile_shape = cute::make_shape(cute::_4{}, cute::Int<Traits::kFloatLoadVecCols>{});
  const auto tile_coord = cute::make_coord(base_row / 4 + warp_id, cute::_0{});
  const auto gDQAccumTile = cute::local_tile(gDQAccumVec, tile_shape, tile_coord);
  auto gDQTile = cute::local_tile(gDQVec, tile_shape, tile_coord);
  const auto coord_tile = cute::local_tile(cute::make_identity_tensor(cute::shape(gDQAccumVec)), tile_shape, tile_coord);

  typename Traits::GmemTiledCopyDQAccum tiled_copy_dq_accum;
  const auto thr_copy_dq_accum = tiled_copy_dq_accum.get_thread_slice(lane_id);
  const auto tDgDQAccum = thr_copy_dq_accum.partition_S(gDQAccumTile);
  const auto tDCoord = thr_copy_dq_accum.partition_S(coord_tile);
  auto rDQAccumVec = cute::make_fragment_like(tDgDQAccum);
  cute::clear(rDQAccumVec);
  const auto tDPred = cute::lazy::transform(tDCoord, [&](const auto &coord) {
    return cute::get<0>(coord) < params.seq_len;
  });
  cute::copy_if(tiled_copy_dq_accum, tDPred, tDgDQAccum, rDQAccumVec);

  const auto rDQAccum = cute::recast<float>(rDQAccumVec);
  auto rDQ = cute::make_fragment_like<half>(rDQAccum);
#pragma unroll
  for (int32_t idx = 0; idx < cute::size(rDQ); ++idx)
  {
    rDQ(idx) = __float2half_rn(rDQAccum(idx));
  }
  const auto rDQVec = cute::recast<uint64_t>(rDQ);
  typename Traits::GmemTiledCopyDQ tiled_copy_dq;
  const auto thr_copy_dq = tiled_copy_dq.get_thread_slice(lane_id);
  auto tDgDQ = thr_copy_dq.partition_D(gDQTile);
  cute::copy_if(tiled_copy_dq, tDPred, rDQVec, tDgDQ);
}

} // namespace sageattention::qattn_cutlass_bwd

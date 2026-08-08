#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cute/algorithm/clear.hpp>
#include <cute/algorithm/fill.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/algorithm/tensor_algorithms.hpp>
#include <cute/algorithm/tensor_reduce.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/numeric/math.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>

#include <cstdint>
#include <type_traits>

namespace sageattention::qattn_cutlass {

constexpr float LOG2E = 1.44269504088896340736f;

constexpr int32_t kWarpSize = 32;
constexpr uint32_t kFullWarpMask = 0xffffffffu;

// Shared tensors

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
    static_assert(Extent % 8 == 0, "Smem stride-one extent must be 4 for 64B swizzle or a multiple of 8 for 128B swizzle");
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
CUTE_HOST_DEVICE constexpr auto make_smem_matrix_layout()
{
  constexpr auto row_major_layout = cute::make_layout(cute::make_shape(cute::Int<Rows>{}, cute::Int<Cols>{}), cute::LayoutRight{});
  return make_smem_layout<Cols>(row_major_layout);
}

template <int32_t PackedCols, int32_t Rows>
CUTE_HOST_DEVICE constexpr auto make_v_smem_mma_layout()
{
  constexpr auto packed_layout = cute::make_layout(cute::make_shape(cute::Int<PackedCols>{}, cute::Int<Rows>{}), cute::LayoutLeft{});
  return make_smem_layout<PackedCols>(packed_layout);
}

template <typename Layout, typename Storage>
__device__ __forceinline__ auto make_smem_matrix(Storage& storage, const Layout layout)
{
  return cute::make_tensor(cute::make_smem_ptr(storage.begin()), layout);
}

template <typename QLayout, typename KLayout, typename VLayout, typename OLayout>
struct SharedStorage
{
  struct QKVStorage
  {
    cute::ArrayEngine<cute::uint128_t, cute::cosize_v<QLayout>> q;
    cute::ArrayEngine<cute::uint128_t, cute::cosize_v<KLayout>> k;
    cute::ArrayEngine<cute::uint128_t, cute::cosize_v<VLayout>> v;
  };

  union
  {
    QKVStorage qkv;
    cute::ArrayEngine<cute::uint128_t, cute::cosize_v<OLayout>> o;
  };
};

// Global tensors

template <int32_t HeadDim, typename T>
__device__ __forceinline__ auto make_logical_gmem_view(T *const base_ptr,
                                                       const int32_t rows,
                                                       const int32_t heads,
                                                       const int32_t batches,
                                                       const int32_t stride_seq,
                                                       const int32_t stride_h,
                                                       const int32_t stride_bz)
{
  using Element = std::remove_cv_t<T>;
  static_assert(std::is_same<Element, half>::value || std::is_same<Element, int8_t>::value);

  const auto gmem_layout = cute::make_layout(cute::make_shape(rows, cute::Int<HeadDim>{}, heads, batches),
                                             cute::make_stride(stride_seq, cute::_1{}, stride_h, stride_bz));
  return cute::make_tensor(cute::make_gmem_ptr(base_ptr), gmem_layout);
}

template <typename T>
__device__ __forceinline__ auto make_lse_view(T *const base_ptr,
                                              const int32_t batches,
                                              const int32_t heads,
                                              const int32_t rows)
{
  const auto lse_layout = cute::make_layout(cute::make_shape(batches, heads, rows), cute::LayoutRight{});
  return cute::make_tensor(cute::make_gmem_ptr(base_ptr), lse_layout);
}

template <typename Tensor>
__device__ __forceinline__ auto select_head_batch(Tensor tensor, const int32_t head_id, const int32_t batch_id)
{
  return tensor(cute::_, cute::_, head_id, batch_id);
}

template <int32_t Rows, typename Tensor>
__device__ __forceinline__ auto partition_rows(Tensor tensor, const int32_t tile_idx)
{
  return cute::local_tile(tensor,
                          cute::make_shape(cute::Int<Rows>{}, cute::size<1>(tensor)),
                          cute::make_coord(tile_idx, cute::_0{}));
}

// Thread grid

template <int32_t NumWarpsQ, int32_t NumWarpsK>
struct WarpCoord
{
  const int32_t q;
  const int32_t k;
};

template <int32_t NumWarpsQ, int32_t NumWarpsK>
__device__ __forceinline__ WarpCoord<NumWarpsQ, NumWarpsK> make_warp_coord(const int32_t warp_id)
{
  constexpr auto warp_layout = cute::make_layout(cute::make_shape(cute::Int<NumWarpsQ>{}, cute::Int<NumWarpsK>{}), cute::LayoutRight{});
  const auto warp_coord = warp_layout.get_flat_coord(warp_id);
  return {cute::get<0>(warp_coord), cute::get<1>(warp_coord)};
}

struct LaneCoord
{
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

  return {cute::get<0>(quad_coord), quad_lane, cute::get<0>(quad_pair_coord), cute::get<1>(quad_pair_coord)};
}

template <int32_t NumWarpsQ, int32_t NumWarpsK>
struct ThreadCoord
{
  const int32_t lane_id;
  const int32_t warp_id;
  const LaneCoord lane;
  const WarpCoord<NumWarpsQ, NumWarpsK> warp;
};

template <int32_t NumWarpsQ, int32_t NumWarpsK>
__device__ __forceinline__ ThreadCoord<NumWarpsQ, NumWarpsK> make_thread_coord()
{
  const int32_t lane_id = threadIdx.x;
  const int32_t warp_id = threadIdx.y;
  return {lane_id, warp_id, make_lane_coord(lane_id), make_warp_coord<NumWarpsQ, NumWarpsK>(warp_id)};
}

// Global-shared copy

template <int32_t Rows, int32_t Cols, typename CopyAtom>
CUTE_HOST_DEVICE constexpr auto make_2d_tiled_copy(CopyAtom copy_atom)
{
  static_assert(Cols == 4 || Cols % 8 == 0, "Copy tile columns must be 4 or a multiple of 8");
  constexpr int32_t line_lanes = (Cols == 4) ? 4 : 8;
  constexpr int32_t copy_lines_per_warp_per_iter = kWarpSize / line_lanes;
  static_assert(Rows % copy_lines_per_warp_per_iter == 0, "Copy tile rows must divide the warp row schedule");

  constexpr int32_t smem_iters_row = Cols / line_lanes;
  constexpr int32_t smem_iters_col = Rows / copy_lines_per_warp_per_iter;
  return cute::make_tiled_copy(
    copy_atom,
    cute::make_layout(cute::make_shape(cute::Int<copy_lines_per_warp_per_iter>{}, cute::Int<line_lanes>{}), cute::LayoutRight{}),
    cute::make_layout(cute::make_shape(cute::Int<smem_iters_col>{}, cute::Int<smem_iters_row>{}), cute::LayoutRight{}));
}

template <typename Gmem, typename Smem, typename Tiler, typename Thread>
__device__ __forceinline__ void load_global_to_shared(Gmem gmem,
                                                      Smem smem,
                                                      const Tiler cta_tiler,
                                                      const int32_t block_idx,
                                                      const Thread thread)
{
  constexpr int32_t rows = decltype(cute::size<0>(smem))::value;
  constexpr int32_t cols = decltype(cute::size<1>(smem))::value;
  const auto cta_coord = cute::make_coord(block_idx, cute::_0{});
  const auto gmem_cta = cute::local_tile(gmem, cta_tiler, cta_coord);
  const auto gmem_tile = partition_rows<rows>(gmem_cta, thread.warp_id);

  const auto tiled_copy = make_2d_tiled_copy<rows, cols>(cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cute::uint128_t>{});
  const auto thread_copy = tiled_copy.get_thread_slice(thread.lane_id);
  const auto thread_gmem = thread_copy.partition_S(gmem_tile);
  auto thread_smem = thread_copy.partition_D(smem);
  cute::copy(tiled_copy, thread_gmem, thread_smem);
  cute::cp_async_fence();
}

template <typename Gmem, typename Smem, typename Tiler, typename Thread>
__device__ __forceinline__ void load_global_to_shared_predicated(Gmem gmem,
                                                                 Smem smem,
                                                                 const Tiler cta_tiler,
                                                                 const int32_t block_idx,
                                                                 const Thread thread,
                                                                 const int32_t max_len)
{
  constexpr int32_t rows = decltype(cute::size<0>(smem))::value;
  constexpr int32_t cols = decltype(cute::size<1>(smem))::value;
  const auto cta_coord = cute::make_coord(block_idx, cute::_0{});
  const auto gmem_cta = cute::local_tile(gmem, cta_tiler, cta_coord);
  const auto gmem_tile = partition_rows<rows>(gmem_cta, thread.warp_id);
  const auto coord = cute::make_identity_tensor(cute::shape(gmem));
  const auto coord_cta = cute::local_tile(coord, cta_tiler, cta_coord);
  const auto coord_tile = partition_rows<rows>(coord_cta, thread.warp_id);

  const auto tiled_copy = make_2d_tiled_copy<rows, cols>(cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, cute::uint128_t>{});
  const auto thread_copy = tiled_copy.get_thread_slice(thread.lane_id);
  const auto thread_gmem = thread_copy.partition_S(gmem_tile);
  const auto thread_coord = thread_copy.partition_S(coord_tile);
  auto thread_smem = thread_copy.partition_D(smem);
  const auto thread_pred = cute::lazy::transform(thread_coord, [max_len](const auto &coord) {
    const int32_t row = cute::get<0>(coord);
    return row < max_len;
  });
  cute::copy_if(tiled_copy, thread_pred, thread_gmem, thread_smem);
  cute::cp_async_fence();
}

template <int32_t StoreRows, typename Smem, typename Gmem, typename Tiler, typename Thread>
__device__ __forceinline__ void store_shared_to_global(Smem smem,
                                                       Gmem gmem,
                                                       const Tiler cta_tiler,
                                                       const int32_t block_idx,
                                                       const Thread thread,
                                                       const int32_t max_len)
{
  constexpr int32_t rows = StoreRows;
  constexpr int32_t cols = decltype(cute::size<1>(smem))::value;
  const auto cta_coord = cute::make_coord(block_idx, cute::_0{});
  const auto gmem_cta = cute::local_tile(gmem, cta_tiler, cta_coord);
  const auto gmem_tile = partition_rows<rows>(gmem_cta, thread.warp.q);
  const auto smem_tile = partition_rows<rows>(smem, thread.warp.q);
  const auto coord = cute::make_identity_tensor(cute::shape(gmem));
  const auto coord_cta = cute::local_tile(coord, cta_tiler, cta_coord);
  const auto coord_tile = partition_rows<rows>(coord_cta, thread.warp.q);

  const auto tiled_copy = make_2d_tiled_copy<rows, cols>(cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, cute::uint128_t>{});
  const auto thread_copy = tiled_copy.get_thread_slice(thread.lane_id);
  const auto thread_smem = thread_copy.partition_S(smem_tile);
  const auto thread_coord = thread_copy.partition_S(coord_tile);
  auto thread_gmem = thread_copy.partition_D(gmem_tile);
  const auto thread_pred = cute::lazy::transform(thread_coord, [max_len](const auto &coord) {
    const int32_t row = cute::get<0>(coord);
    return row < max_len;
  });
  cute::copy_if(tiled_copy, thread_pred, thread_smem, thread_gmem);
}

// MMA traits and fragment layouts

using QKTiledMMA = cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x32_S32S8S8S32_TN>,
                                  cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>,
                                  cute::Tile<cute::_16, cute::_16, cute::_32>>;
using PVTiledMMA = cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x16_F32F16F16F32_TN>,
                                  cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>,
                                  cute::Tile<cute::_16, cute::_16, cute::_16>>;

template<int32_t CTA_Q, int32_t CTA_K, int32_t WARP_Q, int32_t WARP_K, int32_t HeadDim>
struct AttentionTileTraits
{
  static_assert(HeadDim == 64 || HeadDim == 128, "CUTLASS qattn fwd currently supports head_dim 64 or 128");
  static_assert(WARP_K == CTA_K, "CUTLASS qattn fwd currently requires one K warp partition");

  static_assert(CTA_Q % WARP_Q == 0);
  static_assert(CTA_K % WARP_K == 0);
  static_assert(HeadDim % 64 == 0);

  static constexpr int32_t num_warps_q = CTA_Q / WARP_Q;
  static constexpr int32_t num_warps_k = CTA_K / WARP_K;
  static constexpr int32_t num_warps = num_warps_q * num_warps_k;

  static constexpr int32_t qk_mma_m = decltype(cute::tile_size<0>(QKTiledMMA{}))::value;
  static constexpr int32_t qk_mma_n = decltype(cute::tile_size<1>(QKTiledMMA{}))::value;
  static constexpr int32_t qk_mma_k = decltype(cute::tile_size<2>(QKTiledMMA{}))::value;
  static constexpr int32_t pv_mma_n = decltype(cute::tile_size<1>(PVTiledMMA{}))::value;

  static constexpr int32_t num_tiles_q = WARP_Q / qk_mma_m;
  static constexpr int32_t num_tiles_k = WARP_K / qk_mma_n;
  static constexpr int32_t num_tiles_qk_inner = HeadDim / qk_mma_k;
  static constexpr int32_t num_tiles_v = HeadDim / pv_mma_n;

  static constexpr int32_t qk_load_cols = HeadDim / packed_elements<int8_t>();
  static constexpr int32_t v_load_cols = HeadDim / packed_elements<half>();
  static constexpr int32_t o_store_cols = HeadDim / packed_elements<half>();

  static constexpr int32_t q_load_rows = CTA_Q / num_warps;
  static constexpr int32_t k_load_rows = CTA_K / num_warps;
  static constexpr int32_t v_load_rows = CTA_K / num_warps;
  static constexpr int32_t o_store_rows = CTA_Q / num_warps;

  CUTE_HOST_DEVICE static constexpr auto q_smem_layout() { return make_smem_matrix_layout<CTA_Q, qk_load_cols>(); }
  CUTE_HOST_DEVICE static constexpr auto k_smem_layout() { return make_smem_matrix_layout<CTA_K, qk_load_cols>(); }
  CUTE_HOST_DEVICE static constexpr auto v_smem_layout() { return make_v_smem_mma_layout<v_load_cols, CTA_K>(); }
  CUTE_HOST_DEVICE static constexpr auto v_smem_load_layout() { return make_smem_matrix_layout<CTA_K, v_load_cols>(); }
  CUTE_HOST_DEVICE static constexpr auto o_smem_layout() { return make_smem_matrix_layout<CTA_Q, o_store_cols>(); }

  CUTE_HOST_DEVICE static constexpr auto q_cta_tiler() { return cute::make_shape(cute::Int<CTA_Q>{}, cute::Int<qk_load_cols>{}); }
  CUTE_HOST_DEVICE static constexpr auto k_cta_tiler() { return cute::make_shape(cute::Int<CTA_K>{}, cute::Int<qk_load_cols>{}); }
  CUTE_HOST_DEVICE static constexpr auto v_cta_tiler() { return cute::make_shape(cute::Int<CTA_K>{}, cute::Int<v_load_cols>{}); }
  CUTE_HOST_DEVICE static constexpr auto o_cta_tiler() { return cute::make_shape(cute::Int<CTA_Q>{}, cute::Int<o_store_cols>{}); }
};

template <int32_t NumTilesM, int32_t NumTilesN>
__device__ __forceinline__ constexpr auto make_mma_accumulator_layout()
{
  return cute::make_layout(cute::make_shape(cute::make_shape(cute::_2{}, cute::_2{}),
                                            cute::make_shape(cute::Int<NumTilesM>{}, cute::_1{}),
                                            cute::make_shape(cute::Int<NumTilesN>{}, cute::_2{})),
                           cute::make_stride(cute::make_stride(cute::_1{}, cute::_2{}),
                                             cute::make_stride(cute::Int<NumTilesN * 8>{}, cute::_0{}),
                                             cute::make_stride(cute::_8{}, cute::_4{})));
}

template <typename TiledMMA, int32_t NumTilesM, int32_t NumTilesN, typename ThreadIdx>
__device__ __forceinline__ auto make_mma_c_coord(const ThreadIdx thread_idx)
{
  constexpr int32_t mma_m = decltype(cute::tile_size<0>(TiledMMA{}))::value;
  constexpr int32_t mma_n = decltype(cute::tile_size<1>(TiledMMA{}))::value;
  TiledMMA tiled_mma;
  const auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
  const auto identity = cute::make_identity_tensor(cute::make_shape(cute::Int<NumTilesM * mma_m>{}, cute::Int<NumTilesN * mma_n>{}));
  return thr_mma.partition_C(identity);
}

template <typename Fragment>
__device__ __forceinline__ auto select_mma_c_tile(Fragment &&fragment, const int32_t m_tile, const int32_t n_tile)
{
  return fragment(cute::_, cute::make_coord(m_tile, cute::_), cute::make_coord(n_tile, cute::_));
}

template <int32_t NumTilesQ, int32_t NumTilesK>
__device__ __forceinline__ constexpr auto make_prob_layout()
{
  const auto pv_a_layout = cute::make_layout(cute::partition_shape_A(PVTiledMMA{}, cute::select<0, 2>(cute::tile_shape(PVTiledMMA{}))));
  return cute::make_layout(cute::tuple_cat(cute::make_shape(cute::Int<NumTilesQ>{}, cute::Int<NumTilesK>{}), cute::shape(pv_a_layout)),
                           cute::tuple_cat(cute::make_stride(cute::Int<NumTilesK * 8>{}, cute::_8{}), cute::stride(pv_a_layout)));
}

// Epilogue copy

template <typename Src, typename Dst>
__device__ __forceinline__ void convert_f32_to_f16(const Src &src, Dst &dst)
{
  using Converter = cutlass::NumericArrayConverter<cutlass::half_t, float, 8>;
  const auto src_vectors = cute::recast<typename Converter::source_type>(src);
  auto dst_vectors = cute::recast<typename Converter::result_type>(dst);
  cute::copy(cute::lazy::transform(src_vectors, Converter{}), dst_vectors);
}

template <typename TiledMMA, int32_t WarpQ, typename Out, typename Smem, typename Thread>
__device__ __forceinline__ void store_register_to_shared(const Out &out, Smem smem, const Thread thread)
{
  constexpr int32_t num_tiles_q = decltype(cute::size<1, 0>(out))::value;
  constexpr int32_t num_tiles_v = decltype(cute::size<2, 0>(out))::value;
  constexpr int32_t rows = num_tiles_q * decltype(cute::tile_size<0>(TiledMMA{}))::value;
  constexpr int32_t cols = num_tiles_v * decltype(cute::tile_size<1>(TiledMMA{}))::value;

  const auto accumulator_layout = make_mma_accumulator_layout<num_tiles_q, num_tiles_v>();
  auto out_half = cute::make_tensor<half>(accumulator_layout);
  convert_f32_to_f16(out, out_half);

  TiledMMA tiled_mma;
  const auto tiled_copy = cute::make_tiled_copy_C(cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, half>{}, tiled_mma);
  const auto thread_copy = tiled_copy.get_thread_slice(thread.lane_id);

  const auto src_shape = cute::shape(cute::partition_shape_C(tiled_mma, cute::make_shape(cute::Int<rows>{}, cute::Int<cols>{})));
  const auto src_stride = cute::make_stride(cute::stride<0>(accumulator_layout), cute::stride<1, 0>(accumulator_layout), cute::stride<2, 1>(accumulator_layout));
  const auto thread_out = thread_copy.retile_S(cute::make_tensor(out_half.data(), cute::make_layout(src_shape, src_stride)));

  auto smem_mma = partition_rows<WarpQ>(smem, thread.warp.q);
  auto smem_tile = cute::recast<half>(smem_mma);
  auto thread_smem = thread_copy.partition_D(smem_tile);
  cute::copy(tiled_copy, thread_out, thread_smem);
}

// Softmax

template <int32_t CTA_Q, int32_t WARP_Q, typename Thread>
__device__ __forceinline__ float load_q_scale(const float *__restrict__ const scales,
                                              const int32_t batch_id,
                                              const int32_t head_id,
                                              const int32_t num_heads,
                                              const int32_t qo_len,
                                              const int32_t cta_block,
                                              const Thread thread)
{
  constexpr int32_t quant_block_q = 32;
  static_assert(CTA_Q % quant_block_q == 0 && quant_block_q % WARP_Q == 0, "Q32 scales must align with CTA and warp row ownership");
  const int32_t num_scale_blocks = cute::ceil_div(qo_len, quant_block_q);
  const int32_t row = cta_block * CTA_Q + thread.warp.q * WARP_Q;
  const int32_t scale_block = min(row / quant_block_q, num_scale_blocks - 1);
  return scales[(batch_id * num_heads + head_id) * num_scale_blocks + scale_block];
}

template <int32_t CTA_K, int32_t WARP_K>
__device__ __forceinline__ float load_k_scale(const float *__restrict__ const scales,
                                              const int32_t batch_id,
                                              const int32_t head_id,
                                              const int32_t num_heads,
                                              const int32_t kv_len,
                                              const int32_t cta_block)
{
  constexpr int32_t quant_block_k = 64;
  static_assert(quant_block_k % CTA_K == 0 && WARP_K == CTA_K, "K64 scales must align with CTA and warp column ownership");
  const int32_t num_scale_blocks = cute::ceil_div(kv_len, quant_block_k);
  return scales[(batch_id * num_heads + head_id) * num_scale_blocks + cta_block * CTA_K / quant_block_k];
}

template <int32_t CtaRows, int32_t WarpRows, typename Score, typename Thread>
__device__ __forceinline__ void apply_out_of_bound_mask(Score &score,
                                                        const int32_t kv_block_idx,
                                                        const Thread thread,
                                                        const int32_t kv_len)
{
  constexpr int32_t num_tiles_q = decltype(cute::size<1, 0>(score))::value;
  constexpr int32_t num_tiles_k = decltype(cute::size<2, 0>(score))::value;
  CUTE_STATIC_ASSERT_V(cute::shape(score) == cute::shape(make_mma_accumulator_layout<num_tiles_q, num_tiles_k>()));

  const auto score_coord = make_mma_c_coord<QKTiledMMA, num_tiles_q, num_tiles_k>(thread.lane_id);
  const int32_t row_base = kv_block_idx * CtaRows + thread.warp.k * WarpRows;

  cute::transform(score_coord, score, score, [&](const auto &coord, const float value) {
    const int32_t local_row = cute::get<1>(coord);
    return row_base + local_row >= kv_len ? -5000000.0f : value;
  });
}

template <typename Score, typename Out, typename M, typename D>
__device__ __forceinline__ void update_mdo(Score &score,
                                           Out &out,
                                           M &m,
                                           D &d,
                                           const float sm_scale)
{
  const auto m_coord = cute::make_identity_tensor(cute::shape(m));
  cute::for_each(m_coord, [&](const auto &coord) {
    const int32_t fq = cute::get<0>(coord);
    const int32_t k = cute::get<1>(coord);
    auto score_fq_k = score(cute::make_coord(cute::_, k), cute::make_coord(fq, cute::_0{}), cute::_);
    auto out_fq_k = out(cute::make_coord(cute::_, k), cute::make_coord(fq, cute::_0{}), cute::_);

    const float m_prev = m(coord);
    float m_temp = cute::reduce(score_fq_k, -5000000.0f, cute::max_fn{});
    m_temp *= sm_scale;
    m_temp = max(m_temp, __shfl_xor_sync(kFullWarpMask, m_temp, 0x1));
    m_temp = max(m_temp, __shfl_xor_sync(kFullWarpMask, m_temp, 0x2));
    m(coord) = max(m(coord), m_temp);

    const float o_scale = exp2f(m_prev - m(coord));
    d(coord) *= o_scale;

    cute::transform(out_fq_k, [o_scale](const float value) { return value * o_scale; });

    const float negative_m = -m(coord);
    cute::transform(score_fq_k, [sm_scale, negative_m](const float value) {
      return exp2f(fmaf(value, sm_scale, negative_m));
    });
  });
}

template <typename Score, typename D>
__device__ __forceinline__ void accumulate_d(const Score &score, D &d)
{
  const auto d_coord = cute::make_identity_tensor(cute::shape(d));
  cute::for_each(d_coord, [&](const auto &coord) {
    const int32_t fq = cute::get<0>(coord);
    const int32_t k = cute::get<1>(coord);
    const auto score_fq_k = score(cute::make_coord(cute::_, k), cute::make_coord(fq, cute::_0{}), cute::_);
    d(coord) = cute::reduce(score_fq_k, d(coord));
  });
}

template <typename Out, typename D>
__device__ __forceinline__ void normalize_d(Out &out, D &d)
{
  constexpr int32_t num_tiles_q = decltype(cute::size<0>(d))::value;
  constexpr int32_t m_fragments = decltype(cute::size<1>(d))::value;

  cute::transform(d, [](float value) {
    value += __shfl_xor_sync(kFullWarpMask, value, 0x1);
    value += __shfl_xor_sync(kFullWarpMask, value, 0x2);
    return value;
  });

  auto d_rcp = cute::make_tensor<float>(cute::make_shape(cute::Int<num_tiles_q>{}, cute::Int<m_fragments>{}), cute::LayoutRight{});
  cute::transform(d, d_rcp, [](const float value) { return 1.0f / value; });

  const auto out_coord = cute::make_identity_tensor(cute::shape(out));
  cute::for_each(out_coord, [&](const auto &coord) {
    const int32_t fq = cute::get<1, 0>(coord);
    const int32_t k = cute::get<0, 1>(coord);
    out(coord) *= d_rcp(fq, k);
  });
}

// QK and SV compute

template <int32_t NumTilesQKInner, typename SmemQ, typename SmemK, typename Score, typename Thread>
__device__ __forceinline__ void compute_int_qk(SmemQ smem_Q,
                                               SmemK smem_K,
                                               Score &score,
                                               const Thread thread)
{
  constexpr int32_t num_tiles_q = decltype(cute::size<1, 0>(score))::value;
  constexpr int32_t num_tiles_k = decltype(cute::size<2, 0>(score))::value;
  CUTE_STATIC_ASSERT_V(cute::shape(score) == cute::shape(make_mma_accumulator_layout<num_tiles_q, num_tiles_k>()));

  QKTiledMMA tiled_mma;
  const auto smem_copy_q = cute::make_tiled_copy_A(cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, int8_t>{}, tiled_mma);
  const auto smem_copy_k = cute::make_tiled_copy_B(cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, int8_t>{}, tiled_mma);
  const auto thr_mma = tiled_mma.get_thread_slice(thread.lane_id);
  const auto thread_copy_q = smem_copy_q.get_thread_slice(thread.lane_id);
  const auto thread_copy_k = smem_copy_k.get_thread_slice(thread.lane_id);

  const auto q_tiles = cute::local_tile(smem_Q,
                                        cute::make_shape(cute::size<0>(smem_Q), cute::tile_size<2>(tiled_mma)),
                                        cute::make_coord(cute::_0{}, cute::_0{}));
  const auto k_tile0 = cute::local_tile(smem_K,
                                        cute::select<1, 2>(cute::tile_shape(tiled_mma)),
                                        cute::make_coord(cute::_0{}, cute::_0{}));
  auto rq = thr_mma.partition_fragment_A(q_tiles);
  auto rk = thr_mma.partition_fragment_B(k_tile0);
  const auto q_fragment_tiler = cute::make_shape(cute::shape<0>(rq), cute::_1{}, cute::_1{});

#pragma unroll
  for (int32_t iter = 0; iter < NumTilesQKInner; ++iter)
  {
#pragma unroll
    for (int32_t fq = 0; fq < num_tiles_q; ++fq)
    {
      const auto q_tile = cute::local_tile(smem_Q,
                                           cute::select<0, 2>(cute::tile_shape(tiled_mma)),
                                           cute::make_coord(fq, iter));
      auto rq_fragment = cute::local_tile(rq, q_fragment_tiler, cute::make_coord(cute::_0{}, fq, cute::_0{}));
      cute::copy(smem_copy_q, thread_copy_q.partition_S(q_tile), thread_copy_q.retile_D(rq_fragment));
    }

#pragma unroll
    for (int32_t fk = 0; fk < num_tiles_k; ++fk)
    {
      const auto k_tile = cute::local_tile(smem_K,
                                           cute::select<1, 2>(cute::tile_shape(tiled_mma)),
                                           cute::make_coord(fk, iter));
      cute::copy(smem_copy_k, thread_copy_k.partition_S(k_tile), thread_copy_k.retile_D(rk));

#pragma unroll
      for (int32_t fq = 0; fq < num_tiles_q; ++fq)
      {
        auto rq_fragment = cute::local_tile(rq, q_fragment_tiler, cute::make_coord(cute::_0{}, fq, cute::_0{}));
        auto c_fragment = select_mma_c_tile(score, fq, fk);
        if (iter == 0)
        {
          cute::clear(c_fragment);
        }
        cute::gemm(thr_mma, rq_fragment, rk, c_fragment);
      }
    }
  }
}

template <typename SmemV, typename Prob, typename Out, typename Thread>
__device__ __forceinline__ void compute_fp16_sv(SmemV smem_V,
                                                const Prob &prob,
                                                Out &out,
                                                const Thread thread)
{
  constexpr int32_t num_tiles_q = decltype(cute::size<0>(prob))::value;
  constexpr int32_t num_tiles_k = decltype(cute::size<1>(prob))::value;
  constexpr int32_t num_tiles_v = decltype(cute::size<2, 0>(out))::value;
  constexpr int32_t pv_mma_n = decltype(cute::tile_size<1>(PVTiledMMA{}))::value;
  constexpr int32_t pv_mma_k = decltype(cute::tile_size<2>(PVTiledMMA{}))::value;
  CUTE_STATIC_ASSERT_V(cute::size<0>(smem_V) == cute::Int<num_tiles_v * pv_mma_n>{});
  CUTE_STATIC_ASSERT_V(cute::size<1>(smem_V) == cute::Int<num_tiles_k * pv_mma_k>{});
  CUTE_STATIC_ASSERT_V((cute::select<2, 3, 4>(cute::shape(prob)) == cute::partition_shape_A(PVTiledMMA{}, cute::select<0, 2>(cute::tile_shape(PVTiledMMA{})))));
  CUTE_STATIC_ASSERT_V((cute::size<1, 0>(out) == cute::size<0>(prob)));
  CUTE_STATIC_ASSERT_V(cute::shape(out) == cute::shape(make_mma_accumulator_layout<num_tiles_q, num_tiles_v>()));

  PVTiledMMA tiled_mma;
  const auto smem_copy_v = cute::make_tiled_copy_B(cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, half>{}, tiled_mma);
  const auto thr_mma = tiled_mma.get_thread_slice(thread.lane_id);
  const auto thread_copy_v = smem_copy_v.get_thread_slice(thread.lane_id);

  const auto v_tile0 = cute::local_tile(smem_V,
                                        cute::select<1, 2>(cute::tile_shape(tiled_mma)),
                                        cute::make_coord(cute::_0{}, cute::_0{}));
  auto rv = thr_mma.partition_fragment_B(v_tile0);

#pragma unroll
  for (int32_t fk = 0; fk < num_tiles_k; ++fk)
  {
#pragma unroll
    for (int32_t fv = 0; fv < num_tiles_v; ++fv)
    {
      const auto v_tile = cute::local_tile(smem_V,
                                           cute::select<1, 2>(cute::tile_shape(tiled_mma)),
                                           cute::make_coord(fv, fk));
      cute::copy(smem_copy_v, thread_copy_v.partition_S(v_tile), thread_copy_v.retile_D(rv));

#pragma unroll
      for (int32_t fq = 0; fq < num_tiles_q; ++fq)
      {
        const auto prob_fragment = prob(fq, fk, cute::_, cute::_, cute::_);
        auto c_fragment = select_mma_c_tile(out, fq, fv);
        cute::gemm(thr_mma, prob_fragment, rv, c_fragment);
      }
    }
  }
}

template <typename Score>
__device__ __forceinline__ auto make_prob_fragment(const Score &score)
{
  constexpr int32_t num_tiles_q = decltype(cute::size<1, 0>(score))::value;
  constexpr int32_t num_tiles_k = decltype(cute::size<2, 0>(score))::value;
  return cute::make_tensor<half>(make_prob_layout<num_tiles_q, num_tiles_k>());
}

template <bool ScaleScore, bool MaskTail,
          int32_t CTA_K, int32_t WARP_K, int32_t NumTilesQKInner,
          typename SmemQ, typename SmemK,
          typename ScoreI32, typename ScoreMMA, typename Prob,
          typename Out, typename M, typename D, typename Thread>
__device__ __forceinline__ void compute_qk_softmax_prob(SmemQ smem_Q,
                                                        SmemK smem_K,
                                                        const ScoreI32 &score_i32,
                                                        ScoreMMA &score_mma,
                                                        Prob &prob,
                                                        Out &out,
                                                        M &m,
                                                        D &d,
                                                        const int32_t kv_block_idx,
                                                        const Thread thread,
                                                        const float softmax_scale,
                                                        const float score_scale,
                                                        const int32_t kv_len)
{
  compute_int_qk<NumTilesQKInner>(smem_Q, smem_K, score_mma, thread);

  auto score_f32 = cute::make_tensor_like<float>(score_i32);
  if constexpr (ScaleScore)
  {
    cute::copy(cute::lazy::transform(score_i32, [score_scale](const int32_t value) {
      return __int2float_rz(value) * score_scale;
    }), score_f32);
  }
  else
  {
    cute::copy(cute::lazy::transform(score_i32, [](const int32_t value) {
      return __int2float_rz(value);
    }), score_f32);
  }

  if constexpr (MaskTail)
  {
    apply_out_of_bound_mask<CTA_K, WARP_K>(score_f32, kv_block_idx, thread, kv_len);
  }

  update_mdo(score_f32, out, m, d, softmax_scale);
  accumulate_d(score_f32, d);

  const auto score_prob = cute::make_tensor(score_f32.data(), prob.layout());
  convert_f32_to_f16(score_prob, prob);
}

template<int32_t CTA_Q, int32_t CTA_K, int32_t WARP_Q, int32_t WARP_K, int32_t head_dim, bool ReturnLse>
__global__ void qk_int8_sv_f16_accum_f32_attn_kernel(const int8_t *__restrict__ const Q,
                                                     const int8_t *__restrict__ const K,
                                                     const half *__restrict__ const V,
                                                     half *__restrict__ const O,
                                                     float *__restrict__ const Lse,
                                                     const float *__restrict__ const Q_scale,
                                                     const float *__restrict__ const K_scale,
                                                     const int32_t qo_len,
                                                     const int32_t kv_len,
                                                     const int32_t num_kv_groups,
                                                     const int32_t stride_bz_q,
                                                     const int32_t stride_seq_q,
                                                     const int32_t stride_h_q,
                                                     const int32_t stride_bz_k,
                                                     const int32_t stride_seq_k,
                                                     const int32_t stride_h_k,
                                                     const int32_t stride_bz_v,
                                                     const int32_t stride_seq_v,
                                                     const int32_t stride_h_v,
                                                     const int32_t stride_bz_o,
                                                     const int32_t stride_seq_o,
                                                     const int32_t stride_h_o,
                                                     const float sm_scale)
{
  using Traits = AttentionTileTraits<CTA_Q, CTA_K, WARP_Q, WARP_K, head_dim>;

  const float original_sm_scale = sm_scale * LOG2E;

  const int32_t num_batches = gridDim.z;
  const int32_t batch_id = blockIdx.z;
  const int32_t num_qo_heads = gridDim.y;
  const int32_t num_kv_heads = num_qo_heads / num_kv_groups;
  const int32_t qo_head_id = blockIdx.y;
  const int32_t kv_head_id = qo_head_id / num_kv_groups;
  const int32_t q_block_id = blockIdx.x;

  // Global tensors

  constexpr auto Q_cta_tiler = Traits::q_cta_tiler();
  constexpr auto K_cta_tiler = Traits::k_cta_tiler();
  constexpr auto V_cta_tiler = Traits::v_cta_tiler();
  constexpr auto O_cta_tiler = Traits::o_cta_tiler();

  const auto Q_logical_full = make_logical_gmem_view<head_dim>(Q, qo_len, num_qo_heads, num_batches, stride_seq_q, stride_h_q, stride_bz_q);
  const auto K_logical_full = make_logical_gmem_view<head_dim>(K, kv_len, num_kv_heads, num_batches, stride_seq_k, stride_h_k, stride_bz_k);
  const auto V_logical_full = make_logical_gmem_view<head_dim>(V, kv_len, num_kv_heads, num_batches, stride_seq_v, stride_h_v, stride_bz_v);
  const auto O_logical_full = make_logical_gmem_view<head_dim>(O, qo_len, num_qo_heads, num_batches, stride_seq_o, stride_h_o, stride_bz_o);

  const auto Q_logical = select_head_batch(Q_logical_full, qo_head_id, batch_id);
  const auto K_logical = select_head_batch(K_logical_full, kv_head_id, batch_id);
  const auto V_logical = select_head_batch(V_logical_full, kv_head_id, batch_id);
  const auto O_logical = select_head_batch(O_logical_full, qo_head_id, batch_id);

  const auto Q_gmem = cute::recast<cute::uint128_t>(Q_logical);
  const auto K_gmem = cute::recast<cute::uint128_t>(K_logical);
  const auto V_gmem = cute::recast<cute::uint128_t>(V_logical);
  const auto O_gmem = cute::recast<cute::uint128_t>(O_logical);

  // Shared tensors

  const auto thread = make_thread_coord<Traits::num_warps_q, Traits::num_warps_k>();

  constexpr auto Q_smem_layout = Traits::q_smem_layout();
  constexpr auto K_smem_layout = Traits::k_smem_layout();
  constexpr auto V_smem_layout = Traits::v_smem_layout();
  constexpr auto V_smem_load_layout = Traits::v_smem_load_layout();
  constexpr auto O_smem_layout = Traits::o_smem_layout();
  using SharedStorage = SharedStorage<decltype(Q_smem_layout), decltype(K_smem_layout), decltype(V_smem_layout), decltype(O_smem_layout)>;

  extern __shared__ char shared_memory[];
  SharedStorage& shared = *reinterpret_cast<SharedStorage*>(shared_memory);

  auto Q_smem = make_smem_matrix(shared.qkv.q, Q_smem_layout);
  auto K_smem = make_smem_matrix(shared.qkv.k, K_smem_layout);
  auto V_smem = make_smem_matrix(shared.qkv.v, V_smem_layout);
  auto V_smem_load_view = make_smem_matrix(shared.qkv.v, V_smem_load_layout);
  auto O_smem = make_smem_matrix(shared.o, O_smem_layout);

  auto Q_smem_load = partition_rows<Traits::q_load_rows>(Q_smem, thread.warp_id);
  auto K_smem_load = partition_rows<Traits::k_load_rows>(K_smem, thread.warp_id);
  auto V_smem_load = partition_rows<Traits::v_load_rows>(V_smem_load_view, thread.warp_id);

  auto Q_smem_mma = cute::recast<int8_t>(partition_rows<WARP_Q>(Q_smem, thread.warp.q));
  auto K_smem_mma = cute::recast<int8_t>(partition_rows<WARP_K>(K_smem, thread.warp.k));
  auto V_smem_mma = cute::recast<half>(V_smem);

  // Register tensors

  auto score_i32 = cute::make_tensor<int32_t>(make_mma_accumulator_layout<Traits::num_tiles_q, Traits::num_tiles_k>());
  auto score_mma = cute::recast<int32_t>(score_i32);
  auto out = cute::make_tensor<float>(make_mma_accumulator_layout<Traits::num_tiles_q, Traits::num_tiles_v>());
  auto m = cute::make_tensor<float>(cute::make_shape(cute::Int<Traits::num_tiles_q>{}, cute::_2{}), cute::LayoutRight{});
  auto d = cute::make_tensor<float>(cute::make_shape(cute::Int<Traits::num_tiles_q>{}, cute::_2{}), cute::LayoutRight{});

  cute::fill(out, 0.0f);
  cute::fill(m, -5000000.0f);
  cute::fill(d, 1.0f);

  // QK softmax SV pipeline

  load_global_to_shared_predicated(Q_gmem, Q_smem_load, Q_cta_tiler, q_block_id, thread, qo_len);
  cute::cp_async_wait<0>();

  const int32_t num_iterations = cute::ceil_div(kv_len, CTA_K);

  const float q_scale = load_q_scale<CTA_Q, WARP_Q>(Q_scale, batch_id, qo_head_id, num_qo_heads, qo_len, q_block_id, thread);
  float k_scale = load_k_scale<CTA_K, WARP_K>(K_scale, batch_id, kv_head_id, num_kv_heads, kv_len, 0);
  float dequant_scale = q_scale * k_scale;

  load_global_to_shared_predicated(K_gmem, K_smem_load, K_cta_tiler, cute::_0{}, thread, kv_len);
  load_global_to_shared_predicated(V_gmem, V_smem_load, V_cta_tiler, cute::_0{}, thread, kv_len);

#pragma unroll 1
  for (int32_t iter = 1; iter < num_iterations - 1; ++iter)
  {
    cute::cp_async_wait<1>();
    __syncthreads();

    auto prob = make_prob_fragment(score_i32);
    compute_qk_softmax_prob<false, false, CTA_K, WARP_K, Traits::num_tiles_qk_inner>(
      Q_smem_mma, K_smem_mma, score_i32, score_mma, prob, out, m, d, iter - 1, thread,
      original_sm_scale * dequant_scale, 0.0f, kv_len);
    __syncthreads();

    load_global_to_shared(K_gmem, K_smem_load, K_cta_tiler, iter, thread);
    k_scale = load_k_scale<CTA_K, WARP_K>(K_scale, batch_id, kv_head_id, num_kv_heads, kv_len, iter);
    dequant_scale = q_scale * k_scale;

    cute::cp_async_wait<1>();
    __syncthreads();
    compute_fp16_sv(V_smem_mma, prob, out, thread);
    __syncthreads();
    load_global_to_shared(V_gmem, V_smem_load, V_cta_tiler, iter, thread);
  }

  if (num_iterations > 1)
  {
    cute::cp_async_wait<1>();
    __syncthreads();

    auto prob = make_prob_fragment(score_i32);
    compute_qk_softmax_prob<true, false, CTA_K, WARP_K, Traits::num_tiles_qk_inner>(
      Q_smem_mma, K_smem_mma, score_i32, score_mma, prob, out, m, d, num_iterations - 2, thread,
      original_sm_scale, dequant_scale, kv_len);
    __syncthreads();

    load_global_to_shared_predicated(K_gmem, K_smem_load, K_cta_tiler, num_iterations - 1, thread, kv_len);
    k_scale = load_k_scale<CTA_K, WARP_K>(K_scale, batch_id, kv_head_id, num_kv_heads, kv_len, num_iterations - 1);
    dequant_scale = q_scale * k_scale;

    cute::cp_async_wait<1>();
    __syncthreads();
    compute_fp16_sv(V_smem_mma, prob, out, thread);
    __syncthreads();
    load_global_to_shared_predicated(V_gmem, V_smem_load, V_cta_tiler, num_iterations - 1, thread, kv_len);
  }

  {
    cute::cp_async_wait<1>();
    __syncthreads();

    auto prob = make_prob_fragment(score_i32);
    compute_qk_softmax_prob<true, true, CTA_K, WARP_K, Traits::num_tiles_qk_inner>(
      Q_smem_mma, K_smem_mma, score_i32, score_mma, prob, out, m, d, num_iterations - 1, thread,
      original_sm_scale, dequant_scale, kv_len);

    cute::cp_async_wait<0>();
    __syncthreads();
    compute_fp16_sv(V_smem_mma, prob, out, thread);
    __syncthreads();
  }

  // Normalize and epilogue

  normalize_d(out, d);

  store_register_to_shared<PVTiledMMA, WARP_Q>(out, O_smem, thread);
  __syncwarp();
  store_shared_to_global<Traits::o_store_rows>(O_smem, O_gmem, O_cta_tiler, q_block_id, thread, qo_len);

  if constexpr (ReturnLse)
  {
    if (thread.lane.quad_lane < 2 * Traits::num_tiles_q)
    {
      const auto out_coord = make_mma_c_coord<PVTiledMMA, Traits::num_tiles_q, Traits::num_tiles_v>(thread.lane_id);
      const int32_t fq = thread.lane.quad_pair;
      const int32_t k = thread.lane.quad_pair_lane;

      const auto coord = out_coord(cute::make_coord(cute::_0{}, k), fq, cute::_0{});
      const int32_t local_row = cute::get<0>(coord);
      const int32_t lse_idx = q_block_id * CTA_Q + thread.warp.q * WARP_Q + local_row;
      if (lse_idx < qo_len)
      {
        auto lse = make_lse_view(Lse, num_batches, num_qo_heads, qo_len);
        lse(batch_id, qo_head_id, lse_idx) = log2f(d(fq, k)) + m(fq, k);
      }
    }
  }
}

} // namespace sageattention::qattn_cutlass

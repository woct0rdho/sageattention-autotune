#pragma once

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/layout.hpp>

#include <cstdint>

namespace sageattention::qattn_cutlass {

namespace detail {

using Sm80Int8MmaAtom = cute::MMA_Atom<cute::SM80_16x8x32_S32S8S8S32_TN>;
using Sm80Int8Mma = cute::TiledMMA<
    Sm80Int8MmaAtom,
    cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>,
    cute::Tile<cute::_16, cute::_16, cute::_32>>;

// Lane bits [b0,b1,b2,c0,c1] address rows [b0,c0,b1,b2,c1]. This lets
// LDSM.x4 return each group of four consecutive reduction bytes in two words.
using Sm80Int8TransposedBSourceValueLayout = cute::Layout<
    cute::Shape<cute::Shape<cute::_2, cute::_2, cute::_2, cute::_2, cute::_2>, cute::_16>,
    cute::Stride<cute::Stride<cute::_16, cute::_64, cute::_128, cute::_32, cute::_256>, cute::_1>>;
using Sm80Int8TransposedBDestValueLayout = decltype(Sm80Int8Mma{}.get_layoutB_TV());

} // namespace detail

struct SM80_U32x4_LDSM_T_INT8_B
{
  using SRegisters = cute::uint128_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void copy(cute::uint128_t const &smem_src,
                                    uint32_t &dst0,
                                    uint32_t &dst1,
                                    uint32_t &dst2,
                                    uint32_t &dst3)
  {
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
    const uint32_t smem_int_ptr = cute::cast_smem_ptr_to_uint(&smem_src);
    uint32_t l0;
    uint32_t l1;
    uint32_t l2;
    uint32_t l3;
    asm volatile(
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(l0), "=r"(l1), "=r"(l2), "=r"(l3)
        : "r"(smem_int_ptr));

    const int32_t lane_id = static_cast<int32_t>(threadIdx.x) & 31;
    const int32_t output_group = lane_id >> 2;
    const int32_t lane_in_group = lane_id & 3;
    const int32_t lower_src_lane = (output_group >> 1) * 4 + lane_in_group;
    const int32_t upper_src_lane = (4 + (output_group >> 1)) * 4 + lane_in_group;
    const uint32_t selector = (output_group & 1) != 0 ? 0x7531u : 0x6420u;
    constexpr uint32_t kFullWarpMask = 0xffffffffu;

    dst0 = __byte_perm(__shfl_sync(kFullWarpMask, l0, lower_src_lane),
                       __shfl_sync(kFullWarpMask, l1, lower_src_lane),
                       selector);
    dst1 = __byte_perm(__shfl_sync(kFullWarpMask, l2, lower_src_lane),
                       __shfl_sync(kFullWarpMask, l3, lower_src_lane),
                       selector);
    dst2 = __byte_perm(__shfl_sync(kFullWarpMask, l0, upper_src_lane),
                       __shfl_sync(kFullWarpMask, l1, upper_src_lane),
                       selector);
    dst3 = __byte_perm(__shfl_sync(kFullWarpMask, l2, upper_src_lane),
                       __shfl_sync(kFullWarpMask, l3, upper_src_lane),
                       selector);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use SM80 int8 transposed LDSM without LDSM support.");
#endif
  }
};

} // namespace sageattention::qattn_cutlass

namespace cute {

template <>
struct Copy_Traits<sageattention::qattn_cutlass::SM80_U32x4_LDSM_T_INT8_B>
{
  using ThrID = Layout<_32>;
  using SrcLayout = decltype(recast_layout<int8_t, uint1_t>(
      sageattention::qattn_cutlass::detail::Sm80Int8TransposedBSourceValueLayout{}));
  using DstLayout = decltype(recast_layout<int8_t, uint1_t>(
      sageattention::qattn_cutlass::detail::Sm80Int8TransposedBDestValueLayout{}));
  using RefLayout = DstLayout;
};

} // namespace cute

namespace sageattention::qattn_cutlass {

using SmemCopyAtomInt8TransposedB = cute::Copy_Atom<SM80_U32x4_LDSM_T_INT8_B, int8_t>;

static_assert(cute::size<0>(typename SmemCopyAtomInt8TransposedB::ValLayoutSrc{}) == cute::_32{});
static_assert(cute::size<1>(typename SmemCopyAtomInt8TransposedB::ValLayoutSrc{}) == cute::_16{});
static_assert(cute::size<0>(typename SmemCopyAtomInt8TransposedB::ValLayoutDst{}) == cute::_32{});
static_assert(cute::size<1>(typename SmemCopyAtomInt8TransposedB::ValLayoutDst{}) == cute::_16{});
static_assert(cute::coalesce(typename SmemCopyAtomInt8TransposedB::ValLayoutDst{}) ==
              cute::coalesce(detail::Sm80Int8TransposedBDestValueLayout{}));

template <typename Tensor>
CUTE_HOST_DEVICE auto make_int8_transposed_b_view(Tensor tensor)
{
  CUTE_STATIC_ASSERT_V(cute::rank(tensor) == cute::_2{});
  CUTE_STATIC_ASSERT_V(cute::size<0>(tensor) == cute::_32{});
  CUTE_STATIC_ASSERT_V(cute::size<1>(tensor) == cute::_16{});
  const auto transpose = cute::make_layout(cute::make_shape(cute::_16{}, cute::_32{}), cute::GenRowMajor{});
  return cute::make_tensor(tensor.data(), cute::composition(tensor.layout(), transpose));
}

} // namespace sageattention::qattn_cutlass

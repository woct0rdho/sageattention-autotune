#pragma once

#include <cute/arch/copy_sm80.hpp>
#include <cute/atom/copy_traits.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

namespace sageattention::qattn_cutlass {

template <class TS, class TD = TS>
struct SM80_CP_ASYNC_CACHEGLOBAL_NOFILL
{
  using SRegisters = TS[1];
  using DRegisters = TD[1];

  static_assert(sizeof(TS) == sizeof(TD), "cp.async requires sizeof(src_value_type) == sizeof(dst_value_type)");
  static_assert(sizeof(TS) == 16, "cp.async.cg no-fill path expects a 16-byte vector");

  CUTE_HOST_DEVICE static void copy(TS const &gmem_src,
                                    TD       &smem_dst,
                                    bool      pred)
  {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    TS const *gmem_ptr = &gmem_src;
    uint32_t smem_int_ptr = cute::cast_smem_ptr_to_uint(&smem_dst);
    int32_t predicate_int = static_cast<int32_t>(pred);
    asm volatile(
        "{\n"
        " .reg .pred p;\n"
        " setp.ne.b32 p, %0, 0;\n"
        " @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
        "}\n" : : "r"(predicate_int),
        "r"(smem_int_ptr), "l"(gmem_ptr), "n"(sizeof(TS)));
#else
    CUTE_INVALID_CONTROL_PATH("Support for cp.async instructions has not been enabled");
#endif
  }
};

} // namespace sageattention::qattn_cutlass

namespace cute {

template <class S, class D>
struct Copy_Traits<sageattention::qattn_cutlass::SM80_CP_ASYNC_CACHEGLOBAL_NOFILL<S, D>>
{
  using ThrID = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<S>::value>>>;
  using DstLayout = Layout<Shape<_1, Int<sizeof_bits<D>::value>>>;
  using RefLayout = SrcLayout;

  bool pred = true;

  CUTE_HOST_DEVICE constexpr
  Copy_Traits<sageattention::qattn_cutlass::SM80_CP_ASYNC_CACHEGLOBAL_NOFILL<S, D>>
  with(bool pred_) const
  {
    return {pred_};
  }

  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits const &traits,
              Tensor<TS, SLayout> const &src,
              Tensor<TD, DLayout>       &dst)
  {
    static_assert(is_gmem<TS>::value, "Expected gmem source for cp.async.");
    static_assert(is_smem<TD>::value, "Expected smem destination for cp.async.");

    Tensor rS = recast<S>(src);
    Tensor rD = recast<D>(dst);

    CUTE_STATIC_ASSERT_V(size(rS) == Int<1>{},
      "In CopyAtom, src layout doesn't vectorize into registers. This src layout is incompatible with this tiled copy.");
    CUTE_STATIC_ASSERT_V(size(rD) == Int<1>{},
      "In CopyAtom, dst layout doesn't vectorize into registers. This dst layout is incompatible with this tiled copy.");

    sageattention::qattn_cutlass::SM80_CP_ASYNC_CACHEGLOBAL_NOFILL<S, D>::copy(rS[0], rD[0], traits.pred);
  }
};

} // namespace cute

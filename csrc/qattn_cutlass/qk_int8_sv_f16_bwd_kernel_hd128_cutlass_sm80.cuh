#pragma once

namespace sageattention::qattn_cutlass_bwd {

template <int32_t HeadDim, int32_t CtaM, int32_t CtaN, int32_t NumWarps, int32_t QuantBlockQ, int32_t QuantBlockK, bool IsAligned, bool SmoothK>
__global__ __maxnreg__(255)
void fused_mma_kernel_hd128_2d(const int8_t *__restrict__ const Q,
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
                               float *__restrict__ const dSSum,
                               half *__restrict__ const dK,
                               half *__restrict__ const dV,
                               const Params params,
                               const float sm_scale)
{
  using Traits = BwdTileTraits<HeadDim, CtaM, CtaN, NumWarps>;
  using Storage = SharedStorageHD128<Traits>;
  static_assert(HeadDim == 128 && CtaM == 64 && CtaN == 64 && NumWarps == 8, "D128 kernel is specialized for the M64xN64 eight-warp schedule");
  static_assert(QuantBlockQ == 32 && QuantBlockK == 64, "D128 kernel implements the QBlock=32/KBlock=64 quantization format");
  static_assert(Traits::kCtaNMicroTiles == NumWarps / 2 && Traits::kCtaMMicroTiles == 4, "D128 ownership requires one warp per (M-half, N16) tile");

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
  const auto gdO = make_head_matrix_view<HeadDim>(dO, params, batch, head, params.stride_bz_dO, params.stride_seq_dO, params.stride_h_dO);
  const auto gdOInt8 = make_workspace_matrix_view<HeadDim>(dOInt8, params, batch, head);
  const auto gdK = make_head_matrix_view<HeadDim>(dK, params, batch, head, params.stride_bz_dK, params.stride_seq_dK, params.stride_h_dK);
  const auto gdV = make_head_matrix_view<HeadDim>(dV, params, batch, head, params.stride_bz_dV, params.stride_seq_dV, params.stride_h_dV);
  const auto gLse = make_head_vector_view(Lse, params, batch, head);
  const auto gDelta = make_head_vector_view(Delta, params, batch, head);
  constexpr int32_t kPaddedSeqLenMultiple = 2 * Traits::kBlockM;
  const int32_t padded_seq_len = ((params.seq_len + kPaddedSeqLenMultiple - 1) / kPaddedSeqLenMultiple) * kPaddedSeqLenMultiple;
  float *gdSSum = nullptr;
  if constexpr (SmoothK)
  {
    gdSSum = dSSum + (batch * params.num_heads + head) * padded_seq_len;
  }

  constexpr int32_t kDimBlocks = HeadDim / Traits::kBlockK;
  constexpr int32_t kdQNPairs = Traits::kCtaN / (2 * Traits::kBlockN);
  constexpr int32_t kdQDimBlocksPerOwner = kDimBlocks / kdQNPairs;
  static_assert(kDimBlocks == 8 && kdQNPairs == 2 && kdQDimBlocksPerOwner == 4);
  const int32_t pair_blocks = (params.seq_len + 2 * Traits::kBlockM - 1) / (2 * Traits::kBlockM);
  const float *const gdOScale = dOScale + (batch * params.num_heads + head) * pair_blocks * kDimBlocks;
  const int32_t dS_q_extent = (params.seq_len + QuantBlockQ - 1) / QuantBlockQ;
  const int32_t dS_k_extent = (params.seq_len + QuantBlockK - 1) / QuantBlockK;
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
  constexpr int32_t q_stage_offset = cute::cosize_v<typename Traits::SmemLayoutQ>;
  constexpr int32_t dO_stage_offset = cute::cosize_v<typename Traits::SmemLayoutdO>;
  constexpr int32_t dO_fp16_stage_offset = cute::cosize_v<typename Storage::SmemLayoutdOFp16Pair>;
  constexpr int32_t kCtaNLoadRows = Traits::kCtaNLoadRows;

  auto sKStorage = make_smem_tensor(shared.k_i8, typename Traits::SmemLayoutK{});
  auto sVStorage = make_smem_tensor(shared.v, typename Traits::SmemLayoutV{});
  auto sK = cute::recast<int8_t>(sKStorage);
  auto sV = cute::recast<half>(sVStorage);
  auto sLse = make_smem_tensor(shared.lse, typename Storage::SmemLayoutMPair{});
  auto sDelta = make_smem_tensor(shared.delta, typename Storage::SmemLayoutMPair{});

  auto sScorePairStorage = make_smem_tensor(shared.score_pair_i8, typename Traits::SmemLayoutScorePair{}, n_tile * score_pair_offset);
  auto sScorePair = cute::recast<int8_t>(sScorePairStorage);
  constexpr auto score_pair_tile_shape = cute::make_shape(cute::Int<Traits::kBlockN>{}, cute::Int<Traits::kScoreAtomK>{});
  auto sPPair = cute::local_tile(sScorePair, score_pair_tile_shape, cute::make_coord(cute::_0{}, cute::_0{}));
  auto sdSPair = cute::local_tile(sScorePair, score_pair_tile_shape, cute::make_coord(cute::_0{}, cute::_1{}));

  const int32_t dS_slot = 2 * n_pair + m_half;
  auto sWarpScratchStorage = make_smem_tensor(shared.warp_scratch.dS_dKV, typename Traits::SmemLayoutdSdKV{}, dS_slot * warp_scratch_offset);
  auto sWarpScratch = cute::recast<int8_t>(sWarpScratchStorage);
  constexpr auto dS_tile_shape = cute::make_shape(cute::Int<Traits::kBlockM>{}, cute::Int<Traits::kScoreAtomK>{});
  auto sdSMirror = cute::local_tile(sWarpScratch, dS_tile_shape, cute::make_coord(cute::_0{}, cute::_0{}));

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
  constexpr int32_t kTailLoaderWarpBase = 2;
  constexpr int32_t kRowStateLoaderWarp = 6;
  const bool is_tail_loader = warp_id >= kTailLoaderWarpBase && warp_id < kTailLoaderWarpBase + Traits::kMStageLoadWarps;

  constexpr auto cta_k_load_shape = cute::make_shape(cute::Int<kCtaNLoadRows>{}, cute::Int<Traits::kInt8LoadVecCols>{});
  constexpr auto cta_v_load_shape = cute::make_shape(cute::Int<kCtaNLoadRows>{}, cute::Int<Traits::kHalfLoadVecCols>{});
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
      constexpr auto k_pair_shape = cute::make_shape(cute::Int<kKPairRows>{}, cute::Int<Traits::kBlockK>{});
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
      store_packed_mma_b_fragment(shared.k_i8_mma_b, fragment, warp_id * kDimBlocks + dim_block, lane_id);
    }
  }
  auto sQStage0Storage = make_smem_tensor(shared.q_i8, typename Traits::SmemLayoutQ{}, 0);
  auto sdOInt8Stage0Storage = make_smem_tensor(shared.dO_i8, typename Traits::SmemLayoutdO{}, 0);
  auto sdOFp16Stage0Storage = make_smem_tensor(shared.dO_fp16_pair, typename Storage::SmemLayoutdOFp16Pair{}, 0);
  if (is_tail_loader)
  {
    load_q_dO_pair<Traits>(
      gQ,
      gdOInt8,
      gdO,
      sQStage0Storage,
      sdOInt8Stage0Storage,
      sdOFp16Stage0Storage,
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

  auto dKV_accum_frag = cute::make_tensor<float>(cute::make_shape(cute::Int<kDimBlocks>{}, cute::Int<kAccumulatorElements>{}), cute::LayoutRight{});
  cute::clear(dKV_accum_frag);

  const int32_t k_domain_base = n_cta_base / QuantBlockK;
  const int32_t k_scale_index = k_domain_base < k_scale_extent ? k_domain_base : k_scale_extent - 1;
  const int32_t k_factor_index = k_domain_base < dS_k_extent ? k_domain_base : dS_k_extent - 1;
  const float k_block_scale = gKScale(k_scale_index);
  for (int32_t m_pair_base = 0; m_pair_base < params.seq_len; m_pair_base += 2 * Traits::kBlockM)
  {
    const int32_t q_block_index = m_pair_base / QuantBlockQ;
    const float q_block_scale = gQScale(q_block_index);
    const int32_t m_base = m_pair_base + m_half * Traits::kBlockM;
    const int32_t q_dO_stage = q_block_index & 1;
    auto sQStorage = make_smem_tensor(shared.q_i8, typename Traits::SmemLayoutQ{}, q_dO_stage * q_stage_offset);
    auto sdOInt8Storage = make_smem_tensor(shared.dO_i8, typename Traits::SmemLayoutdO{}, q_dO_stage * dO_stage_offset);
    auto sdOFp16PairStorage = make_smem_tensor(shared.dO_fp16_pair, typename Storage::SmemLayoutdOFp16Pair{}, q_dO_stage * dO_fp16_stage_offset);
    auto sQ = cute::recast<int8_t>(sQStorage);
    auto sdOInt8 = cute::recast<int8_t>(sdOInt8Storage);
    auto sdOFp16Pair = cute::recast<half>(sdOFp16PairStorage);
    auto acc_score = cute::partition_fragment_C(score_mma, BlockMNShape{});
    auto acc_dp = cute::partition_fragment_C(half_mma, BlockMNShape{});
    cute::clear(acc_score);
    cute::clear(acc_dp);

    if (n_valid)
    {
#pragma unroll
      for (int32_t dim_base = 0; dim_base < HeadDim; dim_base += Traits::kScoreAtomK)
      {
        constexpr auto score_subtile_shape = cute::make_shape(cute::Int<Traits::kBlockM>{}, cute::Int<Traits::kScoreAtomK>{});
        const auto sQTile = cute::local_tile(
          sQ,
          score_subtile_shape,
          cute::make_coord(m_half, dim_base / Traits::kScoreAtomK));
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
        constexpr auto half_subtile_shape = cute::make_shape(cute::Int<Traits::kBlockM>{}, cute::Int<Traits::kBlockK>{});
        const auto sdOTile = cute::local_tile(
          sdOFp16Pair,
          half_subtile_shape,
          cute::make_coord(m_half, dim_base / Traits::kBlockK));
        const auto sVTile = cute::local_tile(
          sV,
          cute::make_shape(cute::Int<Traits::kBlockN>{}, cute::Int<Traits::kBlockK>{}),
          cute::make_coord(n_tile, dim_base / Traits::kBlockK));
        const auto sdOHalf = cute::recast<cute::half_t>(sdOTile);
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
    const float score_scale = q_block_scale * k_block_scale;
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
      float dS = 0.0f;
      if (IsAligned || (row < params.seq_len && col < params.seq_len))
      {
        const bool upper_row = (idx & 2) != 0;
        const float score = static_cast<float>(acc_score(idx)) * score_scale;
        p = expf(score * sm_scale - (upper_row ? lse_1 : lse_0));
        dS = p * (acc_dp(idx) - (upper_row ? delta_1 : delta_0)) * sm_scale;
        p_max_abs = fmaxf(p_max_abs, p);
      }
      rPFloat(idx) = p;
      acc_dp(idx) = dS;
    }
    if constexpr (SmoothK)
    {
      if (n_valid)
      {
        accumulate_dS_row_sum<IsAligned>(acc_dp, gdSSum, m_base, params.seq_len, lane_id);
      }
    }
    p_max_abs = warp_reduce_max(p_max_abs);
    if (lane_id == 0)
    {
      shared.p_scale[warp_id] = p_max_abs * kInt8ScaleInv + kInt8ScaleFloor;
    }
    __syncthreads();

    const float p_scale = fmaxf(shared.p_scale[warp_id], shared.p_scale[warp_id ^ 1]);
    const int32_t q_factor_next = q_block_index + 1 < dS_q_extent ? q_block_index + 1 : q_block_index;
    const float dO_l2_max = fmaxf(gdSQFactors[2 * q_block_index], gdSQFactors[2 * q_factor_next]);
    const float delta_abs_max = fmaxf(gdSQFactors[2 * q_block_index + 1], gdSQFactors[2 * q_factor_next + 1]);
    const float predicted_dS_max = kdSPredictorGuard * sm_scale / static_cast<float>(params.seq_len) *
      (dO_l2_max * gdSKFactors[k_factor_index] + delta_abs_max);
    const float dS_scale = predicted_dS_max * kInt8ScaleInv + kInt8ScaleFloor;
    const float inv_p_scale = 1.0f / p_scale;
    const float inv_dS_scale = 1.0f / dS_scale;
    constexpr auto transposed_store_shape = cute::make_shape(cute::Int<Traits::kBlockN>{}, cute::Int<Traits::kBlockM>{});
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
        int8_t dS_i8 = 0;
        if (IsAligned || (row < params.seq_len && col < params.seq_len))
        {
          p_i8 = round_to_int8(rPFloat(idx) * inv_p_scale);
          dS_i8 = round_to_int8(acc_dp(idx) * inv_dS_scale);
        }
        rP(idx) = p_i8;
        rdS(idx) = dS_i8;
      }
      auto tCrP = thr_copy_score_c.retile_S(rP);
      auto tCrdS = thr_copy_score_c.retile_S(rdS);
      cute::copy(tiled_copy_score_c, tCrP, thr_copy_score_c.partition_D(sP));
      cute::copy(tiled_copy_score_c, tCrdS, thr_copy_score_c.partition_D(sdS));
      constexpr auto score_store_shape = cute::make_shape(cute::Int<Traits::kBlockM>{}, cute::Int<Traits::kBlockN>{});
      auto sdSMirrorHalf = cute::local_tile(
        sdSMirror,
        score_store_shape,
        cute::make_coord(cute::_0{}, n_tile & 1));
      cute::copy(tiled_copy_score_c, tCrdS, thr_copy_score_c.partition_D(sdSMirrorHalf));
    }
    __syncthreads();

    if (n_valid && m_half == 0)
    {
#pragma unroll
      for (int32_t dim_base = 0; dim_base < HeadDim; dim_base += Traits::kBlockK)
      {
        constexpr auto dO_pair_shape = cute::make_shape(cute::Int<2 * Traits::kBlockM>{}, cute::Int<Traits::kBlockK>{});
        const auto sdOPair = cute::local_tile(
          sdOInt8,
          dO_pair_shape,
          cute::make_coord(cute::_0{}, dim_base / Traits::kBlockK));
        const auto sdOdV = qattn_cutlass::make_int8_transposed_b_view(sdOPair);
        auto dV_acc = cute::partition_fragment_C(score_mma, BlockMNShape{});
        cute::clear(dV_acc);
        auto tdVP = thr_mma_score.partition_fragment_A(sPPair);
        auto tdVdO = thr_mma_score.partition_fragment_B(sdOdV);
        cute::copy(tiled_copy_score_a, thr_copy_score_a.partition_S(sPPair), thr_copy_score_a.retile_D(tdVP));
        cute::copy(tiled_copy_transposed_b, thr_copy_transposed_b.partition_S(sdOdV), thr_copy_transposed_b.retile_D(tdVdO));
        cute::gemm(thr_mma_score, tdVP, tdVdO, dV_acc);
        const int32_t dim_block = dim_base / Traits::kBlockK;
        const float dV_scale = p_scale * gdOScale[(m_pair_base / (2 * Traits::kBlockM)) * kDimBlocks + dim_block];
#pragma unroll
        for (int32_t idx = 0; idx < cute::size(dV_acc); ++idx)
        {
          dKV_accum_frag(dim_block, idx) =
            fmaf(static_cast<float>(dV_acc(idx)), dV_scale, dKV_accum_frag(dim_block, idx));
        }
      }
    }
    else if (n_valid)
    {
#pragma unroll
      for (int32_t dim_base = 0; dim_base < HeadDim; dim_base += Traits::kBlockK)
      {
        constexpr auto q_pair_shape = cute::make_shape(cute::Int<2 * Traits::kBlockM>{}, cute::Int<Traits::kBlockK>{});
        const auto sQPair = cute::local_tile(
          sQ,
          q_pair_shape,
          cute::make_coord(cute::_0{}, dim_base / Traits::kBlockK));
        const auto sQdK = qattn_cutlass::make_int8_transposed_b_view(sQPair);
        auto dK_acc = cute::partition_fragment_C(score_mma, BlockMNShape{});
        cute::clear(dK_acc);
        auto tdKdS = thr_mma_score.partition_fragment_A(sdSPair);
        auto tdKQ = thr_mma_score.partition_fragment_B(sQdK);
        cute::copy(tiled_copy_score_a, thr_copy_score_a.partition_S(sdSPair), thr_copy_score_a.retile_D(tdKdS));
        cute::copy(tiled_copy_transposed_b, thr_copy_transposed_b.partition_S(sQdK), thr_copy_transposed_b.retile_D(tdKQ));
        cute::gemm(thr_mma_score, tdKdS, tdKQ, dK_acc);
        const int32_t dim_block = dim_base / Traits::kBlockK;
        const float dK_scale = dS_scale * q_block_scale;
#pragma unroll
        for (int32_t idx = 0; idx < cute::size(dK_acc); ++idx)
        {
          dKV_accum_frag(dim_block, idx) =
            fmaf(static_cast<float>(dK_acc(idx)), dK_scale, dKV_accum_frag(dim_block, idx));
        }
      }
    }

    const int32_t next_m_pair_base = m_pair_base + 2 * Traits::kBlockM;
    const bool has_next_m_pair = next_m_pair_base < params.seq_len;
    if (has_next_m_pair && is_tail_loader)
    {
      const int32_t next_stage = q_dO_stage ^ 1;
      auto sQNextStorage = make_smem_tensor(shared.q_i8, typename Traits::SmemLayoutQ{}, next_stage * q_stage_offset);
      auto sdOInt8NextStorage = make_smem_tensor(shared.dO_i8, typename Traits::SmemLayoutdO{}, next_stage * dO_stage_offset);
      auto sdOFp16NextStorage = make_smem_tensor(shared.dO_fp16_pair, typename Storage::SmemLayoutdOFp16Pair{}, next_stage * dO_fp16_stage_offset);
      load_q_dO_pair<Traits>(
        gQ,
        gdOInt8,
        gdO,
        sQNextStorage,
        sdOInt8NextStorage,
        sdOFp16NextStorage,
        next_m_pair_base,
        warp_id - kTailLoaderWarpBase,
        lane_id);
    }
    if (has_next_m_pair && warp_id == kRowStateLoaderWarp)
    {
      load_row_state_pair(gLse, gDelta, sLse, sDelta, next_m_pair_base, params.seq_len, lane_id);
    }

    if ((n_tile & 1) == 0)
    {
      const int32_t dQ_dim_begin = n_pair * kdQDimBlocksPerOwner * Traits::kBlockK;
#pragma unroll
      for (int32_t dim_offset = 0; dim_offset < kdQDimBlocksPerOwner * Traits::kBlockK; dim_offset += Traits::kBlockK)
      {
        const int32_t dim_base = dQ_dim_begin + dim_offset;
        auto dQ_acc = cute::partition_fragment_C(score_mma, BlockMNShape{});
        cute::clear(dQ_acc);
#pragma unroll
        for (int32_t dQ_pair = 0; dQ_pair < kdQNPairs; ++dQ_pair)
        {
          const int32_t dQ_dS_slot = 2 * dQ_pair + m_half;
          auto sdQScratchStorage = make_smem_tensor(shared.warp_scratch.dS_dKV, typename Traits::SmemLayoutdSdKV{}, dQ_dS_slot * warp_scratch_offset);
          auto sdQScratch = cute::recast<int8_t>(sdQScratchStorage);
          auto sdSdQ = cute::local_tile(sdQScratch, dS_tile_shape, cute::make_coord(cute::_0{}, cute::_0{}));
          constexpr auto k_pair_shape = cute::make_shape(cute::Int<2 * Traits::kBlockN>{}, cute::Int<Traits::kBlockK>{});
          const auto sKPair = cute::local_tile(
            sK,
            k_pair_shape,
            cute::make_coord(dQ_pair, dim_base / Traits::kBlockK));
          const auto sKdQ = qattn_cutlass::make_int8_transposed_b_view(sKPair);
          auto tdQdS = thr_mma_score.partition_fragment_A(sdSdQ);
          auto tdQK = thr_mma_score.partition_fragment_B(sKdQ);
          cute::copy(tiled_copy_score_a, thr_copy_score_a.partition_S(sdSdQ), thr_copy_score_a.retile_D(tdQdS));
          load_packed_mma_b_fragment(shared.k_i8_mma_b, tdQK, dQ_pair * kDimBlocks + dim_base / Traits::kBlockK, lane_id);
          cute::gemm(thr_mma_score, tdQdS, tdQK, dQ_acc);
        }
        auto dQ_float = cute::make_fragment_like<float>(dQ_acc);
        const float dQ_scale = dS_scale * k_block_scale;
#pragma unroll
        for (int32_t idx = 0; idx < cute::size(dQ_acc); ++idx)
        {
          dQ_float(idx) = static_cast<float>(dQ_acc(idx)) * dQ_scale;
        }
        float *const dQ_head = dQAccum + batch * params.stride_bz_dQAccum + head * params.stride_h_dQAccum;
        accumulate_dQ_float_fragment_workspace<IsAligned, Traits>(
          dQ_float,
          dQ_head,
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

  half *const dKV_stage_base = reinterpret_cast<half *>(shared.warp_scratch.dS_dKV.begin());
  constexpr int32_t dKV_stage_stride = warp_scratch_offset * packed_elements<half>();
  half *const dKV_stage = dKV_stage_base + warp_id * dKV_stage_stride;
  if (n_valid)
  {
#pragma unroll
    for (int32_t dim_block = 0; dim_block < kDimBlocks; ++dim_block)
    {
      if (m_half == 0)
      {
        store_dKV_fragment_coalesced<Traits>(
          dKV_accum_frag,
          dim_block,
          gdV,
          dKV_stage,
          n_base,
          params.seq_len,
          lane_id);
      }
      else
      {
        store_dKV_fragment_coalesced<Traits>(
          dKV_accum_frag,
          dim_block,
          gdK,
          dKV_stage,
          n_base,
          params.seq_len,
          lane_id);
      }
    }
  }
}

} // namespace sageattention::qattn_cutlass_bwd

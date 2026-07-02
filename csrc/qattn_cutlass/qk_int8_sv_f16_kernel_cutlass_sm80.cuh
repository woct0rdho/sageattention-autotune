#pragma once

#include "../qattn/attn_utils.cuh"

#include <cute/arch/mma_sm80.hpp>
#include <cuda_fp16.h>

#include <type_traits>

namespace sageattention::qattn_cutlass {

constexpr uint32_t PACK_SIZE_QK = 16;
constexpr uint32_t PACK_SIZE_V = 8;
constexpr uint32_t PACK_SIZE_O = 8;

constexpr uint32_t MMA_QK_M = 16;
constexpr uint32_t MMA_QK_N = 16;
constexpr uint32_t MMA_QK_K = 32;

constexpr uint32_t MMA_SV_N = 16;

template <bool Init>
__device__ __forceinline__ void mma_s8s8s32_m16n16k32(int32_t *c, const uint32_t *a, const uint32_t *b)
{
  uint32_t *c_u32 = reinterpret_cast<uint32_t*>(c);
  uint32_t d0, d1, d2, d3;

  cute::SM80_16x8x32_S32S8S8S32_TN::fma(
    d0, d1, d2, d3,
    a[0], a[1], a[2], a[3],
    b[0], b[1],
    Init ? 0u : c_u32[0], Init ? 0u : c_u32[1], Init ? 0u : c_u32[2], Init ? 0u : c_u32[3]);
  c_u32[0] = d0;
  c_u32[1] = d1;
  c_u32[2] = d2;
  c_u32[3] = d3;

  cute::SM80_16x8x32_S32S8S8S32_TN::fma(
    d0, d1, d2, d3,
    a[0], a[1], a[2], a[3],
    b[2], b[3],
    Init ? 0u : c_u32[4], Init ? 0u : c_u32[5], Init ? 0u : c_u32[6], Init ? 0u : c_u32[7]);
  c_u32[4] = d0;
  c_u32[5] = d1;
  c_u32[6] = d2;
  c_u32[7] = d3;
}

__device__ __forceinline__ float load_q_scale_broadcast(const float *__restrict__ q_scale_ptr, const uint32_t q_scale_idx, const uint32_t lane_id)
{
  float scale = 0.0f;
  if ((lane_id & 3) == 0)
  {
    scale = q_scale_ptr[q_scale_idx];
  }
  return __shfl_sync(0xffffffff, scale, lane_id & ~3u);
}

__device__ __forceinline__ float load_k_scale_broadcast(const float *__restrict__ k_scale_ptr, const uint32_t k_scale_idx, const uint32_t lane_id)
{
  float scale = 0.0f;
  if (lane_id < 4)
  {
    scale = k_scale_ptr[k_scale_idx];
  }
  return __shfl_sync(0xffffffff, scale, lane_id & 3u);
}

__device__ __forceinline__ void mma_f16f16f32_m16n16k16(float *c, const uint32_t *a, const uint32_t *b)
{
  cute::SM80_16x8x16_F32F16F16F32_TN::fma(
    c[0], c[1], c[2], c[3],
    a[0], a[1], a[2], a[3],
    b[0], b[1],
    c[0], c[1], c[2], c[3]);

  cute::SM80_16x8x16_F32F16F16F32_TN::fma(
    c[4], c[5], c[6], c[7],
    a[0], a[1], a[2], a[3],
    b[2], b[3],
    c[4], c[5], c[6], c[7]);
}

template <uint32_t num_warps_q, uint32_t num_warps_k,
          uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_qk_inner,
          SwizzleMode swizzle_mode, uint32_t stride>
__device__ __forceinline__ void compute_int_qk(const smem_t<swizzle_mode, stride> &smem_Q,
                                               const smem_t<swizzle_mode, stride> &smem_K,
                                               int32_t RS[][num_tiles_k][8],
                                               uint32_t &offset_Q,
                                               uint32_t &offset_K)
{
  uint32_t RQ[num_tiles_q][4];
  uint32_t RK[4];

#pragma unroll
  for (uint32_t iter = 0; iter < 1; iter++)
  {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      smem_Q.ldmatrix_m8n8x4(offset_Q, RQ[fq]);
      offset_Q = smem_Q.template advance_offset_by_row<16>(offset_Q);
    }
    offset_Q = smem_Q.template advance_offset_by_column<2>(offset_Q - (num_tiles_q * 16 * stride), iter);

#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++)
    {
      smem_K.ldmatrix_m8n8x4(offset_K, RK);
      offset_K = smem_K.template advance_offset_by_row<16>(offset_K);

#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        mma_s8s8s32_m16n16k32<true>(RS[fq][fk], RQ[fq], RK);
      }
    }
    offset_K = smem_K.template advance_offset_by_column<2>(offset_K - (num_tiles_k * 16 * stride), iter);
  }

#pragma unroll
  for (uint32_t iter = 1; iter < num_tiles_qk_inner; iter++)
  {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      smem_Q.ldmatrix_m8n8x4(offset_Q, RQ[fq]);
      offset_Q = smem_Q.template advance_offset_by_row<16>(offset_Q);
    }
    offset_Q = smem_Q.template advance_offset_by_column<2>(offset_Q - (num_tiles_q * 16 * stride), iter);

#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++)
    {
      smem_K.ldmatrix_m8n8x4(offset_K, RK);
      offset_K = smem_K.template advance_offset_by_row<16>(offset_K);

#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        mma_s8s8s32_m16n16k32<false>(RS[fq][fk], RQ[fq], RK);
      }
    }
    offset_K = smem_K.template advance_offset_by_column<2>(offset_K - (num_tiles_k * 16 * stride), iter);
  }

  offset_Q -= (2 * num_tiles_qk_inner);
  offset_K -= (2 * num_tiles_qk_inner);
}

template <uint32_t num_warps_q, uint32_t num_warps_k,
          uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_qk_inner,
          SwizzleMode swizzle_mode, uint32_t stride>
__device__ __forceinline__ void compute_int_qk(const smem_t<swizzle_mode, stride> &smem_K,
                                               int32_t RS[][num_tiles_k][8],
                                               uint32_t RQ[][4],
                                               uint32_t offset_K)
{
  static_assert(num_tiles_qk_inner == 1);

  uint32_t RK[4];

#pragma unroll
  for (uint32_t fk = 0; fk < num_tiles_k; fk++)
  {
    smem_K.ldmatrix_m8n8x4(offset_K, RK);
    offset_K = smem_K.template advance_offset_by_row<16>(offset_K);

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      mma_s8s8s32_m16n16k32<true>(RS[fq][fk], RQ[fq], RK);
    }
  }
}

template <uint32_t num_warps_q, uint32_t num_warps_k,
          uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v,
          SwizzleMode swizzle_mode, uint32_t stride>
__device__ __forceinline__ void compute_fp16_sv(const smem_t<swizzle_mode, stride> &smem_V,
                                                uint32_t RS_f16[][num_tiles_k][4],
                                                float RO[][num_tiles_v][8],
                                                uint32_t &offset_V)
{
#pragma unroll
  for (uint32_t fk = 0; fk < num_tiles_k; fk++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      uint32_t RV[4];
      smem_V.ldmatrix_m8n8x4_trans(offset_V, RV);

#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        mma_f16f16f32_m16n16k16(RO[fq][fv], reinterpret_cast<uint32_t*>(RS_f16[fq][fk]), RV);
      }

      offset_V = smem_V.template advance_offset_by_column<2>(offset_V, fv);
    }
    offset_V = smem_V.template advance_offset_by_row<16>(offset_V - (2 * num_tiles_v));
  }

  offset_V -= (16 * num_tiles_k * stride);
}

template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K, uint32_t head_dim, bool ReturnLse, bool BroadcastScaleLoads>
__global__ void qk_int8_sv_f16_accum_f32_attn_kernel(const int8_t *__restrict__ Q,
                                                     const int8_t *__restrict__ K,
                                                     const half *__restrict__ V,
                                                     half *__restrict__ O,
                                                     float *__restrict__ Lse,
                                                     const float *__restrict__ Q_scale,
                                                     const float *__restrict__ K_scale,
                                                     const uint32_t qo_len,
                                                     const uint32_t kv_len,
                                                     const uint32_t num_kv_groups,
                                                     const uint32_t stride_bz_q,
                                                     const uint32_t stride_seq_q,
                                                     const uint32_t stride_h_q,
                                                     const uint32_t stride_bz_k,
                                                     const uint32_t stride_seq_k,
                                                     const uint32_t stride_h_k,
                                                     const uint32_t stride_bz_v,
                                                     const uint32_t stride_seq_v,
                                                     const uint32_t stride_h_v,
                                                     const uint32_t stride_bz_o,
                                                     const uint32_t stride_seq_o,
                                                     const uint32_t stride_h_o,
                                                     float sm_scale)
{
  static_assert(head_dim == 64 || head_dim == 128, "CUTLASS qattn fwd currently supports head_dim 64 or 128");
  static_assert(WARP_K == CTA_K, "Initial CUTLASS qattn fwd requires one K warp partition");
  static_assert(CTA_Q % WARP_Q == 0);
  static_assert(CTA_K % WARP_K == 0);
  static_assert(head_dim % 64 == 0);

  constexpr uint32_t num_warps_q = CTA_Q / WARP_Q;
  constexpr uint32_t num_warps_k = CTA_K / WARP_K;
  constexpr uint32_t num_warps = num_warps_q * num_warps_k;
  constexpr uint32_t num_tiles_q = WARP_Q / MMA_QK_M;
  constexpr uint32_t num_tiles_k = WARP_K / MMA_QK_N;
  constexpr uint32_t num_tiles_qk_inner = head_dim / MMA_QK_K;
  constexpr uint32_t num_tiles_v = head_dim / MMA_SV_N;

  constexpr uint32_t QK_SMEM_STRIDE = head_dim;
  constexpr uint32_t O_SMEM_STRIDE = head_dim;
  constexpr uint32_t V_SMEM_STRIDE = head_dim;

  extern __shared__ int8_t smem[];

  const uint32_t lane_id = get_lane_id();
  const uint32_t warp_id = get_warp_id();
  const uint32_t batch_id = blockIdx.z;
  const uint32_t bx = blockIdx.x;
  const uint32_t num_qo_heads = gridDim.y;
  const uint32_t head_id = blockIdx.y;

  const float original_sm_scale = sm_scale * math::log2e;

  int32_t RS[num_tiles_q][num_tiles_k][8];
  float RO[num_tiles_q][num_tiles_v][8];
  float m[num_tiles_q][2];
  float d[num_tiles_q][2];

#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++)
      {
        RO[fq][fv][k] = 0.0f;
      }
    }
  }

#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++)
    {
      m[fq][k] = -5000000.0f;
      d[fq][k] = 1.0f;
    }
  }

  uint32_t q_scale_idx;
  uint32_t k_scale_idx;
  {
    const uint32_t num_warp_block_q = gridDim.x * num_warps_q;
    q_scale_idx = batch_id * num_qo_heads * (num_warp_block_q * 8) +
                  head_id * (num_warp_block_q * 8) +
                  bx * (num_warps_q * 8) +
                  get_warp_idx_q<num_warps_q, num_warps_k>() * 8 + lane_id / 4;
  }
  {
    const uint32_t num_warp_block_k = div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * (num_warp_block_k * 4) +
                  (head_id / num_kv_groups) * (num_warp_block_k * 4) +
                  get_warp_idx_k<num_warps_q, num_warps_k>() * 4 + lane_id % 4;
  }
  constexpr uint32_t k_scale_advance_offset = (CTA_K / WARP_K) * 4;

  constexpr uint32_t K_smem_idx_offset = CTA_Q;
  constexpr uint32_t V_smem_idx_offset = CTA_Q + CTA_K;

  constexpr SwizzleMode swizzle_mode_QK = (QK_SMEM_STRIDE == 32) ? SwizzleMode::k32B : (QK_SMEM_STRIDE == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_Q(smem);
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_K(smem + K_smem_idx_offset * QK_SMEM_STRIDE);
  constexpr SwizzleMode swizzle_mode_V = (V_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V> smem_V(smem + V_smem_idx_offset * QK_SMEM_STRIDE);
  constexpr SwizzleMode swizzle_mode_O = (O_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_O, O_SMEM_STRIDE / PACK_SIZE_O> smem_O(smem);

  constexpr uint32_t global_to_shared_line_lanes_QK = (QK_SMEM_STRIDE == 32) ? 2 : (QK_SMEM_STRIDE == 64) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_QK = (QK_SMEM_STRIDE == 32) ? 16 : (QK_SMEM_STRIDE == 64) ? 8 : 4;
  constexpr uint32_t global_to_shared_line_lanes_V = (V_SMEM_STRIDE == 32) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_V = (V_SMEM_STRIDE == 32) ? 8 : 4;
  constexpr uint32_t global_to_shared_line_lanes_O = (O_SMEM_STRIDE == 32) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_O = (O_SMEM_STRIDE == 32) ? 8 : 4;

  constexpr uint32_t QK_smem_iters_row = QK_SMEM_STRIDE / (global_to_shared_line_lanes_QK * PACK_SIZE_QK);
  constexpr uint32_t Q_smem_iters_col = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t K_smem_iters_col = CTA_K / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t V_smem_iters_row = V_SMEM_STRIDE / (global_to_shared_line_lanes_V * PACK_SIZE_V);
  constexpr uint32_t V_smem_iters_col = CTA_K / (num_warps * global_to_shared_copy_lines_per_warp_V);
  constexpr uint32_t O_smem_iters_row = O_SMEM_STRIDE / (global_to_shared_line_lanes_O * PACK_SIZE_O);
  constexpr uint32_t O_smem_iters_col = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_O);

  const int8_t *Q_lane_base_ptr = Q + batch_id * stride_bz_q + head_id * stride_h_q +
    (bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) * stride_seq_q +
    (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;
  const int8_t *K_lane_base_ptr = K + batch_id * stride_bz_k + (head_id / num_kv_groups) * stride_h_k +
    (CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) * stride_seq_k +
    (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;
  const half *V_lane_base_ptr = V + batch_id * stride_bz_v + (head_id / num_kv_groups) * stride_h_v +
    (CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_V) * stride_seq_v +
    (lane_id % global_to_shared_line_lanes_V) * PACK_SIZE_V;

  uint32_t Q_smem_offset_load = smem_Q.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_QK * Q_smem_iters_col + lane_id / global_to_shared_line_lanes_QK, lane_id % global_to_shared_line_lanes_QK);
  uint32_t K_smem_offset_load = smem_K.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_QK * K_smem_iters_col + lane_id / global_to_shared_line_lanes_QK, lane_id % global_to_shared_line_lanes_QK);
  uint32_t V_smem_offset_load = smem_V.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_V * V_smem_iters_col + lane_id / global_to_shared_line_lanes_V, lane_id % global_to_shared_line_lanes_V);

  uint32_t Q_smem_offset_mma = smem_Q.get_permuted_offset(get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id % 16, lane_id / 16);
  uint32_t K_smem_offset_mma = smem_K.get_permuted_offset(get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
  uint32_t V_smem_offset_mma = smem_V.get_permuted_offset(get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 16, lane_id / 16);

  uint32_t K_idx_lane_base = get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + 2 * (lane_id % 4);
  uint32_t Q_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;
  uint32_t K_load_idx_lane_base = CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;
  uint32_t V_load_idx_lane_base = CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_V;

  load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, Q_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_Q>(
    &Q_lane_base_ptr, Q_smem_offset_load, stride_seq_q, smem_Q, Q_load_idx_lane_base, qo_len);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncwarp();

  uint32_t RQ[num_tiles_q][4];
  if constexpr (num_tiles_qk_inner == 1)
  {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      smem_Q.ldmatrix_m8n8x4(Q_smem_offset_mma, RQ[fq]);
      Q_smem_offset_mma = smem_Q.template advance_offset_by_row<16>(Q_smem_offset_mma);
    }
  }

  const uint32_t num_iterations = div_ceil(kv_len, CTA_K);
  const float q_scale = BroadcastScaleLoads ? load_q_scale_broadcast(Q_scale, q_scale_idx, lane_id) : Q_scale[q_scale_idx];
  float k_scale = BroadcastScaleLoads ? load_k_scale_broadcast(K_scale, k_scale_idx + 0 * k_scale_advance_offset, lane_id) : K_scale[k_scale_idx + 0 * k_scale_advance_offset];
  float dequant_scale = q_scale * k_scale;

  load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
    &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K, K_load_idx_lane_base, kv_len);
  cp_async::commit_group();

  load_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
    &V_lane_base_ptr, V_smem_offset_load, stride_seq_v, smem_V, V_load_idx_lane_base, kv_len);
  cp_async::commit_group();

  K_load_idx_lane_base += CTA_K;
  V_load_idx_lane_base += CTA_K;

#pragma unroll 1
  for (uint32_t iter = 1; iter < num_iterations - 1; iter++)
  {
    cp_async::wait_group<1>();
    __syncthreads();

    if constexpr (num_tiles_qk_inner == 1)
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK>(
        smem_K, RS, RQ, K_smem_offset_mma);
    }
    else
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK>(
        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }

    float RS_f32[num_tiles_q][num_tiles_k][8];
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]);
        }
      }
    }

    K_idx_lane_base += CTA_K;

    update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false>(RS_f32, RO, m, d, original_sm_scale * dequant_scale);
    accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);

    uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
    RS_32_to_16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);

    __syncthreads();

    load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
      &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K);
    cp_async::commit_group();

    k_scale = BroadcastScaleLoads ? load_k_scale_broadcast(K_scale, k_scale_idx + iter * k_scale_advance_offset, lane_id) : K_scale[k_scale_idx + iter * k_scale_advance_offset];
    dequant_scale = q_scale * k_scale;

    cp_async::wait_group<1>();
    __syncthreads();

    compute_fp16_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
      smem_V, RS_f16, RO, V_smem_offset_mma);

    __syncthreads();

    load_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
      &V_lane_base_ptr, V_smem_offset_load, stride_seq_v, smem_V);
    cp_async::commit_group();

    K_load_idx_lane_base += CTA_K;
    V_load_idx_lane_base += CTA_K;
  }

  if (num_iterations > 1)
  {
    cp_async::wait_group<1>();
    __syncthreads();

    if constexpr (num_tiles_qk_inner == 1)
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK>(
        smem_K, RS, RQ, K_smem_offset_mma);
    }
    else
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK>(
        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }

    float RS_f32[num_tiles_q][num_tiles_k][8];
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]) * dequant_scale;
        }
      }
    }

    K_idx_lane_base += CTA_K;

    update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false>(RS_f32, RO, m, d, original_sm_scale);
    accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);

    uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
    RS_32_to_16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);

    __syncthreads();

    load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
      &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K, K_load_idx_lane_base, kv_len);
    cp_async::commit_group();

    k_scale = BroadcastScaleLoads ? load_k_scale_broadcast(K_scale, k_scale_idx + (num_iterations - 1) * k_scale_advance_offset, lane_id) : K_scale[k_scale_idx + (num_iterations - 1) * k_scale_advance_offset];
    dequant_scale = q_scale * k_scale;

    cp_async::wait_group<1>();
    __syncthreads();

    compute_fp16_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
      smem_V, RS_f16, RO, V_smem_offset_mma);

    __syncthreads();

    load_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
      &V_lane_base_ptr, V_smem_offset_load, stride_seq_v, smem_V, V_load_idx_lane_base, kv_len);
    cp_async::commit_group();

    K_load_idx_lane_base += CTA_K;
    V_load_idx_lane_base += CTA_K;
  }

  {
    cp_async::wait_group<1>();
    __syncthreads();

    if constexpr (num_tiles_qk_inner == 1)
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK>(
        smem_K, RS, RQ, K_smem_offset_mma);
    }
    else
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK>(
        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }

    float RS_f32[num_tiles_q][num_tiles_k][8];
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]) * dequant_scale;
        }
      }
    }

    apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(K_idx_lane_base, RS_f32, kv_len);
    K_idx_lane_base += CTA_K;

    update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false>(RS_f32, RO, m, d, original_sm_scale);
    accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);

    uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
    RS_32_to_16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);

    cp_async::wait_group<0>();
    __syncthreads();

    compute_fp16_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
      smem_V, RS_f16, RO, V_smem_offset_mma);

    __syncthreads();
  }
  normalize_d<num_tiles_q, num_tiles_v, ComputeUnit::kCudaCore>(RO, m, d);

  const uint32_t smem_O_row_base = get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      const uint32_t offset_O = smem_O.get_permuted_offset(smem_O_row_base + fq * MMA_QK_M, fv * (MMA_SV_N / PACK_SIZE_O));
      uint32_t RO_f16[4];
#pragma unroll
      for (uint32_t k = 0; k < 4; k++)
      {
        reinterpret_cast<half2*>(RO_f16)[k] = __float22half2_rn(reinterpret_cast<float2*>(RO[fq][fv])[k]);
      }

      (reinterpret_cast<uint32_t*>(smem_O.base + offset_O))[lane_id % 4] = RO_f16[0];
      (reinterpret_cast<uint32_t*>(smem_O.base + offset_O + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[1];
      (reinterpret_cast<uint32_t*>(smem_O.base + (offset_O ^ 0x1)))[lane_id % 4] = RO_f16[2];
      (reinterpret_cast<uint32_t*>(smem_O.base + (offset_O ^ 0x1) + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[3];
    }
  }

  __syncwarp();

  half *O_lane_ptr = O + batch_id * stride_bz_o + head_id * stride_h_o +
    (bx * CTA_Q + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>() + lane_id / global_to_shared_line_lanes_O) * stride_seq_o +
    lane_id % global_to_shared_line_lanes_O * PACK_SIZE_O;
  uint32_t offset_O = smem_O.get_permuted_offset(get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / global_to_shared_line_lanes_O, lane_id % global_to_shared_line_lanes_O);
  uint32_t O_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_O;

#pragma unroll
  for (uint32_t i = 0; i < O_smem_iters_col; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < O_smem_iters_row; j++)
    {
      if (O_load_idx_lane_base < qo_len)
      {
        smem_O.store_128b(offset_O, O_lane_ptr);
      }
      O_lane_ptr += (global_to_shared_line_lanes_O * PACK_SIZE_O);
      offset_O = smem_O.template advance_offset_by_column<global_to_shared_line_lanes_O>(offset_O);
    }

    offset_O = smem_O.template advance_offset_by_row<global_to_shared_copy_lines_per_warp_O>(offset_O - (O_smem_iters_row * global_to_shared_line_lanes_O));
    O_lane_ptr += ((global_to_shared_copy_lines_per_warp_O * stride_seq_o) - (O_smem_iters_row * global_to_shared_line_lanes_O * PACK_SIZE_O));
    O_load_idx_lane_base += global_to_shared_copy_lines_per_warp_O;
  }

  if constexpr (ReturnLse)
  {
    const uint32_t lse_idx = bx * CTA_Q + lane_id / 4 + 8 * (lane_id % 4) + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>();
    float *lse_lane_ptr = Lse + batch_id * (qo_len * num_qo_heads) + head_id * qo_len + lse_idx;
    const uint32_t fq = (lane_id % 4) / 2;
    const uint32_t k = (lane_id % 4) % 2;

    if (lse_idx < qo_len && (lane_id % 4) < 2 * num_tiles_q)
    {
      lse_lane_ptr[0] = math::ptx_log2(d[fq][k]) + m[fq][k];
    }
  }
}

} // namespace sageattention::qattn_cutlass

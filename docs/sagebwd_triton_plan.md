# SageBwd Triton Kernel Plan

## Goal

Add and tune a trainable Triton SageAttention path that follows the SageBwd design: use int8 tensor core matmuls for the backward matmuls that tolerate quantization, keep the numerically sensitive `dP = dO @ V^T` matmul in fp16/bf16 precision, and validate accuracy against FlashAttention backward.

## Current Implementation Status

- Non-causal attention only.
- Fixed-length dense tensors only.
- Canonical internal layout is `tensor_layout="NHD"`. The trainable wrapper accepts `NHD` and `HND` by converting to/from NHD.
- Trainable paths require `q`, `k`, and `v` to have the same shape. GQA/MQA is not implemented.
- Trainable paths support `torch.float16` only.
- Forward quantization products `q_int8`, `k_int8`, `q_scale`, `k_scale` are saved for backward.
- The default trainable backward is split into fused preprocess/`dO` quantization, `dQ`, and `dK/dV` kernels.
- The fused preprocess kernel computes `Delta` and pre-quantizes `dO` into `DOInt8`/`DOScale` once per Q block.
- Both default `dK/dV` and fused backward consume pre-quantized `DOInt8`/`DOScale` instead of quantizing `dO` inside KV-owned loops.
- The fused trainable backward uses a FlashAttention-style KV-block-owned kernel that computes `dQ`, `dK`, and `dV` from the same `P/dS` tile and accumulates `dQ` through fp32 workspace.
- `SAGEATTN_FUSED_BLOCK=64,128` can force the eager fused path to use FlashAttention's sm86 hdim64 tile for fixed-tile optimization and profiling without adding broad autotune candidates.
- Public APIs are `sageattn_qk_int8_pv_fp16_triton_trainable` and `sageattn_qk_int8_pv_fp16_triton_trainable_fused`. `SAGEATTN_BACKEND=triton_trainable_fused` selects the fused backend.
- Trainable eager and `torch.compile(fullgraph=True, mode="max-autotune")` paths autotune shared outer `(BLOCK_M, BLOCK_N)` by forward + backward total time.
- The Triton forward, backward preprocess, default `dQ`, default `dK/dV`, and fused kernels autotune their inner launch configs independently. Backward preprocess and fused `DQAccum` utility kernels tune `num_warps`. Default `dQ`, default `dK/dV`, and fused tune `num_warps` and `num_stages` with shared-memory pruning.

## Paper Facts

SageBwd backward matmuls are:

```text
S  = Q @ K^T
dV = P^T @ dO
dP = dO @ V^T
dQ = dS @ K
dK = dS^T @ Q
```

SageBwd keeps `dP = dO @ V^T` in fp16 precision and quantizes the other four backward matmuls with per-block int8.

The paper evaluates numerical accuracy with:

```text
CosSim = cosine_similarity(sage_tensor, reference_tensor)
Rel-L2 = ||sage_tensor - reference_tensor||_2 / ||reference_tensor||_2
```

Main reported intermediate accuracy versus full-precision attention:

| Tensor | CosSim | Rel-L2 |
|---|---:|---:|
| delta | 0.9973 | 0.0736 |
| P | 0.9917 | 0.1293 |
| dP | 1.0000 | 0.0000 |
| dS | 0.9789 | 0.2045 |
| O | 0.9969 | 0.0793 |
| dQ | 0.9664 | 0.2579 |
| dK | 0.9537 | 0.3074 |
| dV | 0.9985 | 0.0540 |

Reported speed goals:
- Backward-only: roughly `1.2x` to `1.6x` over FlashAttention in the earlier SageAttention3 paper appendix.
- Forward + backward kernel throughput: up to `1.67x` over FlashAttention2 on RTX4090.
- End-to-end training latency: about `1.15x` in their Llama runs.

## `pv_accum_dtype` Note

The paper does not map SageBwd to this repo's `pv_accum_dtype` option. In the paper, SageBwd forward uses int8 quantization for `P @ V`. This repo currently has `qk_int8_pv_fp16` forward with `pv_accum_dtype="fp32"` or `"fp16"`, and does not yet implement the fp8/int8 `P @ V` path known from SageAttention2++ or SageBwd.

For the current backward implementation:
- Use `pv_accum_dtype="fp32"` as the default forward setting for correctness tests, because it minimizes unrelated forward accumulation error and isolates the backward kernel.
- Keep test coverage for `pv_accum_dtype="fp16"` as a performance-relevant mode after the trainable autotune path is stable.
- Do not block the backward work on `pv_accum_dtype="fp8"`. It is not implemented in this repo and is outside the current scope.
- Accuracy numbers may differ from the paper until the forward path also matches SageBwd's int8 `P @ V` design.

## Kernel Architecture

### Forward

The trainable wrapper currently uses the Triton forward path:

```text
q, k -> per_block_int8 -> q_int8, q_scale, k_int8, k_scale
q_int8, k_int8, v, scales -> Triton forward -> out, lse
```

The forward pass saves the quantized tensors and scales for backward. Shared quantization currently uses `max(abs(x)) / 127.5 + 1e-7` with PTX round-to-nearest conversion and no extra clamp, because local accuracy checks showed this gives lower practical quantization error than clamped `127.0` variants.

### Backward Preprocess

The fused preprocess kernel computes the row-wise softmax-gradient scalar and per-Q-block `dO` quantization:

```text
delta = sum(out * dO, dim=-1)
DOInt8, DOScale = quantize_per_block(dO)
```

`Delta` has shape `[batch, heads, seqlen]` in canonical NHD-derived storage. `DOInt8` matches `dO`'s NHD shape, and `DOScale` has shape `[batch, heads, ceil(seqlen / BLOCK_M)]`.

### `dQ` Kernel

Grid over `(q_block, head, batch)`.

For each Q block, loop over all KV blocks:

```text
S_ij = int8(q_i) @ int8(k_j)^T * q_scale_i * k_scale_j * sm_scale
P_ij = exp(S_ij - lse_i)
dP_ij = dO_i @ V_j^T  # fp16/bf16 matmul, not int8
dS_ij = P_ij * (dP_ij - delta_i)
quantize dS_ij per block
dQ_i += int8(dS_ij) @ int8(k_j) * dS_scale * k_scale_j
```

If `smooth_k=True`, include the correction term:

```text
dQ_i += rowsum(dS_ij) * k_mean
```

### `dK/dV` Kernel

Grid over `(kv_block, head, batch)`.

Before launching this kernel, the fused preprocess quantizes `dO` once per Q block into `DOInt8` and `DOScale`.

For each K/V block, loop over all Q blocks:

```text
S_ij = int8(q_i) @ int8(k_j)^T * q_scale_i * k_scale_j * sm_scale
P_ij = exp(S_ij - lse_i)
quantize P_ij per block
load pre-quantized int8(dO_i), dO_scale_i
dV_j += int8(P_ij)^T @ int8(dO_i) * p_scale * dO_scale

dP_ij = dO_i @ V_j^T  # fp16/bf16 matmul, not int8
dS_ij = P_ij * (dP_ij - delta_i)
quantize dS_ij per block
dK_j += int8(dS_ij)^T @ int8(q_i) * dS_scale * q_scale_i
```

Pre-quantizing `dO` avoids quantizing the same `dO_i` tile once per KV block. For `seq_len=4096` and `BLOCK_N=64`, this removes roughly 64 repeated `dO` quantizations per Q tile. Fusing that quantization into preprocess removes the separate quantization launch and extra standalone `dO` read from the default backend.

This split keeps write ownership simple:
- `dQ` kernel owns one Q block, so no atomics for `dQ`.
- `dK/dV` kernel owns one KV block, so no atomics for `dK` or `dV`.
- `S`, `P`, `dP`, and `dS` are recomputed in both kernels. This remains the conservative default trainable backend.

### Fused Backward Kernel

The fused backend is a FlashAttention-style path with grid over `(kv_block, head, batch)`. Each KV-owned program loads one K/V tile and loops over Q blocks:

```text
S_ij = int8(q_i) @ int8(k_j)^T * q_scale_i * k_scale_j * sm_scale
P_ij = exp(S_ij - lse_i)
quantize P_ij per block
load pre-quantized int8(dO_i), dO_scale_i
dV_j += int8(P_ij)^T @ int8(dO_i) * p_scale * dO_scale

dP_ij = dO_i @ V_j^T  # fp16 matmul, not int8
dS_ij = P_ij * (dP_ij - delta_i)
quantize dS_ij per block
dK_j += int8(dS_ij)^T @ int8(q_i) * dS_scale * q_scale_i
dQ_i partial = int8(dS_ij) @ int8(k_j) * dS_scale * k_scale_j
```

`dK` and `dV` are owned by the KV program and stored directly. `dQ` is accumulated into fp32 `DQAccum` workspace and converted to fp16 by a final utility kernel.

`SAGEATTN_FUSED_DQ_SPLITS` controls optional split-plane `DQAccum` workspace:
- Default is `1`, matching the original single fp32 accumulation plane with `tl.atomic_add`. Nsight Compute reports these updates as global reduction operations.
- Values greater than `1` shard KV blocks across split planes, reducing per-plane reduction contention and adding a final split reduction.
- If the requested split count covers all KV blocks, the fused kernel uses stores instead of reductions for `dQ` partials.
- On the local sm86 validation machine, split counts greater than `1` were correct but slower for the benchmarked 2048/4096/8192 sequence lengths, so the default remains `1`.

## Local Profiling Findings

Latest profiling used Nsight Compute 2026.2 on the local sm86 machine with `batch=1`, `heads=16`, `head_dim=64`, `seq_len=4096`, non-causal fp16.

Representative public benchmark after standalone `dO` pre-quantization, before fused preprocess was added:

| Method | Forward ms | Backward ms | Total ms |
|---|---:|---:|---:|
| FlashAttention | 1.984 | 4.667 | 6.651 |
| Sage default | 1.303 | 6.035 | 7.338 |

Fixed-tile fused benchmark after the first fixed `(64,128)` optimization pass (`SAGEATTN_FUSED_BLOCK=64,128`, `SAGEATTN_FUSED_DQ_SPLITS=1`, `seq_len=4096`, `warmup=5`, `repeats=20`):

| Method | Forward ms | Backward ms | Total ms |
|---|---:|---:|---:|
| FlashAttention | 2.240 | 14.720 | 16.960 |
| Sage fused `(64,128)` | 1.275 | 7.872 | 9.147 |

Default-backend kernel replay NCU summary after standalone `dO` pre-quantization, before fused preprocess was added:

| Kernel | Time ms | Tensor core activity | Registers/thread | Notes |
|---|---:|---:|---:|---|
| FlashAttention main backward | 5.749 | HMMA ~45.7% | 255 | seq-k-parallel fused `dQ/dK/dV` kernel |
| Sage `dQ` | 2.401 | IMMA/HMMA ~21.9% | 255 | owns Q block, no atomics |
| Sage `dK/dV` | 4.907 | IMMA ~16.0%, HMMA ~10.7% | 235 | current dominant kernel |
| Sage fused preprocess + `dO` quant | not re-profiled | none | not re-profiled | computes `delta`, `DOInt8`, and `DOScale` in one launch |

Matched FlashAttention-vs-fused reduction profiling used the same shape and matched Sage fused to FlashAttention's observed main backward tile `(BLOCK_M=64, BLOCK_N=128)`. Nsight Compute reports the `dQ_accum` updates as global reduction operations, not global atom operations.

| Kernel | Main Kernel Time ms | Global Reduction Bytes | Global Reduction Requests | LTS Reduction-Active Cycles | Notes |
|---|---:|---:|---:|---:|---|
| FlashAttention main backward | 5.745 | 536.9 MB | 4.19M | 16.78M | `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel`, grid 512, block 256 |
| Sage fused `(64,128)`, split 1 | 16.720 | 2.147 GB | 4.19M | 67.11M | Uses global reductions for `DQAccum` |
| Sage fused `(64,128)`, split 32 | 15.743 | 0 | 0 | 0 | Store-only main-kernel path. Final split reduction still needed outside this kernel |
| Sage fused optimized `(64,128)`, split 1 | 10.233 | 2.147 GB | 4.19M | 67.11M | Same `DQAccum` RED volume as the earlier split-1 path. Speedup comes from scheduling/live-range changes |

Measured conclusions:
- Default split-backend profiling found `_bwd_dkdv_kernel` was the main default-backend bottleneck at `seq_len=4096`, and pre-quantizing `dO` moved it from roughly `5.65 ms` to `4.91 ms` in NCU kernel replay.
- The fused backend is now the primary optimization target because its fused KV-owned structure most closely matches FlashAttention backward: it computes `dQ`, `dK`, and `dV` from the same score/probability tiles instead of recomputing them in separate split kernels.
- Matched `(64,128)` profiling shows Sage fused has roughly `4x` FlashAttention's global reduction bytes/sectors and LTS reduction-active cycles for `DQAccum`, but a Sage split-1 vs split-32 A/B changes main-kernel time by only about `5.8%`, so reductions are not the only bottleneck.
- A high-level physical `DQAccum` layout-only experiment did not reduce global reduction bytes, requests, sectors, or LTS reduction-active cycles. The optimized fixed-tile fused kernel also keeps the same split-1 RED volume (`2.147 GB`, `4.19M` requests, `67.11M` sectors), so the measured speedup is from scheduling/live-range changes rather than less global reduction traffic.
- The next performance gap is structural: raise tensor-core utilization and reduce register/scheduling pressure in the fused kernel, not split `dK` and `dV` into separate kernels.

## Low-Level Source Notes

FlashAttention and Triton source review gives the following optimization constraints and opportunities:

- FlashAttention's hdim64 backward uses a KV-owned sequence-parallel kernel with separate CUTE MMA layouts for `S/dP`, `dK/dV`, and `dQ`. On large-smem GPUs it prefers `(BLOCK_M=128, BLOCK_N=128, num_warps=8)`. On sm86/sm89 it uses `(64,128,8)` with `V` in registers to fit shared memory.
- FlashAttention comments call out `(128,64)` as slow because it doubles `dQAccum` traffic, and use smaller `M` to reduce `LSE`/row-state register pressure. This matches the local observation that fused should prioritize `N >= M`, especially `(64,128)` for `HEAD_DIM=64`.
- FlashAttention explicitly double-buffers `Q/dO`, stages `K/V` with `cp.async`, writes `P/dS` through shared memory, and overlays shared-memory regions so `P`, `dS`, and `dQ` do not all occupy independent storage for the whole loop.
- Triton's int8 dot lowering on Ampere targets `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`, so the expected 2x int8-MMA arithmetic advantage is available if the generated schedule keeps IMMA fed and avoids excess fp32 live ranges/conversions.
- Triton exposes useful first-line scheduling controls in Python: `tl.range(num_stages=..., disallow_acc_multi_buffer=..., flatten=..., disable_licm=...)`, `num_warps`, `num_stages`, and `Config.maxnreg`. `warp_specialize` is not a near-term sm86 lever because current Triton documents it as Blackwell/simple-matmul oriented.
- Triton does not expose CUTE-style per-MMA atom layouts or warp-contiguous global copy layouts for arbitrary attention kernels. If Triton-level tuning stalls, likely compiler work is layout/scheduler exposure around DotI8, ldmatrix/stmatrix, and loop live-range control rather than another high-level tensor layout change.

## Optimization Roadmap

### Optimize Now

- Re-profile the current code first.
  - Measure total forward+backward and isolated backward for FlashAttention, Sage default, and Sage fused after fused preprocess/`dO` quantization and even-shape specialization.
  - Record `_bwd_fused_kernel` NCU counters for kernel time, registers/thread, spills, achieved occupancy, SM active, IMMA/HMMA active, issue-stall reasons, shared-memory load/store throughput, L2/DRAM traffic, and global reduction counters.
  - Use shapes `seq_len=2048/4096/8192`, `batch=1`, `heads=16`, `head_dim=64` as the primary loop, then confirm trends on `head_dim=128`.
- Use FlashAttention's sm86 hdim64 tile as the fixed optimization baseline.
  - Completed: eager benchmarking/profiling can force `(BLOCK_M=64, BLOCK_N=128)` with `SAGEATTN_FUSED_BLOCK=64,128` or `bench/bench_fwd_bwd.py --fused-block 64,128`.
  - Completed: focused correctness for the fixed `(64,128)` fused tile passes against FlashAttention backward.
  - Measured after fixed-tile plumbing plus first live-range/scheduling cleanup at `seq_len=4096`: FlashAttention public benchmark `fwd=2.241 ms`, `bwd=14.011 ms`, `total=16.252 ms`. Sage fused fixed `(64,128)`, `DQ_SPLITS=1`, `fwd=1.323 ms`, `bwd=15.406 ms`, `total=16.729 ms`.
  - Measured NCU for the current fixed `(64,128)` fused main kernel: `_bwd_fused_kernel=16.260 ms`, `registers/thread=255`, dynamic shared memory `87,564 B/block`, achieved occupancy `8.30%`, SM throughput `29.14%`, DRAM throughput `64.64%`, IMMA active `6.53%` with `16.78M` IMMA instructions, HMMA active `3.26%` with `8.39M` HMMA instructions.
  - Optimize and profile the fused kernel at `(BLOCK_M=64, BLOCK_N=128)` before changing the block search space.
  - Keep `DQ_SPLITS=1` for the first optimization loop so kernel-structure changes are not confounded with split-workspace utility costs.
  - Treat `(64,128)` as the source-of-truth comparison against FlashAttention's hdim64 sequence-parallel backward kernel on sm86/sm89.
- Reduce fused-kernel live ranges in Triton.
  - Rewrite the inner loop so `qk`, `p`, `dp`, and `ds` share storage/lifetime as aggressively as Triton allows: compute `P`, immediately quantize `P` for `dV`, then overwrite score/probability temporaries with `dP`/`dS` before quantizing `dS`.
  - Completed and measured: avoid one extra fp32 tile name by reusing the `dP` tile variable as `dS` instead of materializing a separate `ds` tensor. Fixed `(64,128)` correctness passes, but NCU still reports `255` registers/thread, so further structural pressure reduction is needed.
  - Completed and measured: the fused Q-block loop now uses `tl.range(..., disable_licm=True)` to prevent hoisted loop-invariant address/scale expressions from lengthening register live ranges. Current fixed-tile NCU still shows low occupancy and low tensor-core active percentages.
  - Completed and measured: moved `DOInt8` load to just before `dV` and fp16 `DO` load to just before `dP`, shortening `dO` live ranges. Fixed `(64,128)` correctness passes. Public benchmark initially improved from `bwd=13.791 ms`, `total=15.251 ms` to `bwd=13.417 ms`, `total=14.932 ms`. NCU `_bwd_fused_kernel=13.309 ms`, dynamic shared memory dropped from `81,920 B` to `49,152 B`, occupancy `15.95%`, SM throughput `35.85%`, DRAM throughput `87.04%`, IMMA active `7.93%`, HMMA active `3.96%`, registers remain `255/thread`.
  - Re-measured after reverting the rejected Q-reload experiment: fixed `(64,128)` correctness still passes, NCU remains low-smem with `_bwd_fused_kernel=14.051 ms`, `255` registers/thread, `49,152 B` dynamic shared memory, `16.24%` occupancy, `34.15%` SM throughput, `7.76%` IMMA active, `3.88%` HMMA active, but public benchmark showed `bwd=15.442 ms`, `total=16.787 ms`. Treat the earlier `13.417 ms` public result as a best observed value and continue measuring repeated latency after each change.
  - Rejected and reverted: reloading `Q` separately for `QK` and `dK` to shorten the Q tile live range passed correctness but regressed public fixed `(64,128)` benchmark to `bwd=15.164 ms`, `total=16.627 ms`. NCU `_bwd_fused_kernel=17.115 ms`, `255` registers/thread, dynamic shared memory increased to `69,632 B`, occupancy `16.67%`, SM throughput `48.50%`, IMMA active `9.30%`, HMMA active `6.20%`.
  - Rejected and reverted: reusing the `qk` variable as the probability tile passed correctness but regressed public fixed `(64,128)` benchmark to `bwd=15.932 ms`, `total=17.261 ms`. NCU was effectively unchanged from the restored state with `_bwd_fused_kernel=14.046 ms`, `255` registers/thread, `49,152 B` dynamic shared memory, `16.23%` occupancy, `34.03%` SM throughput, `7.75%` IMMA active, `3.88%` HMMA active.
  - Completed and measured: `tl.range(..., disallow_acc_multi_buffer=True)` passes fixed `(64,128)` correctness and improves public benchmark at `seq_len=4096`, `DQ_SPLITS=1`, fixed `(64,128)` from `bwd=14.396 ms`, `total=15.759 ms` to `bwd=13.791 ms`, `total=15.251 ms`. NCU remains effectively unchanged versus the 8-warp run: `_bwd_fused_kernel=16.694 ms`, `255` registers/thread, `81,920 B` dynamic shared memory, `16.86%` occupancy, `49.09%` SM throughput, `9.67%` IMMA active, `6.45%` HMMA active.
  - Completed and measured: compute `rowsum(dS)` immediately after forming the fp32 `dS` tile, before the `dK` and `dQ` int8 matmuls, so `dS` does not need to stay live across both tensor-core dots solely for the K-mean correction. Fixed `(64,128)` correctness passes. Public benchmark improved to `bwd=7.728 ms`, `total=9.047 ms`. NCU `_bwd_fused_kernel=10.199 ms`, `255` registers/thread, `61,956 B` dynamic shared memory, `8.36%` occupancy, `39.47%` SM throughput, DRAM throughput `7.15%`, IMMA active `10.35%`, HMMA active `5.17%`.
- Improve scheduling before changing algorithms.
  - Completed and measured: fused inner autotune now includes 8-warp configs while keeping the outer fixed tile `(64,128)`. Correctness passes. Public benchmark at `seq_len=4096`, `DQ_SPLITS=1`, fixed `(64,128)` improved from `bwd=15.406 ms`, `total=16.729 ms` to `bwd=14.396 ms`, `total=15.759 ms`.
  - 8-warp NCU kernel replay changed `_bwd_fused_kernel` from block size `128` to `256`, dynamic shared memory from `87,564 B` to `81,920 B`, occupancy from `8.30%` to `16.83%`, SM throughput from `29.14%` to `49.06%`, IMMA active from `6.53%` to `9.67%`, and HMMA active from `3.26%` to `6.45%`. Registers remain `255/thread`. Kernel-replay time was `16.723 ms` versus prior `16.260 ms`, so continue judging with public latency plus NCU resource counters.
  - Added `SAGEATTN_FUSED_INNER=warps,stages` as a narrow experiment/profiling override for fused inner configs without changing block-size autotune.
  - Measured forced `SAGEATTN_FUSED_INNER=8,2` at fixed `(64,128)`: correctness passes, but public benchmark regresses to `bwd=16.728 ms`, `total=18.107 ms`. NCU `_bwd_fused_kernel=18.253 ms`, `255` registers/thread, `82,436 B` dynamic shared memory, `16.61%` occupancy, `47.01%` SM throughput, `8.68%` IMMA active, `5.79%` HMMA active. Do not force `(8,2)`.
  - Measured forced `SAGEATTN_FUSED_INNER=8,1` at fixed `(64,128)`: correctness passes, but public benchmark `bwd=14.327 ms`, `total=15.659 ms` is slower than free autotune with `disallow_acc_multi_buffer=True`. NCU `_bwd_fused_kernel=16.693 ms`, `255` registers/thread, `81,920 B` dynamic shared memory, `16.74%` occupancy, `49.10%` SM throughput, `9.65%` IMMA active, `6.43%` HMMA active. Do not force `(8,1)` as a default.
  - Measured forced `SAGEATTN_FUSED_INNER=4,1` at fixed `(64,128)`: correctness passes, but public benchmark regresses to `bwd=18.555 ms`, `total=19.912 ms`. NCU `_bwd_fused_kernel=14.085 ms`, `255` registers/thread, `49,152 B` dynamic shared memory, `16.25%` occupancy, `33.99%` SM throughput, `7.76%` IMMA active, `3.88%` HMMA active. Do not force `(4,1)`.
  - Measured remaining forced 4-warp stages at fixed `(64,128)`: `(4,2)` correctness passes with public `bwd=13.701 ms`, `total=15.138 ms`, NCU `_bwd_fused_kernel=15.208 ms`, `255` registers/thread, `61,956 B` dynamic shared memory, `8.33%` occupancy. `(4,3)` correctness passes with public `bwd=14.188 ms`, `total=15.534 ms`. `(4,4)` correctness passes but regresses to public `bwd=18.359 ms`, `total=19.664 ms`.
  - Added a shape-specific preferred fused inner config `(num_warps=4, num_stages=2)` for fixed `(64,128, head_dim=64)` unless `SAGEATTN_FUSED_INNER` is set. Correctness passes. Longer public run still showed high variance/regression (`bwd=17.700 ms`, `total=19.014 ms`), with NCU confirming the expected `(4,2)` kernel: `_bwd_fused_kernel=15.236 ms`, `255` registers/thread, `61,956 B` dynamic shared memory, `8.36%` occupancy, `31.04%` SM throughput, `7.02%` IMMA active, `3.51%` HMMA active.
  - Retune or constrain `num_stages=1/2/3/4` at fixed `(64,128)` after deeper live-range changes. High stages can improve load overlap but also increase shared-memory footprint and register pressure in this fused loop.
  - Added `SAGEATTN_FUSED_MAXNREG` as a profiling-only hook that passes `Config.maxnreg` to fused inner configs.
  - Rejected as default: `SAGEATTN_FUSED_MAXNREG=224` at fixed `(64,128)` passes correctness and lowers reported `_bwd_fused_kernel` registers from `255` to `224`, but public benchmark regresses to `bwd=16.831 ms`, `total=18.286 ms`. NCU `_bwd_fused_kernel=17.037 ms`, `49,152 B` dynamic shared memory, `16.26%` occupancy, `30.28%` SM throughput, `6.37%` IMMA active, `3.19%` HMMA active. Keep maxnreg as an experiment knob only.
  - Inspect generated TTIR/PTX/SASS for the fixed `(64,128)` fused kernel to verify the int8 paths are actual IMMA instructions and to count extra `convert_layout`, local-memory spill, and fp32 conversion instructions.
- Increase effective int8-MMA share.
  - Keep `dP = dO @ V^T` as the only fp16/bf16 backward matmul. The four quantization-tolerant matmuls should stay on int8 MMA.
  - Profile the fraction of tensor-core cycles spent in IMMA versus HMMA. If HMMA `dP` or pointwise/quantization work dominates, optimize the schedule and tile shape before considering new kernel splits.
  - Check whether `tl.dot(..., acc=...)` or accumulator ordering can reduce temporary dot results for `dK`, `dV`, or `dQ`. Per-tile scale factors still require care because they vary by Q/KV block.
- Keep the default split backend conservative.
  - Use the default `dQ` plus `dK/dV` split backend for correctness and as a no-`DQAccum` baseline.
  - Do not prioritize separate `dV-only` and `dK-only` kernels now. FlashAttention's fastest path is fused, and the local gap is scheduling/register/tensor-core utilization rather than lack of more split kernels.

### Revisit After Optimization

- Revisit `DQAccum` only after fused-kernel scheduling improves.
  - Current split-plane workspace and high-level layout experiments did not improve total behavior enough on sm86.
  - Any future `DQAccum` work must report total backward time, including zero/convert/reduce utilities, not only `_bwd_fused_kernel` time.
  - Matching FlashAttention's reduction behavior likely needs per-lane/write mapping or CUDA/CUTE-style global copy layout control, not just a different high-level tensor shape.
- Consider inline asm only for specific generated-code gaps.
  - Keep the existing inline PTX int8 rounding helper.
  - Add MMA or copy inline asm only if SASS proves Triton emits inefficient IMMA operand packing, extra layout conversions, or unvectorized memory operations that cannot be fixed with Triton kernel structure/configs.
  - Treat inline `mma.sync.aligned.m16n8k32` as high-risk because it requires matching Triton's lane/register layout. Prefer compiler hooks if the issue is layout selection.
- Consider Triton compiler patches after Triton-kernel experiments.
  - Candidate patches are exposing/controlling DotI8 operand/result layouts, improving ldmatrix/stmatrix selection for int8 tiles, reducing loop live ranges around multiple dots plus pointwise work, and adding scheduler hints closer to FlashAttention's explicit CUTE copy/MMA choreography.
  - Avoid compiler work for `warp_specialize` on the current sm86 target. It is not the likely near-term path for this kernel.
- Defer autotune and larger algorithmic changes.
  - Do not add broad fused block-size autotune candidates until the fixed `(64,128)` fused kernel has lower register pressure and better tensor-core utilization.
  - After fixed-tile optimization, consider adding fused-specific candidates such as `(64,128)` and `(128,128)` to the autotune space. Avoid `(128,64)` unless profiling contradicts FlashAttention's `dQAccum`-traffic warning.
  - Keep q/k quantization scale compatibility in mind: changing backward tile sizes independently from forward requires either shared block-size selection or extra q/k quantization/scales for backward.
  - Do not revisit separate `dK`/`dV` kernels, alternative `DQAccum` split policies, or q/k re-quantization with independent backward tile sizes until the fused kernel has been retuned and re-profiled.
  - Defer bf16, causal, GQA/MQA, varlen, and `head_dim=256` expansion until the non-causal fp16 fused path is closer to FlashAttention on hdim64/128.

## Benchmark and Profiling Workflow

- Use `bench/bench_fwd_bwd.py` for public latency comparisons across `sdpa`, `flash`, `sage`, and `sage_fused`.
  - Primary target shape: `--batch_size 1 --num_heads 16 --head_dim 64 --seq_lens 2048 4096 8192`.
  - Use `--fused-dq-splits` to benchmark split-plane `DQAccum` choices.
- Use Nsight Compute for kernel-level comparisons.
  - Profile after warmup and isolate backward with `cudaProfilerStart`/`cudaProfilerStop`.
  - Prefer kernel replay for Triton/autotuned workloads when application replay sees inconsistent kernels.
  - Track at least kernel time, registers/thread, shared memory, achieved occupancy, DRAM/L2 throughput, HMMA activity, IMMA activity, and global reduction counters for fused/FlashAttention comparisons.
- Expand support only after the non-causal fp16 path is stable.
  - `head_dim=128` and `head_dim=256` correctness/performance sweeps beyond the current autotune validity coverage.
  - bf16.
  - GQA/MQA.
  - Causal mode.
  - Varlen.

## Risks And Open Questions

- int8 quantization granularity for `P` in backward currently follows the paper's Algorithm 3 per-block description. Accuracy may still require experimenting with per-row/per-token scaling.
- `dS` is expected to be the fragile tensor. Debug tooling should expose `dS` metrics if accuracy falls below target.
- Long sequence performance may need additional scheduling work beyond the current split-kernel and fused implementations.
- Fixed-tile fused now beats the local FlashAttention public benchmark at `seq_len=4096` in the latest run, but NCU still shows `_bwd_fused_kernel` at `255` registers/thread and low occupancy, so the remaining optimization target is register pressure and more efficient tensor-core scheduling.
- A single shared trainable block size is the right current design, but it means the chosen config is a compromise across quantization, forward attention, and backward kernels.
- Fused `dQ` accumulation can be sharded to reduce main-kernel reduction traffic/contention, but current sm86 benchmarks show the extra zero/reduce workspace cost outweighs the benefit for the tested shapes, so larger `DQAccum` work is deferred.

## Validation Snapshot

- Correctness uses FlashAttention backward as the reference and flattened cosine similarity plus Frobenius relative L2 metrics.
- Configured fused and forced split-plane `DQAccum` tests pass.
- Latest full validation: `python -m pytest -n 8 tests/ -q` reported `1069 passed`. `pre-commit run --all-files` passed.
- After default/fused `dO` pre-quantization and even-shape specialization, focused validation passed: `python -m pytest tests/test_sagebwd_triton.py tests/test_sagebwd_triton_compile.py tests/test_sagebwd_triton_fused.py tests/test_sagebwd_triton_fused_compile.py -q` reported `17 passed`. `pre-commit run --files sageattention/triton/attn_bwd_qk_int8.py sageattention/triton/attn_bwd_qk_int8_fused.py` passed.

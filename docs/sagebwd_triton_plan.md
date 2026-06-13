# SageBwd Triton Kernel Plan

## Goal

Add and tune a trainable Triton SageAttention path that follows the SageBwd design: use int8 tensor core matmuls for the backward matmuls that tolerate quantization, keep the numerically sensitive `dP = dO @ V^T` matmul in fp16/bf16 precision, and validate accuracy against FlashAttention backward.

## Current Scope and Implementation

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
- `SAGEATTN_FUSED_BLOCK=32,128` forces the best backward-only fixed block for the local sm86 hdim64/seq4096 measurements. `SAGEATTN_FUSED_BLOCK=64,128` remains useful as the FlashAttention-style sm86 tile baseline.
- Public APIs are `sageattn_qk_int8_pv_fp16_triton_trainable` and `sageattn_qk_int8_pv_fp16_triton_trainable_fused`. `SAGEATTN_BACKEND=triton_trainable_fused` selects the fused backend.
- Trainable eager and `torch.compile(fullgraph=True, mode="max-autotune")` paths autotune one shared outer `(BLOCK_M, BLOCK_N)` by forward + backward total time. The shared candidate list includes the original forward-oriented blocks plus fused-oriented `(32,64)`, `(32,128)`, `(64,128)`, and `(128,128)`. Future work may tune forward and backward separately and decouple quantization block size from matmul tile size, but optimization is backward-only.
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

The paper does not map SageBwd to this repo's `pv_accum_dtype` option. In the paper, SageBwd forward uses int8 quantization for `P @ V`. This repo has `qk_int8_pv_fp16` forward with `pv_accum_dtype="fp32"` or `"fp16"`, and does not yet implement the fp8/int8 `P @ V` path known from SageAttention2++ or SageBwd.

For the backward implementation:
- Use `pv_accum_dtype="fp32"` as the default forward setting for correctness tests, because it minimizes unrelated forward accumulation error and isolates the backward kernel.
- Keep test coverage for `pv_accum_dtype="fp16"` as a performance-relevant mode after the trainable autotune path is stable.
- Do not block the backward work on `pv_accum_dtype="fp8"`. It is not implemented in this repo and is outside the scope.
- Accuracy numbers may differ from the paper until the forward path also matches SageBwd's int8 `P @ V` design.

## Kernel Architecture

### Forward

The trainable wrapper uses the Triton forward path:

```text
q, k -> per_block_int8 -> q_int8, q_scale, k_int8, k_scale
q_int8, k_int8, v, scales -> Triton forward -> out, lse
```

The forward pass saves the quantized tensors and scales for backward. Shared quantization uses `max(abs(x)) / 127.5 + 1e-7` with PTX round-to-nearest conversion and no extra clamp, because local accuracy checks showed this gives lower practical quantization error than clamped `127.0` variants.

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

### Asymptotic MMA Cost Model

Assume one full attention-sized fp16 tensor-core matmul costs `G`, int8 tensor-core matmul has exactly 2x fp16 throughput, and non-MMA overheads vanish as sequence length grows. FlashAttention forward performs `QK` and `PV`, so use `2G` as the ratio baseline.

| Path | Dominant MMA work | Ideal time | Ratio to FlashAttention forward |
|---|---:|---:|---:|
| FlashAttention forward | `QK fp16 + PV fp16` | `2G` | `1.00x` |
| FlashAttention backward | `QK recompute + dP + dV + dQ + dK`, all fp16 | `5G` | `2.50x` |
| SageAttention forward | `QK int8 + PV fp16` | `1.5G` | `0.75x` |
| SageAttention fused backward | `QK int8 + dV int8 + dP fp16 + dK int8 + dQ int8` | `3G` | `1.50x` |

This model predicts SageAttention fused backward should asymptotically take `3G / 5G = 0.60x` FlashAttention backward, or be about `1.67x` faster, if the implementation keeps tensor cores saturated and avoids non-MMA bottlenecks. If the repo later adds an int8/low-precision `PV` forward path, SageAttention forward's ideal ratio would drop from `0.75x` to `0.50x`.

## Local Profiling Findings

Nsight Compute 2026.2 measurements below were taken on the local sm86 machine with `batch=1`, `heads=16`, `head_dim=64`, `seq_len=4096`, non-causal fp16 unless noted.

Public backward-only comparison for the local target shape (`warmup=8`, `repeats=30`, `SAGEATTN_FUSED_BLOCK=32,128`, `SAGEATTN_FUSED_DQ_SPLITS=1` for Sage):

| Method | Forward ms | Backward ms | Total ms | Backward vs Flash |
|---|---:|---:|---:|---:|
| FlashAttention | 2.221 | 4.972 | 7.193 | `1.00x` |
| Sage fused `(32,128)` | 1.967 | 5.377 | 7.344 | `1.081x` slower |

Fused autotune snapshot after adding fused-oriented block candidates (`SAGEATTN_FUSED_DQ_SPLITS=1`, `warmup=8`, `repeats=20` for hdim64):

| Shape | Selected block | Forward ms | Backward ms | Total ms |
|---|---:|---:|---:|---:|
| hdim64, seq4096 | `(32,128)` | 1.842 | 5.857 | 7.698 |
| hdim128, seq4096 | `(32,64)` | 3.001 | 13.813 | 16.814 |
| hdim256, seq4096 | `(32,64)` | 7.758 | 33.728 | 41.486 |

Backward-only PyTorch-profiler snapshot with forward computed outside the timed region (`SAGEATTN_FUSED_BLOCK=32,128`, `SAGEATTN_FUSED_DQ_SPLITS=1`):

| Method | Event-timed bwd mean | Profiler CUDA total | Main backward kernel | Utility kernels |
|---|---:|---:|---:|---:|
| Sage fused `(32,128)` | 5.438 ms | 4.495 ms | `_bwd_fused_kernel=4.286 ms` | zero 50 us, preprocess 73 us, convert 86 us |
| FlashAttention | 4.673 ms | 4.462 ms | `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel=4.262 ms` | dot-do-o 114 us, convert 86 us |

NCU replay snapshot for fixed `(32,128)` Sage fused versus FlashAttention (`batch=1`, `heads=16`, `head_dim=64`, `seq_len=4096`):

| Kernel | Time ms | Registers/thread | Dynamic smem | SM throughput | Active warps | DRAM throughput | RED requests | RED sectors | L2 read sectors | L2 write sectors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sage `_bwd_fused_kernel` `(32,128)` | 6.433 | 255 | 49.412 KiB | 65.18% | 15.85% | 3.56% | 4.19M | 16.78M | 17.70M | 0.52M |
| FlashAttention main backward | 5.748 | 255 | 73.728 KiB | 44.73% | 16.66% | 4.20% | 4.19M | 16.78M | 17.83M | 0.52M |
| Sage zero/preprocess/convert total | 0.148 | 16-40 | <=0.128 KiB | 4.95-20.14% | 71.44-89.53% | ~90% | 0 | 0 | ~1.05M | ~0.93M |
| Flash dot-do-o/convert total | 0.141 | 34-42 | <=8.192 KiB | 9.95-15.75% | 75.51-85.33% | ~90% | 0 | 0 | ~1.05M | ~0.80M |

The exact profiler and NCU timings vary by tool/replay mode, but both show the same direction: utility kernels are small, `DQAccum` RED traffic is similar to FlashAttention at this shape, and the main Sage fused loop is the remaining bottleneck.

Earlier fixed-tile fused snapshot after the first `(64,128)` optimization pass (`SAGEATTN_FUSED_BLOCK=64,128`, `SAGEATTN_FUSED_DQ_SPLITS=1`, `warmup=5`, `repeats=20`):

| Method | Forward ms | Backward ms | Total ms |
|---|---:|---:|---:|
| FlashAttention | 2.240 | 14.720 | 16.960 |
| Sage fused `(64,128)` | 1.275 | 7.872 | 9.147 |

Earlier optimized fixed-tile `(64,128)` fused-kernel NCU snapshot:

| Kernel | Time ms | Registers/thread | Dynamic smem | Occupancy | SM throughput | DRAM throughput | IMMA active | HMMA active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `_bwd_fused_kernel` `(64,128)` | 10.199 | 255 | 61,956 B | 8.36% | 39.47% | 7.15% | 10.35% | 5.17% |

Reduction-traffic comparison for the matched FlashAttention tile `(BLOCK_M=64, BLOCK_N=128)`:

| Kernel | Main Kernel Time ms | Global Reduction Bytes | Global Reduction Requests | LTS Reduction-Active Cycles | Notes |
|---|---:|---:|---:|---:|---|
| FlashAttention main backward | 5.745 | 536.9 MB | 4.19M | 16.78M | `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel`, grid 512, block 256 |
| Sage fused `(64,128)`, split 1 baseline | 16.720 | 2.147 GB | 4.19M | 67.11M | Uses global reductions for `DQAccum` |
| Sage fused `(64,128)`, split 32 | 15.743 | 0 | 0 | 0 | Store-only main-kernel path. Final split reduction needed outside this kernel |
| Sage fused optimized `(64,128)`, split 1 | 10.233 | 2.147 GB | 4.19M | 67.11M | Same RED volume as baseline. Speedup came from scheduling/live-range changes |

Useful historical checkpoints:

| Checkpoint | Observation |
|---|---|
| Default split backend before fused preprocess | `_bwd_dkdv_kernel` was the main bottleneck at `seq_len=4096`. Standalone `dO` pre-quantization moved NCU replay from roughly `5.65 ms` to `4.91 ms` |
| Standalone `dO` pre-quantization public benchmark | FlashAttention `fwd=1.984 ms`, `bwd=4.667 ms`, `total=6.651 ms`. Sage default `fwd=1.303 ms`, `bwd=6.035 ms`, `total=7.338 ms` |
| First fixed fused baseline | FlashAttention `fwd=2.241 ms`, `bwd=14.011 ms`, `total=16.252 ms`. Sage fused `(64,128)` `fwd=1.323 ms`, `bwd=15.406 ms`, `total=16.729 ms`. NCU `_bwd_fused_kernel=16.260 ms`, `255` registers/thread, `87,564 B` dynamic smem, `8.30%` occupancy |

Measured conclusions:
- Fused is the primary optimization target because its KV-owned structure matches FlashAttention backward: one score/probability tile feeds `dQ`, `dK`, and `dV` instead of recomputing tiles in split kernels.
- Best fixed backward block on the local sm86 hdim64/seq4096 target is `(32,128)`, not the FlashAttention-style `(64,128)` tile. With this block, public backward is about `8.1%` slower than FlashAttention (`5.377 ms` vs `4.972 ms`).
- The main fused kernel is the bottleneck. Utility kernels are roughly the same total size as FlashAttention's utility kernels, and split-1 `DQAccum` RED requests/sectors match FlashAttention for fixed `(32,128)` in the NCU run.
- The main issue is not raw algorithmic work or DRAM bandwidth. Sage fused does 4 of 5 backward matmuls with int8 MMA but has a slower main-kernel replay than FlashAttention, pointing to register pressure, live ranges, layout conversions, tensor-core scheduling, and lack of explicit CUTE-style copy/MMA layout control.
- Split-plane `DQAccum` and high-level physical layout changes did not improve total behavior enough on sm86. Reduction work is real, but profiler data says it is not the first bottleneck to attack.

## Low-Level Source Notes

FlashAttention and Triton source review gives the following optimization constraints and opportunities:

- FlashAttention's hdim64 backward uses a KV-owned sequence-parallel kernel with separate CUTE MMA layouts for `S/dP`, `dK/dV`, and `dQ`. On large-smem GPUs it prefers `(BLOCK_M=128, BLOCK_N=128, num_warps=8)`. On sm86/sm89 it uses `(64,128,8)` with `V` in registers to fit shared memory.
- FlashAttention comments call out `(128,64)` as slow because it doubles `dQAccum` traffic, and use smaller `M` to reduce `LSE`/row-state register pressure. Local bwd-only measurements favor `(32,128)` for hdim64/seq4096, while `(64,128)` remains the matched FlashAttention-style tile.
- FlashAttention explicitly double-buffers `Q/dO`, stages `K/V` with `cp.async`, writes `P/dS` through shared memory, and overlays shared-memory regions so `P`, `dS`, and `dQ` do not all occupy independent storage for the whole loop.
- Triton's int8 dot lowering on Ampere targets `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`, so the expected 2x int8-MMA arithmetic advantage is available if the generated schedule keeps IMMA fed and avoids excess fp32 live ranges/conversions.
- Triton exposes useful first-line scheduling controls in Python: `tl.range(num_stages=..., disallow_acc_multi_buffer=..., flatten=..., disable_licm=...)`, `num_warps`, and `num_stages`. `warp_specialize` is not a near-term sm86 lever because Triton documents it as Blackwell/simple-matmul oriented.
- Triton does not expose CUTE-style per-MMA atom layouts or warp-contiguous global copy layouts for arbitrary attention kernels. If Triton-level tuning stalls, likely compiler work is layout/scheduler exposure around DotI8, ldmatrix/stmatrix, and loop live-range control rather than another high-level tensor layout change.
- AITER's CK-style FMHA backward generators similarly expose explicit tile/pipeline traits for dot-do-o, convert-dQ, and dq/dk/dv paths. This reinforces that high-performance attention backward implementations tend to control per-phase tile layout and pipeline structure below the abstraction level available in this Triton kernel.

## Optimization Roadmap

### Experiments Done

- Fixed fused tile control and autotune expansion.
  - `SAGEATTN_FUSED_BLOCK=64,128` forces the eager fused path to FlashAttention's sm86 hdim64 tile.
  - `SAGEATTN_FUSED_BLOCK=32,128` forces the best local bwd-only hdim64/seq4096 tile.
  - Focused correctness for `(64,128)` passes against FlashAttention backward.
  - Added fused-oriented shared outer block candidates `(32,64)`, `(32,128)`, `(64,128)`, and `(128,128)`. Correctness passed without loosening tolerances.
  - Sequential hdim64 fwd+bwd benchmark with the expanded list selected `(32,128)` and measured `fwd=1.842 ms`, `bwd=5.857 ms`, `total=7.698 ms`, faster than fixed `(64,128)` (`total=8.819 ms`) and fixed `(128,128)` (`total=8.772 ms`) in the same run.
  - hdim128 and hdim256 default fused autotune selected `(32,64)`, matching the best fixed candidates tested for those head dims.
  - Keep `SAGEATTN_FUSED_DQ_SPLITS=1` for fixed-tile optimization unless specifically measuring split-workspace behavior.
- Fused preprocess and `dO` pre-quantization.
  - Fused preprocess computes `Delta`, `DOInt8`, and `DOScale` once per Q block.
  - This removed repeated `dO` quantization inside KV-owned loops and improved the default split `dK/dV` replay from roughly `5.65 ms` to `4.91 ms` at `seq_len=4096`.
- Fused-kernel live-range/scheduling changes.
  - `tl.range(..., disable_licm=True)` prevents hoisted loop-invariant address/scale expressions from lengthening live ranges and remains useful after the later live-range changes. Retest at forced `(4,2)` fixed `(64,128)` showed removing it regressed `bwd=8.105 ms` to `8.928 ms`.
  - `tl.range(..., disallow_acc_multi_buffer=True)` was useful earlier (`bwd=14.396 ms` to `13.791 ms`) but was retested after later live-range changes and removed. At forced `(4,2)` fixed `(64,128)`, keeping only `disable_licm=True` measured `bwd=8.029 ms` versus `8.105 ms` with both hints.
  - Moving `DOInt8` load to just before `dV` and fp16 `DO` load to just before `dP` shortened `dO` live ranges. Best observed public result after this change was `bwd=13.417 ms`, `total=14.932 ms`. NCU dropped dynamic smem to `49,152 B` with `_bwd_fused_kernel=13.309 ms`, but repeated public runs showed variance.
  - Computing `rowsum(dS)` immediately after forming fp32 `dS`, before the `dK` and `dQ` int8 matmuls, was the strongest win. Public benchmark improved to `bwd=7.728 ms`, `total=9.047 ms`. NCU `_bwd_fused_kernel=10.199 ms`, `61,956 B` dynamic smem, `39.47%` SM throughput, `10.35%` IMMA active, `5.17%` HMMA active.
- Rejected live-range experiments.
  - Reloading `Q` separately for `QK` and `dK` passed correctness but regressed public fixed `(64,128)` to `bwd=15.164 ms`, `total=16.627 ms`. NCU `_bwd_fused_kernel=17.115 ms`, dynamic smem `69,632 B`.
  - Reusing the `qk` variable as the probability tile passed correctness but regressed public fixed `(64,128)` to `bwd=15.932 ms`, `total=17.261 ms`. NCU was effectively unchanged from the restored state.
- Inner-config scheduling experiments.
  - Adding 8-warp candidates improved one early public `(64,128)` run from `bwd=15.406 ms`, `total=16.729 ms` to `bwd=14.396 ms`, `total=15.759 ms`, but NCU replay did not improve main-kernel time and free selection was noisy.
  - `SAGEATTN_FUSED_INNER=warps,stages` is available for forced experiments. For fixed `(32,128)`, a sequential forced sweep measured `(4,2)` best (`bwd=5.238 ms`, `total=7.018 ms`), `(4,1)` close (`bwd=5.365 ms`), and all 8-warp variants slower (`bwd=8.001-8.800 ms`).
  - No implicit fixed `(4,2)` preference is kept. By default the fused kernel exposes all valid inner configs to Triton autotune, and `SAGEATTN_FUSED_INNER` is the explicit override path.
  - A temporary `maxnreg=224` cap lowered reported registers from `255` to `224` but regressed public runtime to `bwd=16.831 ms`, `total=18.286 ms`. No max-register runtime override is kept.
- `DQAccum` experiments.
  - Split-plane `DQAccum` with split count covering all KV blocks removes main-kernel RED traffic, but total behavior was slower once zero/convert/reduce utility costs are considered.
  - A high-level physical `DQAccum` layout experiment did not reduce global RED bytes, requests, sectors, or LTS reduction-active cycles and was reverted.
  - Optimized split-1 fused has the same `DQAccum` RED volume as the baseline, confirming the speedup came from scheduling/live-range changes.

### Next Steps

- Focus optimization on backward-only fixed-block measurements.
  - Use `SAGEATTN_FUSED_BLOCK=32,128`, `SAGEATTN_FUSED_INNER=4,2`, and `SAGEATTN_FUSED_DQ_SPLITS=1` for the local hdim64/seq4096 bwd-only optimization loop unless a change specifically targets tile selection.
  - Keep `SAGEATTN_FUSED_BLOCK=64,128` as the matched FlashAttention-style tile baseline, not as the fastest local bwd-only tile.
  - Compare each change with public backward latency and NCU kernel replay because public benchmark, profiler, and replay can disagree under autotune/cache/driver variance.
  - Primary shapes remain `seq_len=2048/4096/8192`, `batch=1`, `heads=16`, `head_dim=64`. Confirm promising changes on `head_dim=128`.
- Reduce `_bwd_fused_kernel` register pressure and layout pressure.
  - Best fixed `(32,128)` main kernel reports `255` registers/thread and about `16%` active-warps occupancy, so useful changes should shorten fp32 tile/scalar live ranges or reduce layout-conversion pressure.
  - Inspect TTIR/PTX/SASS for local-memory spills, extra `convert_layout`, redundant fp32 conversions, and whether int8 paths are clean IMMA instructions.
  - Revisit `tl.dot(..., acc=...)`, accumulator update ordering, and scale application placement if they reduce temporary dot results without changing per-tile scale semantics.
- Improve tensor-core scheduling and memory staging at the same fixed tile.
  - Try `.cg` cache modifiers on streaming global loads for Q/K/V/dO, matching FlashAttention's cache-global strategy.
  - Retune `num_warps`/`num_stages` after structural changes using `SAGEATTN_FUSED_INNER` for forced measurements, but do not assume 8 warps helps. Fixed `(32,128)` favors `(4,2)`.
  - Track IMMA/HMMA activity separately. `dP = dO @ V^T` should remain the only fp16/bf16 matmul. The other four backward matmuls should stay on int8 MMA unless a deliberately mixed-precision experiment such as fp16 `dV` is being tested.
  - Prototype fp16 `dV` only as a register/scheduling experiment: it loses one int8 MMA in the theoretical model, but may reduce `P` quantization and int8 scale live ranges enough to be informative on sm86.
- Keep the default split backend conservative.
  - Use default `dQ` plus `dK/dV` as a correctness and no-`DQAccum` baseline.
  - Do not prioritize separate `dV-only` and `dK-only` kernels. FlashAttention's fastest path is fused, and the gap is scheduling/register/tensor-core utilization.

### Revisit After Optimization

- `DQAccum` reduction work.
  - Revisit split-plane workspace, physical layout, or per-lane/write mapping only after `_bwd_fused_kernel` scheduling improves.
  - Any future `DQAccum` experiment must report total backward time, including zero/convert/reduce utilities, not only main-kernel time.
  - Matching FlashAttention's reduction behavior likely needs CUDA/CUTE-style global copy layout control rather than only a different high-level tensor shape.
- Separate forward/backward and quantization/matmul tuning.
  - The shared trainable block size is a compromise across forward matmul, q/k scale generation, and backward matmul. The best fwd+bwd autotune choice may not be the best forward-only or backward-only choice.
  - Later work may tune forward and backward block sizes separately and decouple q/k quantization scale block size from matmul tile size. That requires extra scale/quantization plumbing or independent saved tensors.
  - Avoid adding very large `N` candidates such as `(64,256)`, `(128,256)`, and `(256,128)` by default. `(64,256)` was valid for hdim64 but much slower in the fixed-block benchmark.
  - Avoid adding `(128,64)` solely for fused performance. It remains in the original shared list and is useful for forward, but fixed fused hdim64 was slower than `(32,128)`, `(64,128)`, and `(128,128)`.
- Lower-level implementation paths.
  - Keep the existing inline PTX int8 rounding helper.
  - Consider inline MMA/copy asm only if SASS proves Triton emits inefficient IMMA operand packing, extra layout conversions, or unvectorized memory operations that kernel-structure changes cannot fix.
  - Consider Triton compiler patches only after Triton-level experiments, likely around DotI8 operand/result layouts, ldmatrix/stmatrix selection, loop live ranges, and scheduler hints.
- Feature expansion.
  - Defer bf16, causal, GQA/MQA, varlen, and `head_dim=256` until the non-causal fp16 fused path is stable on hdim64/128.

## Benchmark and Profiling Workflow

- Backward-only public latency: run FlashAttention and Sage sequentially with precomputed forward outside the timed backward region when answering bwd-only questions. For the quick end-to-end benchmark, set `SAGEATTN_FUSED_DQ_SPLITS=1`, `SAGEATTN_FUSED_BLOCK=32,128`, and `SAGEATTN_FUSED_INNER=4,2`, then run `python bench/bench_fwd_bwd.py --method sage_fused --batch_size 1 --num_heads 16 --head_dim 64 --seq_lens 4096`.
- Fwd+bwd autotune check: unset `SAGEATTN_FUSED_BLOCK` and `SAGEATTN_FUSED_INNER`, keep `SAGEATTN_FUSED_DQ_SPLITS=1`, and run `python bench/bench_fwd_bwd.py --method sage_fused --batch_size 1 --num_heads 16 --head_dim 64 --seq_lens 2048 4096 8192` only when checking end-to-end behavior.
- FlashAttention comparison: run the same shape with `--method flash` and compare backward time separately from total time.
- NCU kernel replay: profile backward after warmup, isolate with `cudaProfilerStart`/`cudaProfilerStop`, and track kernel time, registers/thread, dynamic shared memory, achieved occupancy/active warps, SM throughput, DRAM/L2 throughput, IMMA/HMMA activity, spills/local memory, and global reduction counters.
- Proton/PyTorch profiler: use when kernel-level launch grouping or eager inter-kernel gaps matter. Proton can add scoped timing around backward regions, while PyTorch profiler gives a quick CUDA-time split for utility kernels versus `_bwd_fused_kernel`.
- Measurement rule: after each performance edit, run focused correctness first, then bwd-only public latency, then NCU if the change is kept or ambiguous. Re-run fwd+bwd autotune only when tile selection or forward-facing behavior changes.

## Risks And Open Questions

- `P` and `dS` quantization follow the paper's per-block description. `dS` remains the fragile tensor for accuracy.
- Best fixed `(32,128)` fused backward is close to FlashAttention but slower on public bwd-only latency and slower under NCU main-kernel replay despite the theoretical int8-MMA advantage.
- `_bwd_fused_kernel` reports `255` registers/thread and low active-warps occupancy. The main risk is that Triton-level scheduling and layout controls are insufficient to close the gap without compiler work or a lower-level CUDA/CUTE implementation.
- Public benchmark timing, PyTorch/Proton profiler timing, and NCU kernel replay can diverge because of autotune/cache/driver/replay variance. Keep all retained measurements in the log and compare directions rather than single samples.
- A single shared trainable block size is expedient but is a compromise across quantization, forward attention, and backward kernels. Future separate fwd/bwd tuning and quantization/matmul decoupling may be needed for best end-to-end speed.
- `DQAccum` sharding can reduce main-kernel reduction traffic, but sm86 measurements show the extra workspace utilities outweigh the benefit for tested shapes.

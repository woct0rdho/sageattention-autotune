# QAttn CUTLASS Backward Kernel Plan

## Scope

- SM80-family only, matching the CUTLASS forward extension.
- fp16 `q`, `k`, `v`, `out`, and `dout`.
- Non-causal fixed-length dense tensors only.
- `HND` and `NHD` layouts.
- No GQA/MQA initially: `q`, `k`, and `v` must have matching shape.
- Head dimensions `64` and `128`.
- Standalone API only. No trainable wrapper, no compile custom op, and no backward autotune.
- A small static set of CUTLASS backward block/warp variants is allowed, chosen from FlashAttention backward heuristics and the documented SageBwd Triton fixed-block results.
- Correctness reference is FlashAttention backward.

## Current Implementation

The fused loop follows the SageBwd split:

```text
preprocess:
  Delta_i = sum(out_i * dout_i)
  zero dQ accumulation workspace
for each KV tile:
  for each Q/dO tile:
    recompute QK with int8 MMA from saved q_int8/k_int8
    compute dP = dO @ V.T with fp16 MMA
    compute P and dS from QK, LSE, dP, and Delta
    quantize P, dS*q_scale, and dS*k_scale tiles
    use preprocess-saved dO int8/scales for dV
    accumulate dV = P.T @ dO with int8 MMA
    accumulate dK = (dS*q_scale).T @ q_int8 with int8 MMA
    accumulate dQ = (dS*k_scale) @ k_int8.T with int8 MMA
postprocess:
  convert accumulated dQ workspace to fp16
```

The CUDA source is split like the forward CUTLASS path: `qk_int8_sv_f16_bwd_kernel_cutlass_sm80.cuh` is torch-free and owns kernel params/helpers/kernels, while `qk_int8_sv_f16_bwd_launch_cutlass_sm80.cuh` owns stable-torch validation, temporary allocations, dispatch, and the public wrapper.

The kernel uses CuTe/CUTLASS MMA instead of `<mma.h>` / `nvcuda::wmma`. QK recompute, dV, dK, and dQ use the SM80 `m16n8k32` int8 atom. The fused kernel launches exactly eight generated static CTA variants with a `threadIdx.y` warp grid; active N-warps process `16x16` microtiles inside the configured `BlockN`, while canonical 32-row Q and preprocess-produced int8-dO pairs are staged once per adjacent M pair and fp16 dO plus M-row state are staged per microtile. K/V are full CTA-resident shared tensors loaded through `GmemTiledCopyK` / `GmemTiledCopyV` before active N-warps select microtile views.

Pair `cp.async` staging overlaps the per-row Q-scale/LSE/Delta loads before the shared-read barrier. K scales stay resident while iterating over M, `dK`/`dV` partials stay in register-fragment-shaped tensors, and the final dKV epilogue uses `SmemCopyAtomdKVC` and `GmemTiledCopydKV` for row-contiguous vector stores. Half-precision dO staging aliases `WarpScratchStorage` behind an explicit lifetime barrier; dV `P` and dK `dS*q_scale` pairs remain in `score_pair_i8` so adjacent-M operands survive the next fp16-dO load and per-microtile dQ phase.

One 32-row preprocessing CTA computes coalesced Delta reductions, clears dQ accumulation through `GmemTiledCopyDQZero`, computes each dO pair scale once, and emits packed int8 dO. The standalone dQ conversion uses `GmemTiledCopyDQAccum` / `GmemTiledCopyDQ`. During score materialization, the dead int32 score fragment is recast in place to retain P and the dP fragment is overwritten by dS, so each logical score executes one `expf`.

dV rescales the first P half to the pair scale when needed and reads resident int8 dO through the custom CuTe transposed-B atom. dK similarly rescales the first `dS*q_scale` half and reads resident Q through that atom. Adjacent N warps contribute separate `dS*k_scale` halves to one 32-column operand, agree on one pair scale, and let only the even owner execute dQ. Predicate-zeroed resident rows cover all three tail cases with no explicit K, Q, or int8-dO shared transpose.

## Comparison With FlashAttention Backward

Ignoring unsupported user-facing features, the main performance-relevant difference is that FlashAttention backward is a mature tuned multi-warp CTA schedule, while the current SageAttention CUTLASS backward has only the first static multi-warp CTA variants and still lacks tuned mainloop buffering, epilogue layouts, and several operand staging layouts.

FlashAttention's SM80 backward uses a sequence-K-parallel, KV-owned main kernel with head-dim-specific traits. For the target `head_dim=64/128` cases it launches 8-warp CTAs, uses `BlockM/BlockN` choices such as `64x128`, `128x128`, or `64x64` depending on head dimension and available shared memory, and gives the `S/dP`, `dKV`, and `dQ` matmuls separate `TiledMMA` layouts. It keeps a K/V tile resident while iterating over Q/dO tiles, double-buffers Q/dO when profitable, accumulates `dK`/`dV` in register fragments across the M loop, and only uses the `dQAccum` workspace for the sequence-parallel reduction.

The current CUTLASS backward still uses `16x16` warp-owned microtiles, but generated launch variants now group them into CTA shapes such as `32x128`, `64x128`, `32x64`, and `64x64` with four or eight warps. This reduces the original scaffolding overhead, but the kernel still lacks FlashAttention's tuned per-phase layouts and deeper Q/dO double buffering.

Other structural differences that matter for performance:
- Both kernels now keep K/V resident across the KV-owned M loop, but FlashAttention still has deeper Q/dO mainloop buffering; the current CUTLASS backward only overlaps Q/dO `cp.async` with row-state scalar loads.
- Both kernels now keep `dK` and `dV` partials in register fragments across the M loop and use shared-memory staging for the final stores; FlashAttention still has more mature tuned epilogue layouts.
- FlashAttention exposes `Is_V_in_regs` / `No_double_buffer` traits and tuned mainloop buffering; the current CUTLASS backward has no Q/dO double-buffered mainloop pipeline.
- FlashAttention's low bank-conflict rate comes from an end-to-end shared-memory contract, not from isolated store ordering. Q/dO, K/V, P/dS, dQ, and dKV are physically stored in swizzled layouts; transposed operands are usually alternate views over that same storage, often with `get_nonswizzle_portion(...)`, rather than separate plain transposed scratch copies. P/dS is especially important: FlashAttention stores score fragments with `make_tiled_copy_C_warpcontiguousN(...)` into a dedicated `Swizzle<3,3,3>` `SmemLayoutPdS`, then reuses swizzled or transposed views for dQ/dK/dV. The current CUTLASS backward now gives K, Q, and int8 dO one-allocation phase views and materializes P/dS through a warp-contiguous CuTe C-copy contract; physical-layout tuning and bank-conflict attribution remain post-migration work.
- FlashAttention specializes common even-M/N and even-K cases. Our target shapes are padded to 256 with `head_dim in {64,128}`, so an optimized SageAttention path can assume far less boundary work than the current generic predicated kernel.
- SageAttention has an algorithmic advantage only if the implementation exposes it: `QK`, `dV`, `dK`, and `dQ` should run on fast int8 MMA and use smaller operands. All four CUTLASS backward int8 paths now use `SM80_16x8x32_S32S8S8S32_TN`.

## Benchmark Gate

The correctness migration and evidence-independent cleanup are complete. No obvious redundant allocation, logical-tile recomputation, idle configured warp, duplicate preprocessing reduction, scalar bulk utility copy, or removable CTA barrier remains. The next decisions require paired timing and NCU evidence:
- Reduce shared-memory bank conflicts in the fused CUTLASS backward kernel. The last pre-migration local SM86 NCU baseline for `seq_len=512`, `batch=1`, `heads=16`, `NHD`, and config `(128,64,32,64)` reported `head_dim=64` CUTLASS shared bank conflicts of `1,614` loads and `164,340` stores, or `0.041%` / `16.29%` conflicts per shared load/store wavefront. For `head_dim=128`, it reported `0` load and `329,050` store conflicts, or `0.00%` / `21.80%`. Refresh these counters from the completed migration baseline before judging the first optimization candidate; FlashAttention's comparison baseline was `0` / `9,298` conflicts for head dimension 64 and `130,465` / `467` for head dimension 128.
- Reduce register pressure and improve active-warps/eligible-warps before timing. The last pre-migration CUTLASS profile used `255` registers/thread for both `head_dim=64` and `128`, with achieved active-warps occupancy at `9.2%` and `8.3%` and only `0.119` / `0.100` eligible warps per scheduler. FlashAttention's main backward also uses `255` registers/thread for both, but reaches about `16.6%` achieved active-warps occupancy and `0.135` / `0.122` eligible warps per scheduler.
- Investigate tensor-core utilization and barrier/scoreboard stalls before speed benchmarking. The last pre-migration CUTLASS profile reported only `3.1%` / `2.3%` tensor-pipe activity for `head_dim=64/128`, with barrier stalls around `2.9` / `2.6` warp cycles per issued instruction and short-scoreboard stalls around `1.5` / `1.6`; FlashAttention reported `23.0%` / `30.7%` tensor-pipe activity, similar barrier stalls, and much lower short-scoreboard stalls (`0.13` / `0.53`).

### Utility-Kernel Baseline

The Triton backward already has equivalents for both CUTLASS utility stages. `_zero_dq_accum_kernel` plus `_bwd_preprocess_delta_do_quant` correspond to the combined CUTLASS `preprocess_delta_zero_dq_kernel`; `_convert_dq_accum_kernel` corresponds to `convert_dq_kernel` when `DQ_SPLITS=1`. Triton uses two preprocessing launches while CUTLASS uses one.

An SM86 production-context profile used `batch=1`, `heads=16`, `NHD`, fp16 O/dO/dQ, `BLOCK_M=32`, `BLOCK_N=128`, five warmups, and 30 interleaved repetitions. Both utility sequences ran as part of their complete backward launch order so conversion observed the cache state left by the fused mainloop. Times are median CUDA kernel durations; the combined column includes preprocessing, dQ clearing, and conversion but excludes the fused mainloop.

| Sequence | Head dim | CUTLASS combined | Triton combined | Faster |
|---:|---:|---:|---:|---:|
| 64 | 64 | 3.904 us | 4.513 us | CUTLASS 1.16x |
| 64 | 128 | 6.368 us | 5.312 us | Triton 1.20x |
| 512 | 64 | 16.688 us | 21.186 us | CUTLASS 1.27x |
| 512 | 128 | 51.296 us | 46.835 us | Triton 1.10x |
| 2048 | 64 | 94.914 us | 92.385 us | Triton 1.03x |
| 2048 | 128 | 204.196 us | 176.548 us | Triton 1.16x |
| 4096 | 64 | 162.148 us | 153.650 us | Triton 1.06x |
| 4096 | 128 | 401.207 us | 349.527 us | Triton 1.15x |

CUTLASS preprocessing is faster for short `head_dim=64` inputs, but this does not extend to long sequences or to `head_dim=128`. Conversion is near parity at sequence lengths 2048 and 4096; its short cache-resident results vary more with the preceding workload. There is therefore no general CUTLASS/CuTe utility-kernel speedup to pursue. Keep the existing CUTLASS utilities because they fit its workspace contract and save one preprocessing launch, not because CuTe intrinsically outperforms Triton here.

This is a native shipped-behavior comparison, not identical arithmetic. Triton emits one dO scale per `BLOCK_M x HEAD_DIM` tile, while CUTLASS emits one per `32 x 16` tile. CUTLASS consequently performs more max reductions and writes more scale metadata, especially for `head_dim=128`; replacing either path requires reconciling that quantization contract before treating timing differences as backend code-generation differences.

## Bank-Conflict Work

The pre-migration NCU results above are attribution history, not a current baseline: resident Q/dO pairs, the custom transposed-B atom, and the CuTe score/dS materialization contract changed the shared-memory traffic. The performance phase must begin with fresh NCU resource, conflict-per-wavefront, tensor-pipe, eligible-warp, and spill measurements.

### Migration Status

The correctness-first CuTe migration is complete. K, Q, and quantized dO each have one canonical resident allocation, all explicit shared transpose paths are gone, score/dS materialization is expressed through tensor views and CuTe C-copy contracts, and the kernel uses canonical `SmemLayout*`, `GmemTiledCopy*`, and `SmemCopyAtom*` trait names.

Final validation compiled the wrapper plus all eight generated backward objects for SM80, relinked `_qattn_cutlass_sm80.pyd`, and passed `tests/test_sagebwd_cutlass.py` on SM86 with `32 passed`. Compute Sanitizer Racecheck passed the same matrix with `0 hazards` (`0 errors`, `0 warnings`). The standalone `SM80_U32x4_LDSM_T_INT8_B` copy-plus-MMA test compiled for SM80 and ran on SM86 with `failures=0`. The generated source set contains exactly eight backward instantiations, and the final shared-memory sizes match the resource snapshot below.

### Alternatives To Benchmark

Benchmark these as explicit alternatives from the validated CuTe baseline; change one ownership/layout decision at a time and keep only measured improvements:
- Compare the full logical-tile transposed-B atom against two hardware-atom-sized copies, accepting duplicated LDSM work only if shorter live ranges materially reduce registers.
- Compare the software-assisted transposed-B atom with the old explicit shared transpose as a retained benchmark patch, including instruction count, registers, shared traffic, and conflict-per-wavefront.
- Probe K/Q/dO physical layouts that interleave byte columns or change the LDSM row-address map to reduce warp shuffles or `prmt` operations without duplicating shared data.
- Probe separate swizzle/no-swizzle families for resident K, resident Q, resident int8 dO, score/dS parents, and the dKV epilogue; do not require one universal physical layout.
- Compare single-buffer and double-buffer Q/fp16-dO/int8-dO pair staging, including head-dimension-specific `No_double_buffer` traits.
- Compare V-resident-in-shared with V-in-register phases, and forward versus reverse M traversal, after register live ranges are visible.
- Probe phase-specific CTA-wide `TiledMMA` and A/B/C copy layouts in this order: dKV, dQ, dK, then score/dP.
- Compare scalar dQ atomics with split accumulation planes, larger N ownership, and a separate reduction kernel.
- Run the final bank-conflict/resource/stall gate only after selecting among these alternatives; use conflict-per-wavefront, tensor-pipe activity, eligible warps, registers, shared memory, and spills as the decision metrics.

### Avoid

Avoid unless new evidence changes the tradeoff:
- Direct forward-style composed swizzles on the dV/dQ LDSM views with the existing `SM75_U32x4_LDSM_N` copy path; the tested variants fail CuTe layout vectorization before profiling.
- Forcing the generic `cp.async` tiled-copy helper to four lanes per row fails CuTe layout construction for `head_dim=128` row-4 staging variants, producing zero-sized tiled-copy shapes before profiling.
- Site-specific four-lane `cp.async` staging through `GmemTiledCopyQ` / `GmemTiledCopyK` / `GmemTiledCopyV` / `GmemTiledCopydO` is not worth keeping without a new destination-layout idea: V-only and K-only variants built and passed but were neutral/slightly worse by conflict-per-wavefront, while dO-only and Q-only are blocked by the `Rows=4`, `LineLanes=4`, `kRowsPerIter=8` copy-contract assertion.
- Replacing the per-warp shared `k_scale` cache with register/shuffle broadcasts; it targeted the remaining `STS` offsets but slightly worsened store conflicts per wavefront (`16.03% -> 16.04%` for `head_dim=64`, `21.35% -> 21.35%` for `head_dim=128`) with no register/shared-memory benefit.
- Forcing volatile 16-bit half stores in the dKV epilogue stage to prevent compiler 32-bit store fusion; tests passed, but conflict-per-wavefront was neutral/slightly worse (`16.03% -> 16.03%` for `head_dim=64`, `21.35% -> 21.35%` for `head_dim=128`) and load conflicts rose for `head_dim=64`.
- Retuning the dKV CuTe epilogue with no-XOR `Swizzle<0,3,3>` stage storage or a 64-bit final `GmemTiledCopydKV`; both built and passed tests, but no-XOR worsened store conflicts (`16.29% -> 17.63%`, `21.81% -> 23.46%`) and the 64-bit final copy was neutral/slightly worse (`16.29% -> 16.29%`, `21.81% -> 21.81%`) with no resource benefit.
- Shared-memory padding for bank conflicts, because it increases the shared-memory footprint.
- Switching backward `dP = dO @ V.T` to the forward V `LDSM_T` pattern; that changes the operand contract and does not match the available int8/fp16 copy atoms.
- Repeating the FlashAttention-style B-copy tiler, direct dKV epilogue, packed P/dS store, packed `dS*k_scale` store, register-fed dQ `dS*k_scale`, or transposed-copy rowgroup-first lane remap experiments without a new attribution result. The packed `dS*k_scale` store reduced raw store conflicts but worsened store conflicts per wavefront (`68.79% -> 69.73%` for `head_dim=64`, `70.92% -> 71.53%` for `head_dim=128`).
- Repeating packed `dS*k_scale` after the swizzled score-pair layouts still worsened store conflicts per wavefront (`19.87% -> 21.87%` for `head_dim=64`, `24.80% -> 26.44%` for `head_dim=128`), so it remains rejected.
- Register-feeding dQ's `dS*k_scale` operand from warp shuffles removed the `sdSk` shared path and reduced raw load/store conflicts, but worsened store conflicts per wavefront (`63.42% -> 65.45%` for `head_dim=64`, `66.66% -> 68.44%` for `head_dim=128`), so it is rejected unless paired with a store-rate fix.
- The transposed-copy rowgroup-first remap reduced store conflict rates (`63.42% -> 58.83%` for `head_dim=64`, `66.66% -> 60.95%` for `head_dim=128`) but worsened load conflict rates (`17.10% -> 25.26%`, `19.83% -> 29.03%`) and raised `head_dim=64` registers (`228 -> 232`).

## Resource Budget Snapshot

Current dynamic shared-memory sizes for the generated fused backward variants are:
- `head_dim=64`: `32x128x4` uses `37,328` bytes, `64x128x8` uses `45,792` bytes, and both `32x64x4` / `64x64x4` use `25,040` bytes.
- `head_dim=128`: `32x128x4` uses `66,000` bytes, `64x128x8` uses `74,464` bytes, and both `32x64x4` / `64x64x4` use `41,424` bytes.
- Active N scratch slots are capped by `kSmemWarps = min(NumWarps, CtaN / 16)`. Each `WarpScratchStorage` slot serves `dS*k_scale` and is later reused by the dKV half epilogue; dV P and dK `dS*q_scale` share the separate score-pair parent.
- Q and int8-dO cooperative pair staging uses two loader warps for `head_dim=64` and four loader warps for `head_dim=128`; fp16 dO uses the same loader-warps-per-microtile schedule. The resident pairs intentionally raise the correctness-migration footprint, with physical-layout and buffering alternatives deferred to post-migration measurement.
- Final SM80 cubin resources have zero stack/local memory for all eight fused variants. Registers are `196`, `164`, `164`, and `164` for the `head_dim=64` `32x128`, `64x128`, `32x64`, and `64x64` variants; the corresponding `head_dim=128` values are `255`, `250`, `252`, and `250`.
- The earlier NCU dynamic-shared values of `34.24 KiB` / `59.84 KiB` predate resident Q/dO pairs. Refresh occupancy, stall, and shared-conflict comparisons from the completed migration baseline before selecting an optimization.

## Benchmark-Guided Performance Issues

These should be decided with paired A/B timing and Nsight Compute after the structural schedule above exists:
- Exact static tile choice among the small set above, plus 4 vs 8 warps and per-phase atom layouts for `S/dP`, `dKV`, and `dQ`, should be tuned separately for `head_dim=64` and `128` on sm80/sm86.
- Pipeline depth and lifetime choices: Q/dO double buffering, V-in-register variants, reverse M iteration, and cp.async placement should be kept only when they reduce scoreboard stalls without causing spills or occupancy loss.
- Even-shape specialization: a fast path for padded `seq_len % 256 == 0` and exact `head_dim` may remove predicate overhead, but prior forward experiments showed full-tile specialization can be noisy, so keep it data-driven.
- `dQAccum` strategy: scalar atomics, split planes, larger `BlockN`, or non-atomic partial reductions trade global reduction traffic against extra workspace and launch cost. FlashAttention also pays `dQAccum` traffic in the sequence-parallel path, so this needs measurement rather than assumption.
- Scale/LSE/Delta placement: shared-memory staging is simple, while register/shuffle staging may reduce barriers and shared traffic but can increase register pressure. Choose based on NCU register, stall, and shared-memory metrics.
- P/dS quantization granularity and max-reduction strategy: per-warp/per-tile scales and where to compute maxima should be benchmarked under the accuracy contract.
- Utility-kernel balance: preprocessing already fuses `Delta`, dO quantization, and `DQAccum` zeroing. The production-context baseline shows no broad CUTLASS advantage and no long-sequence conversion bottleneck; reconsider fusing conversion or changing copy widths only if full-backward profiles on another target GPU attribute material latency to these stages.

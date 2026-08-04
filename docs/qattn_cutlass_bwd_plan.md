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

The kernel uses CuTe/CUTLASS MMA instead of `<mma.h>` / `nvcuda::wmma`. QK recompute, dV, dK, and dQ use the SM80 `m16n8k32` int8 atom. Normal generation retains exactly eight static public CTA instantiations. Seven use the generic `threadIdx.y` N-warp schedule; the `head_dim=64`, public `64x64x32x64` instantiation selects an internal eight-warp `2 M-halves x 4 N-microtiles` specialization while head-128 keeps the generic four-warp path. Canonical 32-row Q, fp16 dO, and preprocess-produced int8-dO pairs are staged once per adjacent M pair. K/V are full CTA-resident shared tensors loaded through `GmemTiledCopyK` / `GmemTiledCopyV` before warps select microtile views.

The arithmetic contract is fixed at four int8 MMA paths (`QK`, dV, dK, and dQ) plus one fp16 MMA path (`dO @ V.T`). The dP path stays fp16 because its accuracy is insufficient with int8; converting dV/dK/dQ to fp16 is also out of scope because it discards the intended 2x SM80 tensor-core throughput and the 0.5x int8 operand footprint. Performance work must expose that int8 advantage by reducing quantization, ownership, and staging overhead without changing the five-MMA precision contract validated by the tests.

Pair `cp.async` staging overlaps the per-row Q-scale/LSE/Delta loads before the shared-read barrier. K scales stay resident while iterating over M. In the active eight-warp specialization, each `(M-half, N-tile)` owner keeps one 32-float dV or dK fragment across the M loop; the final epilogue uses `SmemCopyAtomdKVC` and `GmemTiledCopydKV` for row-contiguous vector stores. Half-precision dO staging aliases `WarpScratchStorage` behind an explicit lifetime barrier. Four score-pair slots retain one complete P/`dS*q_scale` pair per N tile, and four scratch slots retain both `dS*k_scale` M-halves for each adjacent N pair.

One 32-row preprocessing CTA computes coalesced Delta reductions, clears dQ accumulation through `GmemTiledCopyDQZero`, computes each dO pair scale once, and emits packed int8 dO. The standalone dQ conversion uses `GmemTiledCopyDQAccum` / `GmemTiledCopyDQ`. During score materialization, the dead int32 score fragment is recast in place to retain P and the dP fragment is overwritten by dS, so each logical score executes one `expf`.

The active head-64 specialization exchanges P and `dS*q_scale` maxima between the two M-half owners of each N tile, and exchanges `dS*k_scale` maxima between adjacent N-tile owners. It therefore quantizes all three operands directly at their final pair scales and eliminates every pair store/reload/re-round rescale pass. dV reads resident int8 dO, while dK and dQ read resident Q and K through the custom CuTe transposed-B atom. Only the even N-tile owner executes dQ; its 4x8 shuffle ownership halves global reduction sectors to `1,048,576` at the 512-token profile shape. Generic variants retain the validated two-stage path. Predicate-zeroed resident rows cover all tail cases with no explicit K, Q, or int8-dO shared transpose.

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

## Benchmark And Profiling Results

`bench/bench_sagebwd_cutlass.py` now benchmarks the requested `num_heads={16,32}`, `head_dim={64,128}`, and `seq_len={4096,8192}` grid, all four generated backward configurations, and both `kernel_only` and `end_to_end` modes. It reports median, mean, standard deviation, effective backward TFLOPS, Sage speedup versus FlashAttention, and per-shape configuration rank. `kernel_only` reuses prequantized Q/K and output buffers; `end_to_end` includes Sage Q/K quantization and output allocation. FlashAttention setup is outside the timed backward call, and Sage forward setup uses the same block configuration as backward.

The first full matrix and reversed-order sweeps still show Sage below FlashAttention. The best schedules are generally `64x64x32x64` and `128x32x32x32`, with Sage/Flash speedups approximately `0.41-0.57x` for head dimension 64 and `0.44-0.48x` for head dimension 128. Kernel-only and end-to-end results differ by roughly `0-2%`, so the main gap is the fused backward kernel rather than quantization or allocation. A focused post-optimization sweep at `batch=1`, `heads=16`, `seq_len=4096`, and `kernel_only` measured:

| Head dim | `128x64x32x64` | `128x64x16x64` | `128x32x32x32` | `64x64x32x64` | Flash median |
|---:|---:|---:|---:|---:|---:|
| 64 | 14.202 ms | 13.200 ms | 10.905 ms | **10.696 ms** | 6.212 ms |
| 128 | 29.081 ms | 22.378 ms | 19.833 ms | **19.417 ms** | 9.325 ms |

The table records the pre-eight-warp comparison. The accepted launch-bounded head-64 specialization measures `0.329-0.330 ms` at 512 tokens and `9.02-9.22 ms` at 4096 tokens across 30-warmup/100-repeat serial runs; FlashAttention measured `0.429-0.460 ms` and `4.55-4.63 ms`. Thus Sage wins the short shape but remains about `1.98x` slower at 4096. Repeated measurements are interpreted with warmups and reversed order. A serial thermal diagnostic on `heads=32`, `head_dim=128`, `seq_len=8192` showed about `2.3%` latency drift while GPU telemetry varied from `58-77 C`, `102.95-123.6 W`, and `1417-1725 MHz`, so close results are not treated as decisive from one run.

### Current Nsight Attribution

The refreshed SM86 Nsight Compute baseline uses `seq_len=512`, `batch=1`, `heads=16`, NHD, config `64x64x32x64`, and ten warmups. The fused head-64 kernel takes `290.432 us`, executes `17.176M` instructions, uses `164` registers and `25,040` bytes of dynamic shared memory, and has `0.29` eligible warps per scheduler. It reports `20.92%` barrier stalls and `21.46%` short-scoreboard stalls. The fused head-128 kernel takes `684.672 us`, executes `23.745M` instructions, uses `250` registers and `41,424` bytes of shared memory, and has `0.23` eligible warps per scheduler, with `16.86%` barrier and `22.47%` short-scoreboard stalls.

Shared-memory traffic is substantial but is not yet the clearest first target: head-64 reports `417,754` load conflicts and `8,309` store conflicts over `3,006,426` load and `590,547` store wavefronts; head-128 reports `938,443` load conflicts and `16,128` store conflicts over `5,116,371` load and `639,327` store wavefronts. Tensor activity remains low relative to FlashAttention: Sage head-64 reports `5.53%` IMMA and `2.77%` HMMA active cycles, while head-128 reports `4.69%` IMMA and `2.35%` HMMA. FlashAttention's main backward kernels execute about `2.716M` and `4.908M` instructions and reach approximately `22.97%` and `30.61%` HMMA activity for head dimensions 64 and 128.

Line-info source correlation on the older generic head-64 path attributed instruction and global-sector overhead to repeated scalar int8 conversion/repacking and dQ accumulation. The current bounded eight-warp profile instead executes about `15.112M` instructions at 128 registers with `25,312` bytes of dynamic shared memory, `3.58` active and `0.44` eligible warps per scheduler, and `1,048,576` global-reduction sectors. Its `802,816` local-load and `61,440` local-store sectors do not come from the persistent 32-float dKV fragment: SASS keeps that fragment in `R5` through `R36`. Ten scalar/shared-view values are reloaded on every M pair, with four more reloads around dQ atomics. The next no-arithmetic-change experiment should shorten or reconstruct those phase-specific CuTe view/address values rather than move dKV accumulation to shared memory.

### Retained Optimization

The direct pair-scale dS quantization change is retained only for `HeadDim=64, CtaM=64, CtaN=64, NumWarps=4`. It removes one pair of shared reads, one int8 store pass, and one re-rounding pass per adjacent N-warp pair. The focused candidate reduced representative head-64 fused execution from `17.176M` to `15.933M` instructions and duration from `290.432 us` to `276.352 us` in separate `--clock-control none` profiles, while preserving `164` registers, `25,040` bytes of shared memory, zero spills, and the existing correctness contract. The analogous un-gated head-128 experiment was rejected because it reached `255` registers and regressed the long-shape timing; the original head-128 path was restored and reprofiled at `23.745M` instructions and `250` registers.

### Focused dQ Experiment

Optimization is temporarily focused on the head-64 `64x64x32x64` public block configuration. `SAGEATTN_CUTLASS_BWD_FOCUSED_BUILD=1` makes `setup.py` generate and link only that head/config dispatch while iterating; an unset variable still generates the normal eight backward instantiations. A full build and the complete test matrix remain required before retention.

The retained dQ register permutation gives each atomic instruction four rows by eight contiguous columns. It uses two scalar 32-bit warp shuffles per output iteration, applies to every dQ fragment, and reduces `l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum` from `2,097,152` to `1,048,576` at the 512-token profile shape. The earlier four-warp form measured about `10.25 ms` at 4096 tokens and raised registers to `168`; the accepted eight-warp schedule absorbs the same permutation while reducing total instruction count and splitting dKV ownership. Four focused tail/layout cases pass and the production-selected path is Racecheck-clean.

The earlier all-lane row-major permutation is rejected: it paid 32 hardware shuffles per 16x16 fragment and regressed the 4096-token kernel-only result to about `12.11 ms`. The first packed shuffle order is also rejected because it increased reduction sectors to `5,242,880`.

### Allocation-Lifetime Audit

- The score accumulator is already recast in place as P and the dP accumulator is overwritten as dS. These register allocations have disjoint phases and should remain reused.
- `do_fp16` already aliases `WarpScratchStorage`: fp16 dO is consumed by dP before the same bytes hold `dS*k_scale`, and the scratch is reused again for the final half dKV epilogue. This is the main shared-memory lifetime reuse and must retain its barriers.
- `q_i8` and `do_i8` cannot alias in the active eight-warp schedule because the dV and dK owners consume both resident pairs in the same phase. Resident K and V also span the complete M loop and overlap.
- P and `dS*q_scale` occupy disjoint halves of `score_pair_i8` and coexist until the dV/dK MMAs. Each warp's single dK or dV accumulation fragment spans the complete M loop and cannot alias score state.
- Line-info shows that phase-spanning shared tensor views and scalar coordinates, not the dKV fragment, cause the launch-bound local traffic. Reconstructing selected views inside their use phase is the next low-risk register test.
- LSE, Delta, and row scales are small. They overlap during score/P/dS materialization; aliasing the dead tail with the four-float scale exchange would save too little shared memory to affect occupancy.

The accepted structural schedule keeps all four int8 MMAs and changes warp ownership only for the focused `64x64` CTA. Eight warps cover a `2 M-halves x 4 N-microtiles` grid concurrently. For each N tile, one M-half warp owns persistent dV and the other owns persistent dK, so each warp carries one 32-float dKV accumulator instead of both. The two M warps exchange maxima before directly quantizing P and `dS*q_scale` at their final pair scales; adjacent N tiles similarly exchange maxima before directly quantizing `dS*k_scale`. This removes all P, dSq, and dSk pair-rescale passes without changing the four-int8/one-fp16 contract.

The first implementation passes the four focused correctness cases and Racecheck with zero hazards. It reduces the fused profile from the four-warp shuffle candidate's `17.480M` instructions and `168` registers to `14.654M` instructions and `146` registers, keeps reduction sectors at `1,048,576`, uses `25,312` bytes of shared memory, and reports zero local spills. Shared load conflicts are `393,216` over `2,867,200` wavefronts and stores fall to `50` conflicts over `399,922` wavefronts. The split ownership also reduces short-scoreboard stalls to `13.51%`, but the 256-thread block is limited to one resident CTA by registers: active warps fall to `2.01` per scheduler, eligible warps to `0.27`, barrier stalls rise to `30.04%`, profile duration rises to about `340 us`, and 4096-token kernel-only time regresses to about `11.69 ms`.

A `__launch_bounds__(256, 2)` build reaches the required `128` registers and raises active/eligible warps to `3.58` / `0.44` per scheduler. It improves the focused 4096-token result to about `9.02-9.05 ms`, but compiler spilling produces `802,816` local-load and `61,440` local-store sectors at 512 tokens; instructions rise to `15.112M` and barrier stalls remain high at `32.81%`. CUDA 13 provides `.pragma "enable_smem_spilling"` on SM75 and newer. PTXAS rejects the pragma on a kernel that declares dynamic shared memory. Converting the exact specialization's fixed allocation to static shared makes the pragma legal, but does not migrate these spills. Source-correlated SASS later established that the 32-float dKV fragment remains scalarized in registers while phase-spanning address/view state is spilled.

The bounded partial-accumulator variant explicitly kept two of the four 16-column float dKV fragments in registers and laid out the other two in lane-contiguous shared memory. The added `16,384` bytes raised explicit storage to `41,696` bytes and reduced local sectors to `327,680` loads / `45,056` stores, but added shared traffic and regressed 4096-token timing from about `9.05` to `9.31 ms`. Splitting the retained register tensor into fixed 1D fragments did not change generated resources or local traffic. This explicit shared-accumulator path is rejected; restore the faster compiler-spilled launch-bounded form and use line-info attribution before another ownership change.

## Benchmark Gate

The correctness migration and evidence-independent cleanup are complete. No obvious redundant allocation, logical-tile recomputation, idle configured warp, duplicate preprocessing reduction, scalar bulk utility copy, or removable CTA barrier remains. The next decisions require paired timing and NCU evidence:
- Reduce shared-memory bank conflicts only when a change also improves timing or removes a measured dependency. The current migration baseline reports head-64 fused shared conflicts of `417,754` loads and `8,309` stores, and head-128 conflicts of `938,443` loads and `16,128` stores for the measured `64x64` path. These are paired with substantial wavefront counts and must be evaluated as conflicts per wavefront, not raw totals.
- Reduce the active head-64 path's local traffic without exceeding `128` registers: it has `3.58` active and `0.44` eligible warps per scheduler but reloads ten scalar/view values per M pair. Head-128 remains at `250` registers and `0.23` eligible warps per scheduler on its generic path.
- Reduce the active head-64 path's `32.81%` barrier stalls after local-address pressure. The historical generic baseline reported `5.53%` / `4.69%` IMMA activity and `2.77%` / `2.35%` HMMA activity for head dimensions 64/128; FlashAttention's main backward reaches about `22.97%` / `30.61%` HMMA activity.

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
- For head-64 `64x64`, shorten or reconstruct the source-attributed phase-spanning CuTe shared-view/address state while preserving the accepted eight-warp ownership, 128-register occupancy threshold, and four-int8/one-fp16 MMA contract.
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
- Replacing dV, dK, or dQ with fp16 MMA. The validated Sage backward accuracy/performance contract is four int8 MMA paths plus fp16 `dO @ V.T`; fp16 gradient MMAs give up both the SM80 int8 throughput advantage and compact operand storage.
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
- `head_dim=64`: `32x128x4` uses `37,328` bytes, `64x128x8` uses `45,792` bytes, `32x64x4` uses `25,040` bytes, and the public `64x64x4` instantiation's internal eight-warp kernel uses `25,312` bytes.
- `head_dim=128`: `32x128x4` uses `66,000` bytes, `64x128x8` uses `74,464` bytes, and both `32x64x4` / `64x64x4` use `41,424` bytes.
- Active generic N scratch slots are capped by `kSmemWarps = min(NumWarps, CtaN / 16)`. The exact eight-warp storage has four score-pair slots and four dSk/dKV scratch slots; fp16 dO aliases the latter behind a CTA barrier.
- Q and int8-dO cooperative pair staging uses two loader warps for `head_dim=64` and four loader warps for `head_dim=128`; fp16 dO uses the same loader-warps-per-microtile schedule.
- The active head-64 `64x64` kernel is launch-bounded to `128` registers and reports `802,816` local-load / `61,440` local-store sectors at 512 tokens. The other head-64 generic variants use `196`, `164`, and `164` registers for `32x128`, `64x128`, and `32x64`; head-128 uses `255`, `250`, `252`, and `250` registers. The seven generic fused variants retain zero stack/local memory.

## Benchmark-Guided Performance Issues

These should be decided with paired A/B timing and Nsight Compute after the structural schedule above exists:
- Exact static tile choice among the small set above, plus 4 vs 8 warps and per-phase atom layouts for `S/dP`, `dKV`, and `dQ`, should be tuned separately for `head_dim=64` and `128` on sm80/sm86.
- Pipeline depth and lifetime choices: Q/dO double buffering, V-in-register variants, reverse M iteration, and cp.async placement should be kept only when they reduce scoreboard stalls without causing spills or occupancy loss.
- Even-shape specialization: a fast path for padded `seq_len % 256 == 0` and exact `head_dim` may remove predicate overhead, but prior forward experiments showed full-tile specialization can be noisy, so keep it data-driven.
- `dQAccum` strategy: scalar atomics, split planes, larger `BlockN`, or non-atomic partial reductions trade global reduction traffic against extra workspace and launch cost. FlashAttention also pays `dQAccum` traffic in the sequence-parallel path, so this needs measurement rather than assumption.
- Scale/LSE/Delta placement: shared-memory staging is simple, while register/shuffle staging may reduce barriers and shared traffic but can increase register pressure. Choose based on NCU register, stall, and shared-memory metrics.
- P/dS quantization granularity and max-reduction strategy: per-warp/per-tile scales and where to compute maxima should be benchmarked under the accuracy contract.
- Utility-kernel balance: preprocessing already fuses `Delta`, dO quantization, and `DQAccum` zeroing. The production-context baseline shows no broad CUTLASS advantage and no long-sequence conversion bottleneck; reconsider fusing conversion or changing copy widths only if full-backward profiles on another target GPU attribute material latency to these stages.

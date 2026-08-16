# QAttn CUTLASS Forward Kernel Plan

## Scope

- Target architectures: sm80 (sm86).
- Inputs: fp16 `q`, `k`, and `v`. bf16 is deferred.
- Non-causal only.
- `pv_accum_dtype="fp32"` only. No instruction-buffer PV variant.
- Optional LSE output for backward consumers.
- Fixed-length dense tensors only.
- Supported tensor layouts: `HND` and `NHD` via the same stride handling as the existing CUDA qattn path.
- Head dims: `64` and `128` only. Both use the canonical Q32/K64 per-block artifact.
- Block configs:
  - `(blk_q=128, blk_k=64, warp_q=32, warp_k=64)`
  - `(blk_q= 64, blk_k=64, warp_q=32, warp_k=64)`
  - `(blk_q=128, blk_k=32, warp_q=32, warp_k=32)`
  - `(blk_q=128, blk_k=64, warp_q=16, warp_k=64)`
- Exposed through `sageattention.cutlass_attn.sageattn_qk_int8_pv_fp16_cutlass` with CUDA-style eager and `torch.compile` autotune plumbing, but not selected as the default `sageattn` backend.

## Current Implementation

The maintained wrapper uses the canonical Triton `per_block_int8` layout with `QBlock=32` and `KBlock=64` for both supported head dimensions. The CTA/warp execution configuration remains independently selectable. The wrapper does:

```text
head_dim in {64,128}: q, k -> per_block_int8(Q32,K64) -> q_int8, q_scale, k_int8, k_scale
q_int8, k_int8, v, scales -> qattn kernel -> output[, lse]
```

The kernel mirrors the existing CUDA forward kernel's high-level dataflow:

```text
for each Q CTA block:
  load Q tile once into shared memory
  for each K/V tile:
    load K tile into shared memory
    load V tile into shared memory
    S = int8(Q) @ int8(K)^T * q_scale * k_scale * sm_scale
    update online softmax state m, d
    convert P tile to fp16
    O += fp16(P) @ fp16(V) with fp32 accumulation
  normalize O by d
  store O
```

### Mutation-only compile contract

Inductor's ordinary `torch.compile` path already supports custom operations that return no tensors while mutating caller-owned buffers, but its custom-op autotuner currently requires a tensor result layout. The project installs `sageattention.torch_compile_patch.install_mutation_only_custom_op_autotuning` before CUTLASS autotune registration. For mutation-only candidates, the patch adds a private zero-element `uint8` token while tracing and benchmarking, preserving the real buffer mutations and letting the unused token be removed from the final compiled wrapper. Tensor-returning custom ops continue through the unmodified Inductor path.

Mutation-only registrations do not get Inductor's ordinary eager fallback choice because that choice has no output layout. A baseline implementation, when needed, must be supplied as one of the `CustomOpConfig` candidates. The autotuning patch does not change dependency analysis or alias metadata; the custom-op schema remains responsible for declaring every mutated buffer.

The Python frontends allocate the padded output buffer outside the opaque operation. They allocate a caller-owned float32 LSE buffer only when `return_lse=True`; otherwise they pass Python `None`. The CUDA and CUTLASS stable operators declare `Tensor? lse`, retain a `nullptr` launch pointer when it is absent, and skip LSE validation and writes in that mode. Triton follows the same caller-owned buffer contract. Direct FakeTensor calls take this allocation and mutation-only custom-op path as well, avoiding eager autotuning against fake data pointers. The internal forward ops also register a narrow Functionalize redispatch rule, so standalone `torch.func.functionalize` unwraps functional tensors and executes the native mutation while preserving the caller-visible result. Consequently, the no-LSE compiled graph contains no LSE allocation or LSE input.

The kernel is self-contained apart from library headers. Shared, global, and register storage are exposed through CuTe tensors where doing so preserves the existing schedule and generated work. Q/K/V global-to-shared loads and O shared-to-global stores use CuTe `TiledCopy` partitioning, full-problem identity-coordinate tensors tiled and partitioned with the data, lazy predicate tensors, and `copy_if`. Predicated global-to-shared loads use CuTe's `SM80_CP_ASYNC_CACHEGLOBAL_ZFILL` atom so out-of-bounds boundary lanes are initialized to zero. Shared-memory tiles use stride-derived CuTe swizzle layouts and CuTe `ldmatrix` copy atoms. CTA/warp row selection uses local CuTe layout helpers and `local_tile` over full gmem matrices, while Q/K scale indexing and lane subcoordinates are centralized in small CuTe layout helpers instead of open-coded warp/lane arithmetic.

The core MMA instructions execute the same logical QK `m16n16k32` and PV `m16n16k16` work. `compute_int_qk` and `compute_fp16_sv` route through local CuTe `TiledMMA` helpers and `cute::gemm` over CuTe-owned register fragments. Q/K/V smem-to-register loads allocate fragments with `thr_mma.partition_fragment_A/B` and retile the relevant `TiledCopy` slices into those fragments. Tail masking, online-softmax state updates, denominator accumulation, output normalization, and fp16 probability handoff use CuTe `transform`, `for_each`, and `reduce` over tensor slices or coordinate tensors instead of open-coded element loops. The probability handoff aliases `score_f32` with `prob.layout()` and runs the vectorized f32-to-f16 converter once. The output store uses shared-memory staging, with a local CuTe C-copy source layout for `make_tiled_copy_C`. Its stride is derived from the accumulator layout because the direct nested PV C layout has a different register traversal order.

## Benchmark Snapshot

The maintained benchmark always uses raw K (`smooth_k=False`). FlashAttention is called through normal `flash_attn_func` dispatch without a Sage block-size constraint, so each Sage execution configuration is compared with FlashAttention's library-selected kernel. Effective forward throughput uses `4 * batch * heads * head_dim * seq_len^2` FLOPs for QK and PV and the median CUDA-event time.

The final grid used batch 1, 25 warmups, and 100 repeats on an NVIDIA GeForce RTX 3080 Ti Laptop GPU. `end_to_end` includes Q32/K64 Q/K quantization, output allocation, and the CUTLASS attention kernel. `kernel_only` uses prequantized Q/K and a preallocated output. The table selects the lowest CUTLASS median independently for each shape and mode; Flash is measured once per shape with its unconstrained dispatch.

| Layout | Heads | Head dim | Seq | Flash TFLOPS | Sage end-to-end TFLOPS | End-to-end speedup | End-to-end config | Sage kernel-only TFLOPS | Kernel-only speedup | Kernel-only config |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| NHD | 16 | 64 | 4096 | 39.302 | 49.545 | 1.261x | `128x64x32x64` | 56.303 | 1.433x | `128x64x32x64` |
| NHD | 16 | 64 | 8192 | 41.660 | 57.108 | 1.371x | `128x32x32x32` | 59.487 | 1.428x | `128x32x32x32` |
| NHD | 16 | 128 | 4096 | 42.333 | 56.998 | 1.346x | `128x64x32x64` | 59.881 | 1.415x | `128x64x32x64` |
| NHD | 16 | 128 | 8192 | 43.079 | 60.067 | 1.394x | `128x64x32x64` | 61.788 | 1.434x | `128x64x32x64` |
| NHD | 32 | 64 | 4096 | 40.814 | 53.082 | 1.301x | `128x32x32x32` | 56.776 | 1.391x | `128x64x32x64` |
| NHD | 32 | 64 | 8192 | 41.469 | 55.545 | 1.339x | `128x64x32x64` | 57.478 | 1.386x | `128x32x32x32` |
| NHD | 32 | 128 | 4096 | 41.089 | 55.594 | 1.353x | `128x64x32x64` | 59.011 | 1.436x | `128x64x32x64` |
| NHD | 32 | 128 | 8192 | 41.783 | 57.927 | 1.386x | `128x32x32x32` | 60.941 | 1.458x | `128x64x32x64` |
| HND | 16 | 64 | 4096 | 39.246 | 48.630 | 1.239x | `128x32x32x32` | 55.485 | 1.414x | `128x64x32x64` |
| HND | 16 | 64 | 8192 | 40.568 | 56.849 | 1.401x | `128x32x32x32` | 59.692 | 1.471x | `128x32x32x32` |
| HND | 16 | 128 | 4096 | 41.491 | 56.421 | 1.360x | `128x64x32x64` | 58.015 | 1.398x | `128x64x32x64` |
| HND | 16 | 128 | 8192 | 41.123 | 59.593 | 1.449x | `128x64x32x64` | 61.805 | 1.503x | `128x64x32x64` |
| HND | 32 | 64 | 4096 | 40.833 | 53.030 | 1.299x | `128x32x32x32` | 57.604 | 1.411x | `128x32x32x32` |
| HND | 32 | 64 | 8192 | 41.813 | 55.994 | 1.339x | `128x32x32x32` | 58.129 | 1.390x | `128x64x32x64` |
| HND | 32 | 128 | 4096 | 41.191 | 56.240 | 1.365x | `128x64x32x64` | 59.705 | 1.449x | `128x64x32x64` |
| HND | 32 | 128 | 8192 | 42.111 | 59.550 | 1.414x | `128x64x32x64` | 60.075 | 1.427x | `128x32x32x32` |

Best-config end-to-end speedup is `1.239-1.449x` with a `1.350x` geometric mean. Kernel-only speedup is `1.386-1.503x` with a `1.427x` geometric mean. End-to-end Sage throughput spans `48.630-60.067 TFLOPS`; kernel-only throughput spans `55.485-61.805 TFLOPS`.

Full per-config results are in `build/bench_qattn_cutlass_rawk_NHD_4096_8192.csv` and `build/bench_qattn_cutlass_rawk_HND_4096_8192.csv`. Absolute laptop timings remain clock- and temperature-sensitive, so sub-percent kernel changes still require paired same-build controls.

## Comparison with SageAttention CUDA Kernel

The CUTLASS kernel intentionally mirrors the CUDA kernel's main architecture for the overlapping fp16, non-causal, fp32-PV path: same CTA/warp tiling choices, same `m16n16k32` QK and `m16n16k16` PV tensor-core shapes, same single shared-memory Q/K/V staging buffer overlaid with the O staging buffer, same K/V software pipeline, same CUDA-core denominator strategy, and the same shared-memory epilogue pattern. CUTLASS now uses fixed Q32/K64 scale domains for both head dimensions, while the CUDA path retains its execution-tile-dependent per-thread metadata. The remaining differences below are architectural or scheduling choices that can affect performance, not just CuTe-vs-CUDA code style.

### Global-memory copy tiling matches the scoped CUDA path

The CUDA kernel chooses copy-lane grouping from the original packed element stride. For `head_dim=64`, Q/K use `4` lanes per 128-bit row segment and V/O use `8` lanes. For `head_dim=128`, Q/K use `8` lanes and V/O use `8` lanes. The CUTLASS `make_2d_tiled_copy` helper matches those choices for the scoped int8-QK/fp16-V/O shapes: Q/K have `HeadDim / 16` packed columns and V/O have `HeadDim / 8` packed columns. This keeps warp row coverage and 128-bit traffic aligned with the hand CUDA path. Remaining copy differences are in CuTe partitioning and predicate expression rather than the lane grouping policy.

### Q-fragment caching is not active for supported dims

The CUDA kernel has a special Q-in-register path when `num_tiles_qk_inner == 1`. The CUTLASS scope supports only `head_dim in {64,128}`, which gives `num_tiles_qk_inner in {2,4}` for the QK `k32` MMA, so this fast path is inactive for the supported CUTLASS shapes. Prior CUTLASS experiments that cached Q fragments for `head_dim=64` regressed, likely from extra register pressure, so the CUTLASS kernel continues to reload Q fragments from shared memory inside each QK tile iteration.

### Denominator accumulation uses CUDA cores

The existing CUDA kernel supports both CUDA-core and tensor-core denominator accumulation. The CUTLASS path keeps only the CUDA-core strategy for the scoped fp32-PV path. Tensor-core denominator accumulation was tested in the CUTLASS path. On the local sm86 target it increased HMMA instructions by `12.5%` and slowed the profiled target kernel by about `7.8%`, with no occupancy/resource improvement. This leaves some scalar/shuffle work in normalization, but it is faster for the measured fp32-accum path.

### Boundary handling remains generic

The CUTLASS staged loop uses unpredicated middle-tile K/V loads and keeps predicated boundary loads plus final score masking for correctness on arbitrary sequence lengths. Full-tile fast paths for K/V and Q/output boundaries were tested on the standard divisible sequence lengths, but their median timing was mixed and they were reverted. Keep the generic boundary path unless profiling shows predicate/control overhead is a primary remaining bottleneck.

## Comparison vs FlashAttention Forward Kernels

This comparison focuses on FlashAttention ideas that can apply to SM80/SM86. Hopper/Blackwell-only mechanisms are intentionally omitted.

### Tile policy is more aggressively tuned

FlashAttention's SM80 forward kernels tune `(BlockM, BlockN, num_warps, Q_in_regs)` by head dimension, architecture, dropout, causal/local mode, split-K, and sometimes SM86/SM89 specifically. Examples from the SM80 fp16 path include `128x128` with `4` warps for `head_dim=64` non-dropout and a smaller K tile such as `128x32` for `head_dim=128` non-causal on SM86 to improve occupancy. Newer FlashAttention SM80 policies also expose `Q_in_regs` and `8`-warp choices for some `head_dim=128+` cases. The CUTLASS qattn path tests a small hand-CUDA-inspired config set, with the default/best standard result usually `(128,64,32,64)`. Retuning CTA-K, warp-Q, warp count, and Q-in-register choices against the int8 score/register footprint is likely more useful than assuming the fp16 FlashAttention tile policy transfers directly.

### Pipeline depth is a tuning lever

FlashAttention's newer SM80 mainloop carries an explicit K/V pipeline stage dimension in shared memory and has a `num_stages` policy, even though many fp16 `head_dim=64/128` SM80 choices use one stage. The CUTLASS qattn path uses a CUDA-style staged schedule with one K and one V tile resident at a time. A deeper K/V circular buffer could help only if profiling shows `cp.async` wait or long-scoreboard stalls. Otherwise it risks extra shared memory, lower occupancy, and more register pressure. This is especially sensitive for the int8 path because the faster QK MMA can shift the bottleneck toward softmax/PV and scalar work.

### Reverse K iteration is worth considering

FlashAttention processes K/V blocks from the tail toward the beginning, so the first iteration handles the sequence-tail mask and the remaining non-causal iterations are mostly unmasked. The CUTLASS qattn path iterates forward and handles the tail at the end. Both do the same amount of logical work for dense non-causal attention, but the reverse order can reduce loop-carried state, organize masked/unmasked loops more cleanly, and interact differently with prefetch placement. This is a plausible micro-schedule experiment if NCU points at control/predicate overhead or load scheduling.

### Even-shape specialization is already a known tradeoff

FlashAttention specializes common aligned cases with compile-time `Is_even_MN` and `Is_even_K` flags to remove predicate work. The CUTLASS qattn path keeps generic predicated boundary copies and final score masking. Full-tile specialization was already tested locally and reverted because timing was not robust, so this should stay deferred unless a larger schedule/layout rewrite changes the cost model.

### Split-K and packed GQA are larger transferable optimizations

FlashAttention has split-KV forward variants with a combine kernel, and newer kernels can pack multiple Q heads per KV head to reduce duplicated K/V work for GQA/MQA. The CUTLASS qattn path currently launches one CTA grid over query heads and reuses grouped K/V only through head indexing, not through packed-head tiling. These ideas are transferable to SM80/SM86 and may matter for low grid counts, long KV lengths, or large `num_kv_groups`, but they require new partial-output/LSE storage or a different CTA tile shape and should be treated as larger features rather than local cleanups.

### Quantized QK changes the bottleneck

Several differences are justified by int8 Q/K rather than missing FlashAttention optimizations. The qattn path uses `m16n16k32` int8 QK MMA, int32 score accumulators, fixed Q32/K64 scale loads, dequant scaling, and int8-specific shared-memory packing. FlashAttention's fp16/bf16 SM80 path uses fp16/bf16 QK MMA and has no per-tile dequant scales. The int8 path has lower Q/K bandwidth and faster QK math, but pays extra scalar conversion/scale work and can become more sensitive to score-fragment size, softmax, denominator accumulation, and PV scheduling. That is why larger FA-style K tiles or Q-in-register caching need separate int8 benchmarks instead of direct adoption.

### Denominator strategy is int8-path specific

FlashAttention tracks row sums as part of its online softmax state and normalizes the output at the end. The CUTLASS qattn path similarly keeps online `m/d`, but it accumulates the denominator over dequantized fp32 scores before converting probabilities to fp16 for PV. A tensor-core row-sum variant was tested and was slower because it added HMMA work without improving occupancy. Keeping CUDA-core denominator accumulation is therefore a justified int8/fp32-PV choice until profiling on another architecture or schedule says otherwise.

## Completed Work

### Kept Changes

- K/V software pipeline: restored the CUDA-style staged preload/middle/second-last/last schedule. This closed most of the original large-sequence gap.
- CUDA-core denominator: kept after tensor-core denominator testing showed slower runtime on the local sm86 target.
- Block config expansion: added `(128,32,32,32)` and `(128,64,16,64)` after fixing dynamic shared-memory sizing to `max(Q+K+V, O)`.
- Q-tile warp sync: kept replacing the post-Q-load CTA barrier with `__syncwarp()`. Standard median A/B showed `22` wins, `2` tiny losses, and average CUTLASS delta `-2.24%`.
- Fixed Q32/K64 scale path: head dimensions 64 and 128 consume the same flat per-block artifact. Q scale indexing follows Q32 row ownership, K scale indexing follows K64 column ownership, and neither depends on the selected CTA/warp execution tile. The legacy per-thread scale view, runtime scale mode, and warp-broadcast load path have been removed from CUTLASS.

### Completed but Reverted Experiments

- K/V full-tile boundary fast path: skipped predicated final-tile K/V loads and final mask when `kv_len % CTA_K == 0`. Reverted because it averaged roughly flat (`+0.06%`) and lost in `15/24` standard cases.
- Fused denominator accumulation: folded `accumulate_d` into score exponentiation. Correctness passed, but median timing regressed (`+1.34%`, `17` losses, `6` wins, `1` tie), likely from extra register/live-range pressure and changed scheduling.
- Earlier next-K prefetch: moved next-K `cp.async` before score postprocessing. Reverted despite one large outlier win because it won only `7/24` cases and regressed multiple long-sequence configs.
- Q/output full-tile fast path: skipped predicated Q load and output bounds branch when `qo_len % CTA_Q == 0`. Reverted because the average was flat (`+0.01%`) with mixed results and a large short-sequence regression.
- Compile-time Q/K/V full-tile specialization: added a separate `FullTiles=true` instantiation selected when both `qo_len % CTA_Q == 0` and `kv_len % CTA_K == 0`, removing Q/K/V load predicates, final K mask, and output bounds checks from that binary. Correctness passed (`pytest -q tests/test_sageattn_cutlass.py`, `18 passed` at the time), but timing was mixed and not robust: a 12-case historical centered-K NHD main-config run showed `7` wins, `5` losses, and average `-1.81%` only because of one large 4096x64 outlier. A repeat focused 4096x64 run was flat/slower (`1.5703 ms` vs earlier best `1.2626 ms`). Reverted.
- Direct final K-mask base: replaced loop-carried `K_idx_lane_base += CTA_K` bookkeeping with a direct final-tile base expression in this non-causal kernel. Correctness passed, but focused timing was tied (`1.2640 ms` vs `1.2626 ms`) and the 12-case main-config run had `5` wins, `7` losses. The average looked faster only because of the same noisy 4096x64 case. Reverted.
- Tail-tile dequant-scale movement: removed per-element `* dequant_scale` from the second-last and last score-conversion loops and passed `original_sm_scale * dequant_scale` into `update_mdo`, matching the middle-loop math. Correctness passed, but immediate A/B on the 12-case historical centered-K NHD main-config grid regressed on average (`+0.49%`, `5` wins, `6` losses, `1` tie), despite an earlier non-paired run looking mildly favorable. Reverted.
- Extra `(64,32,32,32)` config: compiled and exposed the smaller-CTA candidate, and expanded the focused launch test (`19 passed`). The standard benchmark used a temporary CUTLASS-only timing script for this config. It was slower than existing configs across nearly all standard shapes (e.g. `head_dim=128, seq_len=2048, heads=16` at `0.7086 ms` vs existing best around `0.606-0.629 ms`. `head_dim=64, seq_len=8192, heads=16` at `5.79 ms` vs existing best around `4.85-4.96 ms`). Reverted to avoid extra compile cost.
- `num_kv_groups == 1` specialization: added a `KvGroupsOne` template and launcher dispatch to remove runtime grouped-query divisions for standard MHA. Correctness passed, but a paired 8-case long-sequence grid regressed on average (`+0.30%`, `2` wins, `6` losses), with notable losses for `head_dim=128`. Reverted because the extra binary variants were not justified.
- Direct final K/V load predicate base: computed final predicated K/V preload base indices once and removed loop-carried `K_load_idx_lane_base += CTA_K` / `V_load_idx_lane_base += CTA_K` updates. Correctness passed, but paired long-sequence timing regressed (`+1.74%`, `2` wins, `6` losses). Reverted.
- Mixed NHD input with HND-packed Q/K quantized storage: Triton emitted Q/K int8 as `(B,H,N,D)` while V/O stayed NHD, and the CUTLASS launcher accepted a mixed-layout code. Correctness was bit-identical, but paired kernel-only timing was noise-level: first 12-case NHD grid averaged `-2.27%` only due to a large outlier, while the repeat averaged just `-0.18%` with several losses and per-case stddevs larger than the median deltas. Reverted because the speedup did not justify the extra quantizer/launcher complexity.
- Cached Q fragments in registers: added a static kernel variant that preloaded all Q `ldmatrix` fragments for `head_dim=64` and reused them across K tiles. Correctness passed, but the 6-case `head_dim=64`, config `(128,64,32,64)` grid regressed on average (`+4.43%`), with significant losses at `(heads=16, seq=4096)` (`+19.79%`, far above stddev) and `(heads=32, seq=8192)` (`+7.64%`). The only wins were within stddev/noise. Reverted.
- Direct output epilogue: added a static variant that stores normalized register fragments directly to global memory and drops O shared-memory staging from dynamic-smem sizing. Correctness passed. The main-config-only run had one large local win, but the full supported-config run showed it was not useful under best-config selection: per-config average regressed (`+0.91%`), best-config-per-shape regressed on average (`+0.61%`), and the only significant best-shape delta was a loss. Some `head_dim=64, seq=8192` non-default configs won significantly, but they did not beat the existing best baseline config for those shapes. Reverted.
- K-only HND quantized storage for NHD inputs: Triton emitted K int8 as `(B,H,N,D)` while Q/V/O stayed NHD, and CUTLASS accepted a mixed K-HND layout code. Correctness was bit-identical, but the all-config paired kernel-only grid did not justify the extra layout plumbing: per-config average was `-0.78%`, but best-config average was only `-0.27%` with two significant best-shape wins and two significant best-shape losses (`heads=32, head_dim=64, seq=8192` and `heads=32, head_dim=128, seq=4096`). Reverted.
- Fused K-swizzled producer layout: adapted the Nunchaku-style lane-packed swizzle idea so Triton emitted K int8 in the same row-dependent 128-bit pack order as the CUTLASS shared-memory tile, and CUTLASS consumed it through a pre-swizzled K load path. Correctness was bit-identical, but the speedup did not justify the extra Triton quantizer, layout code, launcher branch, and kernel variant: all-config per-config average was `-0.98%`, while best-config swizzled-only average was only `-0.26%` with one significant best-shape loss (`heads=32, head_dim=128, seq=8192`). Default config was mixed (`+0.44%` average). Reverted.
- Tensor-core denominator: reverted because target NCU slowed from `1.5434 ms` to `1.6634 ms` and HMMA instructions increased by `12.5%`.
- Local inline-PTX MMA wrappers: reverted to CuTe wrappers for maintainability. Compact NCU did not show a meaningful resource or tensor-op difference.

## Remaining Work

### Larger Remaining Directions

- Register-pressure redesign: compact NCU already reports `255` registers/thread, so optimizations that lengthen `RS`/`RO` live ranges often regress. Larger redesigns should consider reducing live score/output fragments, especially for `head_dim=128`, even if that adds shared-memory traffic.
- Architecture-level config expansion: test configs that are not constrained to mirror the hand CUDA kernel only when profiling or autotune data justifies the compile cost. `(64,32,32,32)` was tested and reverted.
- Split-K / split-sequence forward: for low grid counts or long `kv_len`, split the K dimension across CTAs and combine partial online-softmax states. FlashAttention has split forward variants. This likely needs extra partial-output/LSE storage and a reduction kernel, so it is a larger feature rather than a micro-optimization.
- Selective K/V double buffering: try only if NCU shows exposed memory latency. It may help `head_dim=64`, but for `head_dim=128` extra shared memory can reduce occupancy or worsen register pressure.
- Q-in-register / shared-Q-memory tradeoff: FlashAttention exposes `Is_Q_in_regs` and `Share_Q_K_smem` traits. A larger CUTLASS redesign could use similar variants to reduce shared-memory footprint or syncs, but should be benchmarked separately by head dimension.
- sm80/sm86 warp-specialization variants: do not copy FlashAttention-3's Hopper producer/consumer design directly. FA3 benefits from sm90 TMA, WGMMA, warpgroup barriers, and per-warpgroup register allocation. This kernel targets sm80-family GPUs and launches only math-partition warps with cooperative `cp.async` K/V staging. A dedicated producer warp would reduce math warps in the default 4-warp config and is especially risky with `255` registers/thread. Test only if NCU shows `cp.async` waits or long-scoreboard stalls dominate while tensor-pipe utilization is low.
- Coarser per-MMA-tile scales: test per-16-row or per-MMA-tile Q/K scales only if accuracy targets allow changing the quantization contract.

### Deferred or Revisit Only With New Evidence

- Compile-time even-shape/full-tile specialization unless a major schedule/layout rewrite or NCU profile identifies predicate/control overhead as a bottleneck.
- Additional block configs unless profiling or autotune data justifies the compile-time cost.
- Retrying quantized storage/layout variants only with new profiling evidence or a larger CuTe tiled-copy rewrite that changes the consumer contract.
- Retrying tensor-core denominator only on a different architecture or after a major schedule change.
- Retrying core MMA-shape changes or Hopper-style warp specialization without a broader schedule/layout rewrite and profiling evidence that the MMA granularity or cooperative `cp.async` pipeline is the bottleneck.

## Benchmark Workflow

Use fixed raw-K shapes. Treat end-to-end timing against unconstrained FlashAttention dispatch as the primary product result and prequantized kernel-only timing as an optimization diagnostic.

Target shape set:

```text
batch_size=1, layout in {NHD, HND}, fp16, non-causal, smooth_k=False
num_heads in {16, 32}
seq_len in {2048, 4096, 8192}
head_dim in {64, 128}
configs in supported CUTLASS configs
```

For each code change:

1. Build:

```text
python setup.py build_ext --inplace
```

2. Correctness:

```text
pytest tests/test_sageattn_cutlass.py tests/test_sageattn_cutlass_compile.py
```

3. Focused kernel-only benchmark:

```text
python bench/bench_qattn_cutlass.py --mode kernel-only --layout NHD --seq-lens 4096 --num-heads 16 --head-dims 64 --repeats 50 --warmup 10
```

4. Broader benchmark if the focused shape improves:

```text
foreach ($layout in @('NHD', 'HND')) {
  python bench/bench_qattn_cutlass.py --mode all --layout $layout --seq-lens 4096 8192 --num-heads 16 32 --head-dims 64 128 --repeats 100 --warmup 25 --csv "build/bench_qattn_cutlass_rawk_${layout}_4096_8192.csv"
}
```

5. Nsight Compute after meaningful performance changes.

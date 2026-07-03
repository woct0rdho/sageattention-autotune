# QAttn CUTLASS Forward Kernel Plan

## Scope

- Target architectures: sm80 (sm86).
- Inputs: fp16 `q`, `k`, and `v`. bf16 is deferred.
- Non-causal only.
- `pv_accum_dtype="fp32"` only. No instruction-buffer PV variant.
- Optional LSE output for backward consumers.
- Fixed-length dense tensors only.
- Supported tensor layouts: `HND` and `NHD` via the same stride handling as the existing CUDA qattn path.
- Head dims: `64` and `128` only.
- Block configs:
  - `(blk_q=128, blk_k=64, warp_q=32, warp_k=64)`
  - `(blk_q= 64, blk_k=64, warp_q=32, warp_k=64)`
  - `(blk_q=128, blk_k=32, warp_q=32, warp_k=32)`
  - `(blk_q=128, blk_k=64, warp_q=16, warp_k=64)`
- Exposed through `sageattention.cutlass_attn.sageattn_qk_int8_pv_fp16_cutlass` with CUDA-style eager and `torch.compile` autotune plumbing, but not selected as the default `sageattn` backend.

## Current Implementation

The wrapper deliberately reuses the existing Triton `per_thread_int8` quantization layout instead of introducing a new quantized layout. The wrapper does:

```text
q, k -> per_thread_int8 -> q_int8, q_scale, k_int8, k_scale
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

The kernel is self-contained apart from library headers. Shared, global, and register storage are exposed through CuTe tensors where doing so preserves the existing schedule and generated work. Q/K/V global-to-shared loads and O shared-to-global stores use CuTe `TiledCopy` partitioning, full-problem identity-coordinate tensors tiled and partitioned with the data, lazy predicate tensors, and `copy_if`. Predicated global-to-shared loads use CuTe's `SM80_CP_ASYNC_CACHEGLOBAL_ZFILL` atom so out-of-bounds boundary lanes are initialized to zero. Shared-memory tiles use stride-derived CuTe swizzle layouts and CuTe `ldmatrix` copy atoms. CTA/warp row selection uses local CuTe layout helpers and `local_tile` over full gmem matrices, while Q/K scale indexing and lane subcoordinates are centralized in small CuTe layout helpers instead of open-coded warp/lane arithmetic.

The core MMA instructions execute the same logical QK `m16n16k32` and PV `m16n16k16` work. `compute_int_qk` and `compute_fp16_sv` route through local CuTe `TiledMMA` helpers and `cute::gemm` over CuTe-owned register fragments. Q/K/V smem-to-register loads allocate fragments with `thr_mma.partition_fragment_A/B` and retile the relevant `TiledCopy` slices into those fragments. Tail masking, online-softmax state updates, denominator accumulation, output normalization, and fp16 probability handoff use CuTe `transform`, `for_each`, and `reduce` over tensor slices or coordinate tensors instead of open-coded element loops. The probability handoff aliases `score_f32` with `prob.layout()` and runs the vectorized f32-to-f16 converter once. The output store uses shared-memory staging, with a local CuTe C-copy source layout for `make_tiled_copy_C`. Its stride is derived from the accumulator layout because the direct nested PV C layout has a different register traversal order.

## Benchmark Snapshot

```text
python bench/bench_qattn_cutlass.py --mode kernel-only --layout NHD --seq-lens 2048 4096 8192 --num-heads 16 32 --head-dims 64 128 --repeats 50 --warmup 10 --smooth-k --csv $Env:TEMP/qattn_cutlass_kernel_baseline.csv
```

The script measures two levels:
- `end_to_end`: Python wrapper including Q/K quantization, output allocation, and attention kernel.
- `kernel_only`: prequantized Q/K and preallocated output, measuring only the existing CUDA attention kernel versus the new CUTLASS attention kernel.

The kernel-only result is the more reliable optimization signal because both paths share the same quantization producer. The benchmark reports median time as the primary `cuda_ms` / `cutlass_ms` columns, with mean and stdev retained as context in `cuda_mean_ms` / `cutlass_mean_ms` and `*_stdev_ms`.

The benchmark below predates the latest CuTe-style helper cleanup, including the softmax loop-removal pass. No new benchmark has been run for those migrations because the validation policy is correctness-only unless benchmarking is requested.

Latest standard kernel-only run:

```text
python bench/bench_qattn_cutlass.py --mode kernel-only --layout NHD --seq-lens 2048 4096 8192 --num-heads 16 32 --head-dims 64 128 --repeats 50 --warmup 10 --smooth-k --csv $Env:TEMP/qattn_cutlass_after_cute_helpers.csv
```

Best-config median summary from `$Env:TEMP/qattn_cutlass_after_cute_helpers.csv`:

| Num heads | Head dim | Seq len | Best CUDA median ms | Best CUTLASS median ms | Best CUTLASS config | CUTLASS / CUDA |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 64 | 2048 | 0.4639 | 0.4567 | `(128,64,32,64)` | 0.985 |
| 16 | 64 | 4096 | 1.2257 | 1.2114 | `(128,64,32,64)` | 0.988 |
| 16 | 64 | 8192 | 4.8731 | 4.8461 | `(128,64,32,64)` | 0.994 |
| 16 | 128 | 2048 | 0.5908 | 0.6036 | `(128,64,32,64)` | 1.022 |
| 16 | 128 | 4096 | 2.1683 | 2.1422 | `(128,64,32,64)` | 0.988 |
| 16 | 128 | 8192 | 8.5350 | 8.3379 | `(128,64,32,64)` | 0.977 |
| 32 | 64 | 2048 | 0.6041 | 0.6025 | `(128,64,32,64)` | 0.997 |
| 32 | 64 | 4096 | 2.3843 | 2.3244 | `(128,64,32,64)` | 0.975 |
| 32 | 64 | 8192 | 9.4387 | 9.1587 | `(128,64,32,64)` | 0.970 |
| 32 | 128 | 2048 | 1.1335 | 1.1356 | `(128,64,32,64)` | 1.002 |
| 32 | 128 | 4096 | 4.3345 | 4.2726 | `(128,64,32,64)` | 0.986 |
| 32 | 128 | 8192 | 17.1561 | 16.8873 | `(128,64,32,64)` | 0.984 |

Average best CUTLASS / best CUDA across this grid: `0.989`.

Notes:
- Full per-config data is kept in the CSV rather than copied into this plan.
- Timing variance on the local laptop GPU is visible, so repeat A/B runs and NCU profiles should guide major decisions.

Summary:
- The CUTLASS path is at parity or slightly faster by best config across the standard `smooth_k=True` kernel-only grid.
- Config choice matters, especially for `head_dim=64` short sequences.
- Further micro-optimizations should be kept only when the median A/B result is robust across `seq_len in {2048,4096,8192}` and `head_dim in {64,128}`.

## Comparison with SageAttention CUDA Kernel

The CUTLASS kernel intentionally mirrors the CUDA kernel's main architecture for the overlapping fp16, non-causal, fp32-PV path: same CTA/warp tiling choices, same `m16n16k32` QK and `m16n16k16` PV tensor-core shapes, same single shared-memory Q/K/V staging buffer overlaid with the O staging buffer, same K/V software pipeline, same warp-broadcasted Q/K scales, same CUDA-core denominator strategy, and the same shared-memory epilogue pattern. The remaining differences below are architectural or scheduling choices that can affect performance, not just CuTe-vs-CUDA code style.

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

Several differences are justified by int8 Q/K rather than missing FlashAttention optimizations. The qattn path uses `m16n16k32` int8 QK MMA, int32 score accumulators, per-thread Q/K scale loads, dequant scaling, and int8-specific shared-memory packing. FlashAttention's fp16/bf16 SM80 path uses fp16/bf16 QK MMA and has no per-tile dequant scales. The int8 path has lower Q/K bandwidth and faster QK math, but pays extra scalar conversion/scale work and can become more sensitive to score-fragment size, softmax, denominator accumulation, and PV scheduling. That is why larger FA-style K tiles or Q-in-register caching need separate int8 benchmarks instead of direct adoption.

### Denominator strategy is int8-path specific

FlashAttention tracks row sums as part of its online softmax state and normalizes the output at the end. The CUTLASS qattn path similarly keeps online `m/d`, but it accumulates the denominator over dequantized fp32 scores before converting probabilities to fp16 for PV. A tensor-core row-sum variant was tested and was slower because it added HMMA work without improving occupancy. Keeping CUDA-core denominator accumulation is therefore a justified int8/fp32-PV choice until profiling on another architecture or schedule says otherwise.

## Completed Work

### Kept Changes

- K/V software pipeline: restored the CUDA-style staged preload/middle/second-last/last schedule. This closed most of the original large-sequence gap.
- CUDA-core denominator: kept after tensor-core denominator testing showed slower runtime on the local sm86 target.
- Block config expansion: added `(128,32,32,32)` and `(128,64,16,64)` after fixing dynamic shared-memory sizing to `max(Q+K+V, O)`.
- Q-tile warp sync: kept replacing the post-Q-load CTA barrier with `__syncwarp()`. Standard median A/B showed `22` wins, `2` tiny losses, and average CUTLASS delta `-2.24%`.
- Scale-load broadcast layout: kept a static scale-load policy where Q scale is loaded once per 4-lane row group and K scale once per 4-lane K group, then broadcast with warp shuffles. Correctness passed (`pytest -q tests/test_sageattn_cutlass.py`, `32 passed` at the time). The all-config NHD grid was mixed (`+1.01%` average over every config), but the default `(128,64,32,64)` config had significant wins on several target shapes and a selective default-config policy would improve best-config median by `-0.74%` with no significant losses. Kept always-on per the quant-layout exploration policy. Revisit with an explicit toggle only if later data shows decisive regressions.

### Completed but Reverted Experiments

- K/V full-tile boundary fast path: skipped predicated final-tile K/V loads and final mask when `kv_len % CTA_K == 0`. Reverted because it averaged roughly flat (`+0.06%`) and lost in `15/24` standard cases.
- Fused denominator accumulation: folded `accumulate_d` into score exponentiation. Correctness passed, but median timing regressed (`+1.34%`, `17` losses, `6` wins, `1` tie), likely from extra register/live-range pressure and changed scheduling.
- Earlier next-K prefetch: moved next-K `cp.async` before score postprocessing. Reverted despite one large outlier win because it won only `7/24` cases and regressed multiple long-sequence configs.
- Q/output full-tile fast path: skipped predicated Q load and output bounds branch when `qo_len % CTA_Q == 0`. Reverted because the average was flat (`+0.01%`) with mixed results and a large short-sequence regression.
- Compile-time Q/K/V full-tile specialization: added a separate `FullTiles=true` instantiation selected when both `qo_len % CTA_Q == 0` and `kv_len % CTA_K == 0`, removing Q/K/V load predicates, final K mask, and output bounds checks from that binary. Correctness passed (`pytest -q tests/test_sageattn_cutlass.py`, `18 passed` at the time), but timing was mixed and not robust: a 12-case `smooth_k=True` NHD main-config run showed `7` wins, `5` losses, and average `-1.81%` only because of one large 4096x64 outlier. A repeat focused 4096x64 run was flat/slower (`1.5703 ms` vs earlier best `1.2626 ms`). Reverted.
- Direct final K-mask base: replaced loop-carried `K_idx_lane_base += CTA_K` bookkeeping with a direct final-tile base expression in this non-causal kernel. Correctness passed, but focused timing was tied (`1.2640 ms` vs `1.2626 ms`) and the 12-case main-config run had `5` wins, `7` losses. The average looked faster only because of the same noisy 4096x64 case. Reverted.
- Tail-tile dequant-scale movement: removed per-element `* dequant_scale` from the second-last and last score-conversion loops and passed `original_sm_scale * dequant_scale` into `update_mdo`, matching the middle-loop math. Correctness passed, but immediate A/B on the 12-case `smooth_k=True` NHD main-config grid regressed on average (`+0.49%`, `5` wins, `6` losses, `1` tie), despite an earlier non-paired run looking mildly favorable. Reverted.
- Extra `(64,32,32,32)` config: compiled and exposed the smaller-CTA candidate, and expanded the focused launch test (`19 passed`). The standard benchmark could not use the CUDA comparison path because the existing CUDA kernel does not support this config, so a temporary CUTLASS-only timing script was used. It was slower than existing configs across nearly all standard shapes (e.g. `head_dim=128, seq_len=2048, heads=16` at `0.7086 ms` vs existing best around `0.606-0.629 ms`. `head_dim=64, seq_len=8192, heads=16` at `5.79 ms` vs existing best around `4.85-4.96 ms`). Reverted to avoid extra compile cost.
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

Use fixed shapes and prequantized kernel-only comparisons while optimizing.

Target shape set:

```text
batch_size=1, layout=NHD, fp16, non-causal, smooth_k=True
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
python bench/bench_qattn_cutlass.py --mode kernel-only --layout NHD --seq-lens 4096 --num-heads 16 --head-dims 64 --repeats 50 --warmup 10 --smooth-k
```

4. Broader benchmark if the focused shape improves:

```text
python bench/bench_qattn_cutlass.py --mode kernel-only --layout NHD --seq-lens 2048 4096 8192 --num-heads 16 32 --head-dims 64 128 --repeats 50 --warmup 10 --smooth-k --csv $Env:TEMP/qattn_cutlass_baseline.csv
```

5. Nsight Compute after meaningful performance changes.

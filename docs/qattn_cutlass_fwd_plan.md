# QAttn CUTLASS Forward Kernel Plan

## Goal

Implement and optimize a standalone CUTLASS/CuTe C++ forward kernel for SageAttention's quantized attention path, starting from the existing `qk_int8_sv_f16` CUDA kernel contract and narrowing scope enough to iterate quickly. The first target is correctness parity and competitive kernel-only speed against the hand-written CUDA forward kernel on SM80-family GPUs.

## Scope

- Target architectures: SM80-family only (`sm80` and `sm86`).
- Inputs: fp16 `q`, `k`, and `v` only. bf16 is deferred.
- Attention mode: non-causal only.
- Value path: `pv_accum_dtype="fp32"` only.
- No instruction-buffer PV variant.
- Optional LSE output for backward/training consumers. Disabled by default for inference timing.
- Fixed-length dense tensors only.
- Supported tensor layouts: `HND` and `NHD` via the same stride handling as the existing CUDA qattn path.
- Supported padded head dims: `64` and `128` only.
- Block configs:
  - `(blk_q=128, blk_k=64, warp_q=32, warp_k=64)`
  - `(blk_q=64, blk_k=64, warp_q=32, warp_k=64)`
  - `(blk_q=128, blk_k=32, warp_q=32, warp_k=32)`
  - `(blk_q=128, blk_k=64, warp_q=16, warp_k=64)`
- Exposed through `sageattention.cutlass_attn.sageattn_qk_int8_pv_fp16_cutlass` with CUDA-style eager and `torch.compile` autotune plumbing, but not selected as the default `sageattn` backend.

## Current Implementation

Files:
- `csrc/qattn_cutlass/qk_int8_sv_f16_kernel_cutlass_sm80.cuh`
- `csrc/qattn_cutlass/qk_int8_sv_f16_launch_cutlass_sm80.cuh`
- `csrc/qattn_cutlass/qk_int8_sv_f16_accum_f32_attn_cutlass.cu`
- `csrc/qattn_cutlass/pybind_sm80.cpp`
- `sageattention/cutlass_compile.py`
- `sageattention/cutlass_autotune.py`
- `sageattention/cutlass_attn.py`
- `tests/test_sageattn_cutlass.py`
- `tests/test_sageattn_cutlass_compile.py`

The wrapper deliberately reuses the existing Triton `per_thread_int8` quantization layout instead of introducing a new quantized layout. The wrapper does:

```text
q, k -> per_thread_int8 -> q_int8, q_scale, k_int8, k_scale
q_int8, k_int8, v, scales -> CUTLASS/CuTe SM80 forward kernel -> output[, lse]
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

The kernel uses CuTe SM80 MMA wrappers for the core MMA instructions. Compact NCU captures showed the CuTe-wrapper and project-local inline-PTX-wrapper variants have identical tensor-op/resource counts for the target shape, so the implementation keeps the idiomatic CUTLASS/CuTe MMA path to preserve room for later CuTe-level optimization.

The shared-memory tile layouts, `ldmatrix` loading, online-softmax helpers, and output store layout are intentionally close to the CUDA kernel to reduce correctness risk.

## Correctness Status

Focused test coverage exists in `tests/test_sageattn_cutlass.py` and `tests/test_sageattn_cutlass_compile.py`:
- fp16 only.
- `head_dim in {64, 128}`.
- `tensor_layout in {HND, NHD}`.
- `smooth_k in {False, True}`.
- all supported block configs selected through the same `_valid_configs` pattern as the CUDA tests.
- skips devices outside `sm80`/`sm86`.
- optional LSE output compared numerically against FlashAttention with `rtol=3e-3`, `atol=7e-2`, chosen from CUDA/Triton/CUTLASS quantized-forward LSE samples where causal rows were the worst case.

Local result:

```text
pytest -q tests/test_sageattn_cutlass.py tests/test_sageattn_cutlass_compile.py
39 passed
```

Typical CUTLASS-vs-CUDA output error in the benchmark is small:

```text
rel_err ~= 7e-5
max_abs <= 2.4e-4
```

## Benchmark Snapshot

```text
python bench/bench_qattn_cutlass.py --mode kernel-only --layout NHD --seq-lens 2048 4096 8192 --num-heads 16 32 --head-dims 64 128 --repeats 50 --warmup 10 --smooth-k --csv $Env:TEMP/qattn_cutlass_kernel_baseline.csv
```

The script measures two levels:
- `end_to_end`: Python wrapper including Q/K quantization, output allocation, and attention kernel.
- `kernel_only`: prequantized Q/K and preallocated output, measuring only the existing CUDA attention kernel versus the new CUTLASS attention kernel.

The kernel-only result is the more reliable optimization signal because both paths share the same quantization producer. The benchmark reports median time as the primary `cuda_ms` / `cutlass_ms` columns, with mean and stdev retained as context in `cuda_mean_ms` / `cutlass_mean_ms` and `*_stdev_ms`.

### Kept Variant, `smooth_k=True`, NHD

Kept source includes:
- staged K/V software pipeline;
- CuTe SM80 MMA wrappers;
- CUDA-core denominator accumulation;
- four supported block configs with fixed dynamic shared-memory sizing;
- Q-tile post-load `__syncwarp()` instead of the earlier CTA-wide barrier;
- warp-broadcasted Q/K scale loads.

Latest standard kernel-only run:

```text
python bench/bench_qattn_cutlass.py --mode kernel-only --layout NHD --seq-lens 2048 4096 8192 --num-heads 16 32 --head-dims 64 128 --repeats 50 --warmup 10 --smooth-k --csv $Env:TEMP/qattn_cutlass_current_baseline_h16_h32.csv
```

Best-config median summary from `$Env:TEMP/qattn_cutlass_current_baseline_h16_h32.csv`:

| Num heads | Head dim | Seq len | Best CUDA median ms | Best CUTLASS median ms | Best CUTLASS config | CUTLASS / CUDA |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 64 | 2048 | 0.4639 | 0.4526 | `(128,64,32,64)` | 0.976 |
| 16 | 64 | 4096 | 1.2657 | 1.2543 | `(128,64,32,64)` | 0.991 |
| 16 | 64 | 8192 | 4.9480 | 4.9591 | `(128,64,32,64)` | 1.002 |
| 16 | 128 | 2048 | 0.6088 | 0.6062 | `(128,32,32,32)` | 0.996 |
| 16 | 128 | 4096 | 2.2384 | 2.1714 | `(128,64,32,64)` | 0.970 |
| 16 | 128 | 8192 | 8.6420 | 8.5878 | `(128,64,32,64)` | 0.994 |
| 32 | 64 | 2048 | 0.6409 | 0.6252 | `(128,64,32,64)` | 0.975 |
| 32 | 64 | 4096 | 2.4371 | 2.4141 | `(128,64,32,64)` | 0.991 |
| 32 | 64 | 8192 | 9.7213 | 9.4495 | `(128,64,32,64)` | 0.972 |
| 32 | 128 | 2048 | 1.1459 | 1.1469 | `(128,64,32,64)` | 1.001 |
| 32 | 128 | 4096 | 4.4754 | 4.3888 | `(128,64,32,64)` | 0.981 |
| 32 | 128 | 8192 | 17.5734 | 17.2662 | `(128,64,32,64)` | 0.983 |

Average best CUTLASS / best CUDA across this grid: `0.986`.

Notes:
- Full per-config data is kept in the CSV rather than copied into this plan.
- The benchmark reports medians in `cuda_ms` / `cutlass_ms`. Use those columns for keep/revert decisions.
- Timing variance on the local laptop GPU is visible, so repeat A/B runs and NCU profiles should guide major decisions.
- Correctness remains at `rel_err ~= 6e-5` to `1.2e-4`, depending on shape/config.

Summary:
- Correctness is good.
- The CUTLASS path is at parity or slightly faster by best config across the standard `smooth_k=True` kernel-only grid.
- Config choice matters, especially for `head_dim=64` short sequences.
- Further micro-optimizations should be kept only when the median A/B result is robust across `seq_len in {2048,4096,8192}` and `head_dim in {64,128}`.

## Key Differences vs Existing CUDA Kernel

### K/V software pipeline

The existing CUDA kernel overlaps compute with prefetch of the next K/V tiles. It preloads K and V before the main loop, then uses `cp_async::wait_group<1>` while scheduling the next tile's K/V loads during current-tile compute.

The initial CUTLASS kernel used a simpler load-wait-compute loop, which left more memory latency exposed. Phase 2 restored the CUDA-style staged loop in the CUTLASS kernel, and the kernel-only benchmark shows near parity or better timings on the local sm86 target.

### Denominator accumulation uses CUDA cores

The existing CUDA `pv_accum_dtype="fp32"` path uses tensor-core row-sum accumulation on the fp16 probability tile for the denominator. The CUTLASS path was tested both ways. On the local sm86 target, tensor-core denominator accumulation increased HMMA instructions by `12.5%` and slowed the profiled target kernel by about `7.8%`, with no occupancy/resource improvement. The CUTLASS path therefore keeps CUDA-core denominator accumulation:

```text
accumulate_d<ComputeUnit::kCudaCore>(RS_f32, d)
```

This leaves some scalar/shuffle work in normalization, but it is faster for the measured fp32-accum path.

### Boundary handling remains generic

The CUTLASS staged loop uses unpredicated middle-tile K/V loads and keeps predicated boundary loads plus final score masking for correctness on arbitrary sequence lengths. Full-tile fast paths for K/V and Q/output boundaries were tested on the standard divisible sequence lengths, but their median timing was mixed and they were reverted. Keep the generic boundary path unless profiling shows predicate/control overhead is a primary remaining bottleneck.

### MMA wrapper choice

The actual MMA instructions are equivalent in intent:
- int8 `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
- fp16 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`

Compact NCU captures showed identical registers, shared memory, occupancy, IMMA count, and HMMA count for the CuTe-wrapper and local-wrapper CUDA-denominator variants. Runtime varied slightly run-to-run, but the implementation intentionally keeps the idiomatic CuTe MMA wrappers to make later CuTe/CUTLASS optimization easier.

### MMA shape choice

Do not change the core SM80 MMA shapes as a standalone optimization. The logical int8 `m16n16k32` QK wrapper is built from two native `m16n8k32` instructions, and the logical fp16 `m16n16k16` PV wrapper is built from two native `m16n8k16` instructions. FlashAttention's SM80 fp16 path uses the same `SM80_16x8x16_F32F16F16F32_TN` atom and gets most of its speed from macro-tiling, copy layout, Q-in-register/shared-memory tradeoffs, and pipeline structure rather than a different MMA atom.

Smaller alternatives such as int8 `m16n8k16` or fp16 `m16n8k8` would mostly increase instruction count for the same score/output tile. Larger logical tiles would compose more native `m16n8` atoms and likely worsen the already-high score/output live ranges. The MoonMath CDNA3 lesson applies at the schedule level: MMA shape can determine what surrounding optimizations are possible, but on SM80 this kernel is already using the small high-throughput atom. Revisit only as part of a deliberate score-fragment lifetime or layout redesign, not as a raw tensor-core throughput change.

SageAttention's quantized path is not structurally identical to FlashAttention's fp16 QK path. QK is int8 with per-warp/tile dequant scales before online softmax, then PV consumes an fp16 probability tile. The `N=16` score tile naturally feeds the PV `K=16` HMMA path through `RS_32_to_16`. Changing MMA granularity would also change scale application, softmax layout, and score-to-PV conversion pressure.

### Block config coverage

The existing CUDA path supports more configs, including `(128,32,32,32)` and `(128,64,16,64)`. CUTLASS compiles those two configs as well. Enabling `(128,32,32,32)` exposed that the first CUTLASS launcher under-allocated dynamic shared memory for the final output staging tile. The launcher uses the same `max(Q+K+V, O)` sizing as the CUDA path.

### Profiling snapshot

Compact NCU captures for `seq_len=4096`, `head_dim=64`, `smooth_k=True`, config `(128,64,32,64)` showed:

| Metric | CUDA | CUTLASS CuTe + CUDA denom | CUTLASS CuTe + tensor denom | CUTLASS local + CUDA denom |
|---|---:|---:|---:|---:|
| Time, ns | 1.5717e6 | 1.5434e6 | 1.6634e6 | 1.5438e6 |
| Registers/thread | 255 | 255 | 255 | 255 |
| Dynamic smem/block | 20480 B | 20480 B | 20480 B | 20480 B |
| Occupancy limit, registers | 2 blocks | 2 blocks | 2 blocks | 2 blocks |
| Active warps | 15.91% | 15.83% | 15.88% | 15.83% |
| HMMA instructions | 9.437M | 8.389M | 9.437M | 8.389M |
| IMMA instructions | 4.194M | 4.194M | 4.194M | 4.194M |

Interpretation:
- Register pressure and shared memory are identical across the compared attention kernels.
- Tensor-core denominator adds the same HMMA row-sum work as CUDA but is slower in this CUTLASS schedule.
- CuTe-wrapper vs local-wrapper differences are not visible in these compact resource counters. Keep CuTe wrappers for maintainability and future optimization unless deeper SASS scheduling analysis proves they are a blocker.

## Completed Work

This section records optimization work that has already been tried. Kept changes are part of the source. Reverted experiments should not be retried without new profiling evidence or a larger schedule/layout change.

### Kept Changes

- K/V software pipeline: restored the CUDA-style staged preload/middle/second-last/last schedule. This closed most of the original large-sequence gap.
- CUDA-core denominator: kept after tensor-core denominator testing showed slower runtime on the local sm86 target.
- CuTe MMA wrappers: kept after local inline-PTX wrappers showed identical compact NCU resource/tensor-op counts and no robust timing advantage.
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

## Remaining Optimization Work

### Near-Term Remaining Work

- Config/default selection: CUTLASS has the same eager/compile autotune plumbing shape as CUDA. Remaining work is collecting enough timing data to decide whether `(128,64,32,64)` should stay first/default or whether a different config ordering is better.

### Larger Remaining Directions

- Full CuTe tiled-copy/tiled-MMA rewrite: move from manual shared-memory offsets toward CuTe `TiledMMA`/`TiledCopy` traits. FlashAttention and CUTLASS's Ampere FA2 example use this structure to make layout, copy atom, output epilogue, and register-pipeline variants easier to test.
- Register-pressure redesign: compact NCU already reports `255` registers/thread, so optimizations that lengthen `RS`/`RO` live ranges often regress. Larger redesigns should consider reducing live score/output fragments, especially for `head_dim=128`, even if that adds shared-memory traffic.
- Architecture-level config expansion: test configs that are not constrained to mirror the hand CUDA kernel only when profiling or autotune data justifies the compile cost. `(64,32,32,32)` was tested and reverted.
- Split-K / split-sequence forward: for low grid counts or long `kv_len`, split the K dimension across CTAs and combine partial online-softmax states. FlashAttention has split forward variants. This likely needs extra partial-output/LSE storage and a reduction kernel, so it is a larger feature rather than a micro-optimization.
- Selective K/V double buffering: try only if NCU shows exposed memory latency. It may help `head_dim=64`, but for `head_dim=128` extra shared memory can reduce occupancy or worsen register pressure.
- Q-in-register / shared-Q-memory tradeoff: FlashAttention exposes `Is_Q_in_regs` and `Share_Q_K_smem` traits. A larger CUTLASS redesign could use similar variants to reduce shared-memory footprint or syncs, but should be benchmarked separately by head dimension.
- SM80/SM86 warp-specialization variants: do not copy FlashAttention-3's Hopper producer/consumer design directly. FA3 benefits from SM90 TMA, WGMMA, warpgroup barriers, and per-warpgroup register allocation. This kernel targets SM80-family GPUs and launches only math-partition warps with cooperative `cp.async` K/V staging. A dedicated producer warp would reduce math warps in the default 4-warp config and is especially risky with `255` registers/thread. Test only if NCU shows `cp.async` waits or long-scoreboard stalls dominate while tensor-pipe utilization is low.
- Coarser per-MMA-tile scales: test per-16-row or per-MMA-tile Q/K scales only if accuracy targets allow changing the quantization contract. This is no longer part of the layout-only optimization pass.

### Deferred or Revisit Only With New Evidence

- bf16 support, `head_dim=256`, causal support, and default-backend integration.
- Compile-time even-shape/full-tile specialization unless a major schedule/layout rewrite or NCU profile identifies predicate/control overhead as a bottleneck.
- Additional block configs unless profiling or autotune data justifies the compile-time cost.
- Retrying quantized storage/layout variants only with new profiling evidence or a larger CuTe tiled-copy rewrite that changes the consumer contract.
- Retrying tensor-core denominator only on a different architecture or after a major schedule change.
- Retrying core MMA-shape changes or Hopper-style warp specialization without a broader schedule/layout rewrite and profiling evidence that the MMA granularity or cooperative `cp.async` pipeline is the bottleneck.

## Benchmark Workflow

Use fixed shapes and prequantized kernel-only comparisons while optimizing.

Suggested standard shape set:

```text
batch=1, layout=NHD, fp16, non-causal
num_heads in {16, 32}
seq_len in {2048, 4096, 8192}
head_dim in {64, 128}
smooth_k in {False, True}
configs in supported CUTLASS configs
```

For each code change:

1. Build:

```text
python setup.py build_ext --inplace
```

2. Correctness:

```text
pytest -q tests/test_sageattn_cutlass.py tests/test_sageattn_cutlass_compile.py
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

## Acceptance Targets

Completed or maintained:
- Correctness is within tolerance for fp16, non-causal, dense `head_dim in {64,128}` coverage, including optional LSE against FlashAttention.
- The kept variant meets the short-term local parity target for the standard `smooth_k=True` kernel-only grid by best config.
- The compile matrix remains limited to four supported configs.

Remaining:
- Beat the existing CUDA kernel robustly for at least one target shape without regressing other target shapes by more than `3%`.
- Validate the CUTLASS config ordering/default policy with more timing data.
- Document stall/resource differences with Nsight Compute after a meaningful performance change.
- Decide whether the CUTLASS path should replace, complement, or simply inform the existing hand CUDA path.
- Decide whether Q/K quantization layout should stay per-thread or become CUTLASS-specific.
- Prepare a backward-kernel design only after forward performance is understood.

## Open Questions

- Which remaining gaps are due to scheduling/stalls rather than block config or denominator choices?
- Should CUTLASS autotune config ordering continue to prefer `(128,64,32,64)` first, given shape-specific alternatives occasionally win?
- After a larger schedule/layout rewrite, is there a reason to revisit even-shape predicate specialization?
- How far should the implementation move from CuTe MMA wrappers toward full CuTe-style layout/atom composition?
- Would split-K / split-sequence forward improve occupancy enough to pay for partial-state reduction?
- Can a different quantized layout benefit both forward and future backward, or would it fragment the implementation?

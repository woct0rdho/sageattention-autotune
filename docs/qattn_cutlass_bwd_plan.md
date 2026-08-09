# QAttn CUTLASS Backward Kernel Plan

## Current State

The maintained backward path targets native SM86, fp16, dense non-causal fixed-length attention with head dimension 64 and NHD/HND layouts. Its dS mirror uses one swizzled 16x64 int8 parent per N32 pair, with static 16x32 M-half views for dQ. Two explicitly configured specializations are generated:

```text
QBlock=32, KBlock=64, CTA M64xN128, eight warps (default)
QBlock=32, KBlock=64, CTA M64xN256, sixteen warps (experimental)
```

`_BWD_CONFIG` remains `(32, 64, 64, 128)`, while `_BWD_CONFIGS` also exposes `(32, 64, 64, 256)` for direct tests and benchmarks. There is no backward autotuner and the public wrapper continues to select N128. The old filename/module `sm80` is a legacy identifier; `setup.py` emits only `compute_86,sm_86`. Both configured paths use the fixed Q32/K64 metadata contract, the separable dS predictor, and the internal swizzled dQAccum workspace described below.

Both configured paths support raw and centered K. The public wrapper defaults to `smooth_k=True`, matching CUTLASS forward. Callers always provide the corrected natural-log LSE in the raw-score domain. For smoothing, Python recomputes the sequence K mean, quantizes `K - mean(K)`, and converts LSE to the centered-score domain with `LSE_centered = LSE_raw - softmax_scale * dot(Q, mean(K))`. The native kernel accumulates one exact reconstructed-dS row sum per query row, and Triton restores the missing dQ term `dS_sum * mean(K)` while inverting the swizzled dQ workspace. Raw K compiles out the row reduction and reuses `Delta` as an ignored operator placeholder, so it does not allocate or clear the padded dS-sum workspace.

The remainder of this plan is organized by contract, pipeline, correctness, final performance, bottlenecks, and completed or deferred work. The public forward still discards its quantized Q/K artifacts, so the backward timing is a reuse proxy rather than an integrated training benchmark; product and API limitations are collected under `Deferred`.

### Supported Contract

- Inputs: fp16 Q, K, V, output, and dOutput; float32 LSE.
- Layouts: NHD and HND.
- Attention: dense, non-causal, fixed length.
- K policy: `smooth_k=False/True`; LSE is natural-log and corrected to the raw-score domain at the public boundary.
- Head dimension: 64 after padding.
- Last dimension: contiguous; wrappers materialize contiguous inputs.
- Batch and head strides: arbitrary int32-compatible strides for the logical input layout.
- Workspace tensors: contiguous, explicitly typed, and stored in fixed `[batch, heads, sequence, dimension]` logical order where applicable.
- Out of scope: causal, variable length, GQA/MQA, additional dtypes, and head dimension 128 dispatch.

### Active Pipeline

Python/Triton owns workspace allocation and utility phases. The stable-torch CUTLASS operator only launches the fused main kernel.

```text
Triton preprocess:
  Delta = sum(output * dOutput) per row
  quantize dOutput to int8 with one scale per Q32 x D16 tile
  clear the fp32 dQ accumulation workspace
  clear a padded fp32 dS-row-sum workspace only for smooth K
  emit Q32 factors: max row ||dO||2 and max row |Delta|
  emit K64 factors: max row ||V||2

CUTLASS K128 main kernel:
  recompute QK with int8 MMA
  compute dP with fp16 MMA and fp32 accumulation
  reconstruct P and dS
  for smooth K, atomically sum reconstructed dS by query row
  quantize P with a local scale
  synthesize one dS scale from Q32/K64 factors
  accumulate dV, dK, and dQ with int8 MMA
  write owned dK/dV and atomically accumulate dQ

Triton postprocess:
  invert the physical fp32 dQAccum layout and convert to public fp16 dQ
  for smooth K, add dS_sum * mean(K) before fp16 conversion
```

The active CTA schedule and shared-memory lifetimes are:

```text
cooperative K/V load and packed-K publication
  -> startup CTA barrier
initial Q/dO pair and LSE/Delta publication
  -> initial-state CTA barrier
for each M32 pair:
  each N16 warp computes score, dP, P, and dS
  producer C fragments are transposed directly into dV/dK MMA-A registers
  Q/dO producer warps publish four packed MMA-B fragments each
  dS is mirrored into one swizzled 16x64 parent per N32 pair for the cross-warp dQ consumer
    -> pre-dK/dV CTA barrier
  direct P/dS registers remain warp-local across the barrier
  each N16 warp accumulates owned dK/dV across all four D16 blocks
  loader warps prefetch the next Q/dO pair and row state
  even n_tile warps load each packed K fragment once and reuse it for both dQ M16 halves
  dQ atomics update the internal swizzled global dQAccum MMA-TV slots directly
    -> end-of-iteration CTA barrier
final dK/dV epilogues reuse the dead dS mirror slots
```

There are four static CTA barriers. The resident-V region is aliased by the Q/dO packet publication only after V fragments have been captured in registers. The pre-dK/dV barrier publishes the Q/dO packets and mirrored dS; P and dS for dV/dK no longer cross shared memory. There is no post-dK barrier before dQ. The dQ MMA fragments atomically update the internal swizzled global dQAccum workspace directly; Triton applies the inverse mapping during final conversion. The end-of-iteration barrier both publishes the next asynchronous Q/dO/row-state load and prevents the next iteration from overwriting shared mirror and packet storage still in use.

The current backward wrapper still quantizes Q/K on each call with `per_block_int8`. The head-64 forward uses the same Q32/K64 quantization domain, but its internal tensors are discarded; the backward benchmark therefore prepares the compatible Q/K INT8 tensors and scales outside the timed region as a saved-state reuse proxy. The timed backward path includes Triton preprocessing, workspace/output allocation, the CUTLASS main kernel, and dQ conversion.

The public CUTLASS forward uses `per_block_int8(QBlock=32,KBlock=64)` for head dimensions 64 and 128 and returns only output and optional LSE. Its Q/K scale tensors have the same flat artifact shape required by native backward, while attention CTA and warp dimensions remain independent execution choices. Native backward still consumes only the head-64 specialization. No maintained public forward path routes saved Q/K artifacts into the native CUTLASS backward operator, and no maintained forward path publishes P, P maxima, or a backward dS scale table.

The synthesized predicted maximum is:

```text
1.5 * (softmax_scale / sequence_length) * (max_row ||dO||2 * max_row ||V||2 + max_row |Delta|)
```

The fixed `1.5` safety factor is unconditional. It balances predictor underestimation against additional INT8 quantization error without introducing a policy branch. The active scale constants are:

```text
INT8 reciprocal: 0x1.010122p-7
INT8 scale floor: 0x1.0p-126f
```

The reciprocal represents `1 / (127.5 - 2^-12)` rounded to the selected FP32 value. The floor is the smallest normal FP32 value and is suitable for future BF16 support. C++ uses saturating `cvt.rni.sat.s8.f32`; Triton dOutput quantization uses the shared unclamped `_round_to_int8` helper because its scale is an exact max-abs scale.

No `O(N^2)` scale table is allocated. The predictor work is `O(N*D)` and the V summary is stored at K64 granularity.

### dQ Ownership and Flash Reference

The grid maps `blockIdx.x` to one CTA-owned N block, `blockIdx.y` to head, and `blockIdx.z` to batch. Both configurations keep `n_tile=warp_id`, `n_pair=n_tile/2`, and `n_domain=n_tile/4`, with one physical warp per N16 tile. N128 uses eight warps; its four even-`n_tile` warps cover the four D16 dQ slices and each owner sums all four N32 pairs. N256 uses sixteen warps; physical warps 0-3 own the four D16 dQ slices and each owner sums all eight N32 pairs across four K64 scale domains. Both atomically add one accumulated M16 fragment per CTA directly into the common global fp32 dQ workspace. N256 therefore halves the number of global dQ contributions relative to N128 without changing the public workspace or atomic element count per CTA. The internal workspace is in MMA TV order so each warp's fragment index is contiguous in global memory.

FlashAttention's `compute_dq_dk_dv_1colblock` has the same ownership-level reduction: one CTA owns one K/N column block and contributes a dQ partial for that block. Its normal path accumulates into global `dq_accum`; deterministic mode adds `blockIdx.x * dq_accum_split_stride` and writes a separate split for a later reduction. A Sage split workspace, larger-N CTA, or separate reduction would therefore be a new ownership contract, not evidence for reopening the current one-workspace ownership schedule.

### Swizzled dQAccum Contract

The retained native dQ workspace is an internal physical format; the public dQ tensor remains unchanged. Triton allocates `dQAccum` as contiguous fp32 `(batch, heads, ceil(seq_len / 32) * 32, head_dim)` storage, padding rows to the Q32 contract so preprocessing can clear the workspace unconditionally and native tail atomics remain in-bounds. For a 16x16 tile, the physical element offset is:

```text
(row >> 4) * (16 * head_dim) + (col >> 4) * 256 + lane + 32 * value
```

where the MMA TV coordinates are:

```text
lane = (row & 7) * 4 + ((col & 7) >> 1)
value = (col & 1) | (((row >> 3) & 1) << 1) | (((col >> 3) & 1) << 2)
```

The mapping is the CuTe `right_inverse(ScoreMMA{}.get_layoutC_TV())` contract and is bijective over every 16x16 tile. `tests/cuda/test_qattn_cutlass_bwd_dq_workspace_layout.cu` exhaustively checks the tile mapping and padded global layout for aligned and tail sequence lengths. The native kernel keeps the same FP32 contribution grouping and global atomic count; the swizzle removes only the shared dQ transpose/staging.

`convert_dq` owns the inverse mapping. It loads contiguous physical blocks, uses Triton `reshape` plus `permute` to restore logical row/column order, and stores the public NHD/HND dQ with a mask on the sequence tail. The conversion mapping is independently exact for aligned and tail lengths 32, 33, 65, 96, and 97 in both layouts.

### Reduction Audit

The Triton reductions follow the normal Triton reduction contract and are shaped intentionally:
- `tl.sum(..., axis=1)` reduces each `[BLOCK_M, HEAD_DIM]` tile to one scalar per row for Delta and dOutput L2 norms.
- `tl.sum(..., axis=1)` similarly reduces each `[BLOCK_N, HEAD_DIM]` V tile to row L2 norms.
- `tl.max` over the resulting row vector produces one Q32 or K64 summary value.
- Full-tile `tl.max` is appropriate for the dOutput and V maxima; reducing a smaller axis would change the predictor rather than improve the implementation.

The native CUTLASS kernel has a narrower reduction footprint than the historical dS-policy path. It does not reduce the exact maximum `abs(dS)` on the fly. Instead, it synthesizes the dS scale from the Q32/K64 summaries emitted by Triton. It does retain one cross-thread maximum reduction for the local P quantization scale: `p_max_abs` is accumulated while reconstructing P and passed through `warp_reduce_max`, which uses `__reduce_max_sync`. The pairwise `fmaxf` operations used to combine predictor factors are scalar maxima, not full-tile or cross-warp reductions.

Triton 3.8 lowers its preprocessing reductions through the standard reduction primitive. There is no general faster replacement that preserves these L2/max summaries. A different reduction tree would require a custom CUDA kernel or a weaker bound and must be evaluated end to end. These preprocessing reductions are `O(N*D)` and are not the explanation for the native kernel's shared-store wavefronts; the remaining native costs are P/dS reconstruction, quantization, fragment staging, and scalar scale application.

Moving `sm_scale` out of per-element dS is not an established reduction. In exact real arithmetic, an unscaled dS representation could move `sm_scale` into the dK/dQ dequantization product. The maintained contract also includes the additive scale floor, reciprocal scaling, FP32 rounding, saturating INT8 conversion, and source-order accumulation, so the transformed quantized values are not assumed equivalent. Any such candidate starts as an offline reconstruction and SASS proof, not an algebraic cleanup.

### Correctness

The rebuilt extension and focused FlashAttention-referenced matrix currently pass:

```text
python -m pytest -q tests/test_sagebwd_cutlass.py
6 passed
```

The matrix covers NHD/HND and aligned/tail sequence lengths 64/65 for the single active configuration, plus the CuTe dQ workspace mapping and physical-to-logical conversion. During integration, two correctness issues were fixed:
- Triton dQ conversion had NHD/HND sequence and head strides reversed.
- Tail CTAs retain dQ ownership across the valid N microtiles; invalid N microtiles must not be treated as independent dQ dimension owners.

The normal profile entry point completes for aligned 512 and odd-tail 513:

```text
python build/profile_sagebwd_once.py sage --head-dim 64 --seq-len 512 --num-heads 16 --warmup 1 --block-config 32,64,64,128
python build/profile_sagebwd_once.py sage --head-dim 64 --seq-len 513 --num-heads 16 --warmup 1 --block-config 32,64,64,128
```

Compute Sanitizer Racecheck and Synccheck pass for NHD/HND at aligned length 512 and odd-tail length 513. Racecheck reports `0 hazards displayed (0 errors, 0 warnings)` and Synccheck reports `ERROR SUMMARY: 0 errors` in all four runs. The validation uses `data/racecheck_cutlass_bwd_kernel_only.py`, which does not launch Triton kernels and filters instrumentation with `kernel_substring=fused_mma_kernel_k128_8warp`; ordinary PyTorch setup is outside the checked kernel filter. The earlier whole-process invocation timed out because it also tracked PyTorch setup and Triton compilation/autotuning/runtime kernels; it was not evidence of a CUTLASS deadlock.

For the retained register handoff, candidate/control captures on NHD/HND at sequence 513 and 4096 make dK and dV bitwise identical. The unchanged dQ path differs in 8-297 fp16 elements by at most `6.10e-5`, consistent with cross-run floating accumulation order. The standalone proof in `tests/cuda/test_qattn_cutlass_bwd_score_c_to_a.cu` matches all 512 lane bytes exactly.

The long-shape promotion gates remain:

```text
1 - cosine < 2e-3
relative error < 0.06
maximum absolute error < 0.20
no NaN/Inf
```

For diffusion-training data, cosine and gradient direction are primary. Relative error below `0.10` is the target and below `0.20` is exploratory; these application metrics do not replace the stricter default-promotion gate.

### Performance Against FlashAttention

These CUDA-event timings use the final swizzled image with `smooth_k=False` on an NVIDIA GeForce RTX 3080 Ti Laptop GPU with batch 1, head dimension 64, 16/32 heads, and NHD/HND layouts. Each layout was measured in four independent blocks with alternating head/sequence order; the table reports the median of the four block medians. Each block used 25 warmups and 100 repeats. FlashAttention uses its normal library dispatch and is not constrained to the Sage CTA shape. Effective backward throughput uses `8 * batch * heads * head_dim * seq_len^2` FLOPs.

`kernel_only` includes the CUTLASS launch and Triton dQ conversion after prequantized Q/K and precomputed workspaces. `end_to_end` follows the backward-reuse contract: it reuses prequantized backward-format Q/K and includes Triton preprocessing, workspace/output allocation, the CUTLASS launch, and dQ conversion. It excludes Q/K quantization and the forward pass.

| Layout | Heads | Seq | Flash median ms | Sage end-to-end ms | End-to-end speedup | Sage kernel-only ms | Kernel-only speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| NHD | 16 | 4096 | 4.773 | 4.116 | 1.160x | 3.884 | 1.229x |
| NHD | 16 | 8192 | 17.811 | 15.552 | 1.145x | 15.108 | 1.179x |
| NHD | 32 | 4096 | 9.136 | 8.147 | 1.121x | 7.735 | 1.181x |
| NHD | 32 | 8192 | 34.739 | 30.526 | 1.138x | 29.948 | 1.160x |
| HND | 16 | 4096 | 4.789 | 4.126 | 1.161x | 3.902 | 1.227x |
| HND | 16 | 8192 | 17.917 | 15.597 | 1.149x | 15.049 | 1.191x |
| HND | 32 | 4096 | 9.279 | 8.078 | 1.149x | 7.711 | 1.203x |
| HND | 32 | 8192 | 34.747 | 30.594 | 1.136x | 29.802 | 1.166x |

Sage end-to-end throughput is `33.309-36.019 TFLOPS`; kernel-only throughput is `35.223-36.894 TFLOPS`. Across the eight final rows, end-to-end speedup is `1.121-1.161x` with a `1.145x` geometric mean, and kernel-only speedup is `1.160-1.229x` with a `1.192x` geometric mean. Full consolidated rows are in `build/bench_sagebwd_cutlass_rawk_NHD_4096_8192.csv` and `build/bench_sagebwd_cutlass_rawk_HND_4096_8192.csv`; the four raw blocks per layout are retained as `build/bench_sagebwd_cutlass_final_swizzled_{NHD,HND}_{a,b,c,d}.csv`.

The separate same-build swizzle promotion comparison remains distinct from the FlashAttention table: the candidate/control ratio is `0.956300x`, or `1.045697x` geometric speedup overall; end-to-end speedup is `1.046422x` and native kernel plus conversion speedup is `1.044972x`. The swizzle removes shared dQ staging while preserving the global contribution count; these candidate/control values must not be multiplied with the FlashAttention rows above.

A separate pair-interleaved dS mirror comparison keeps the Q32/K64 arithmetic, four barriers, and global dQ contribution count unchanged. Four alternating-order blocks per layout used 25 warmups and 100 repeats; the table reports the median of four block medians, with control and candidate measured in separate same-build processes. This is a control comparison, not an additional FlashAttention result.

| Layout | Heads | Seq | Control native+convert ms | Pair native+convert ms | Native speedup | Control end-to-end ms | Pair end-to-end ms | End-to-end speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NHD | 16 | 4096 | 3.893232 | 3.875328 | 1.005x | 4.142336 | 4.093080 | 1.012x |
| NHD | 16 | 8192 | 15.035904 | 15.055824 | 0.999x | 15.409888 | 15.365888 | 1.003x |
| NHD | 32 | 4096 | 7.751424 | 7.750616 | 1.000x | 8.176584 | 8.139248 | 1.005x |
| NHD | 32 | 8192 | 30.178776 | 30.248144 | 0.998x | 30.628320 | 30.517248 | 1.004x |
| HND | 16 | 4096 | 3.902136 | 3.847936 | 1.014x | 4.153064 | 4.076032 | 1.019x |
| HND | 16 | 8192 | 15.252736 | 15.145472 | 1.007x | 15.370160 | 15.369968 | 1.000x |
| HND | 32 | 4096 | 7.726000 | 7.676928 | 1.006x | 8.102600 | 8.052416 | 1.006x |
| HND | 32 | 8192 | 29.837808 | 29.710544 | 1.004x | 30.575616 | 30.498816 | 1.003x |

The pair-interleaved image has geometric speedup `1.004x` for native plus conversion and `1.006x` end-to-end. End-to-end latency is lower in all eight rows, while native plus conversion is effectively tied in the two NHD sequence-8192 rows. Raw block CSVs are `build/bench_sagebwd_pair_interleaved_reuse_{NHD,HND}_{a,b,c,d}_{candidate,control}.csv`; the consolidated rows are in `build/bench_sagebwd_pair_interleaved_reuse_all.csv`.

The final D64 forward grid used below measures `1.239-1.401x` end-to-end speedup with a `1.318x` geometric mean, and `1.386-1.471x` kernel-only speedup with a `1.415x` geometric mean. Its full rows are in `build/bench_qattn_cutlass_rawk_NHD_4096_8192.csv` and `build/bench_qattn_cutlass_rawk_HND_4096_8192.csv`.

#### Derived Forward Plus Backward

These head-64 totals sum the final forward end-to-end median and the final backward-reuse end-to-end median for each implementation. The forward term uses the best D64 CUTLASS execution tile from the retained final forward grid. Sage Q/K quantization is counted exactly once in forward and excluded from backward. Total effective throughput uses `12 * batch * heads * head_dim * seq_len^2` FLOPs. No combined forward/backward function was implemented or timed.

| Layout | Heads | Seq | Flash total ms | Sage total ms | Flash total TFLOPS | Sage total TFLOPS | Overall speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| NHD | 16 | 4096 | 6.5213 | 5.5031 | 31.613 | 37.462 | 1.185x |
| NHD | 16 | 8192 | 24.4088 | 20.3655 | 33.784 | 40.492 | 1.199x |
| NHD | 32 | 4096 | 12.5033 | 10.7359 | 32.977 | 38.406 | 1.165x |
| NHD | 32 | 8192 | 47.9966 | 40.4236 | 34.362 | 40.800 | 1.187x |
| HND | 16 | 4096 | 6.5397 | 5.5393 | 31.524 | 37.218 | 1.181x |
| HND | 16 | 8192 | 24.6926 | 20.4317 | 33.396 | 40.360 | 1.209x |
| HND | 32 | 4096 | 12.6448 | 10.6695 | 32.608 | 38.644 | 1.185x |
| HND | 32 | 8192 | 47.8949 | 40.4124 | 34.435 | 40.811 | 1.185x |

Overall speedup spans `1.165-1.209x`; the geometric mean of row speedups is `1.187x`, while the ratio of all summed Flash medians to all summed Sage medians is `1.189x`. This is a derived scope result for the native-SM86, fp16, fixed-length, non-causal, head-64 workload, not a general claim across architectures or attention modes.

### Accuracy and Predictor Evidence

The predictor experiments use two independent SDXL-derived sequence-4096 captures with ten heads and sigma indices `300/700/900`. They established:
- A true pre-forward predictor is not viable. Same-timestep dOutput RMS differed by approximately `28x`; same-RMS random dOutput still changed block scales by p99 ratios up to `31.5x` and a maximum around `642x`.
- The pre-backward separable predictor is the useful asymptotic class. In the simulator, using dequantized Q/K and exact-dS references to isolate scale-policy error, the base formula measured mean/min dQ cosine `0.996286/0.993382`, dK `0.998667/0.994683`, mean/max dQ relative error `7.73%/11.50%`, and dK `4.10%/10.31%` on both captures and all ten heads.
- A dOutput-max-derived approximation measured mean/min dQ cosine `0.996062/0.992853`, dK `0.998681/0.994846`, mean/max dQ relative error `8.00%/11.95%`, and dK `4.10%/10.15%`; peak saturation was below `64 ppm` in that experiment.
- The active safety factor is `1.5`. The extended replay covers both captures, all ten heads, exact sequence 4096, sliced 512/513 controls, and a deterministic tiled 8192 control. The 8192 rows exercise the long kernel schedule but are synthetic extensions of sequence-4096 data, not independent natural-length captures.
- All 24 aggregate cases and all 240 per-head cases are finite. Fifteen of 24 aggregate rows and 144 of 240 per-head rows pass the strict promotion gate. At exact sequence 4096, four of six aggregates and 42 of 60 heads pass. Mean dQ/dK relative error remains `5.49%/4.42%`, worst relative error is `11.88%/19.06%`, and mean/worst dV relative error is `0.86%/1.70%`. Both sigma-900 records fail the strict aggregate gate; the seed-2 record also fails dQ and dK cosine, relative, and maximum-absolute gates.
- A matched smooth-K replay covers both captures, all six exact-4096 records, and all 60 heads. Aggregate dS clipping changes only from `12.507` to `12.451 ppm`; mean p99 and maximum true/predicted dS ratios move from `1.2003x/15.235x` to `1.1970x/15.117x`. Smoothing therefore does not materially improve predictor calibration, and the `1.5x` predictor remains unchanged. It does reduce mean aggregate dQ relative error from `5.49%` to `3.60%` and dV from `0.86%` to `0.62%`; aggregate strict passes remain `4/6`, while per-head passes improve from `42/60` to `46/60`. Raw rows are paired with centered rows in `build/cutlass_bwd_smooth_k_predictor_4096.csv`.
- Reconstructed source-order telemetry matches the kernel's P `32x16` warp scale, dS `32x64` predictor scale, reciprocal multiplication, and saturating conversion. At sequence 4096, P has zero clipping, `0.024-0.105%` zeros, and approximately `0.207%` endpoint saturation, which is expected from local max scaling. dS has `22.35-42.67%` zeros, `7.47-29.34 ppm` endpoint saturation, and `7.31-28.94 ppm` clipping across the six aggregate records.
- Rare predictor misses are real but do not explain the accuracy ranking alone. The aggregate maximum true-dS/predicted-dS ratio reaches `30.04x`; however, the worst gradient record clips only approximately `10 ppm`, while the other sigma-900 record clips approximately `29 ppm`. Across all shape controls, aggregate P-scale medians span `4.48e-6` to `7.73e-5`, dS-scale medians span `2.10e-8` to `6.41e-5`, and the evaluator emits fixed log2 histograms plus min/p01/p10/p50/p90/p99/p999/max percentiles.
- The staged FP32 reconstruction matches the real predicted-scale kernel within `0.08%` relative error at exact sequence 4096. Q/K reconstruction alone and exact-max dS quantization each pass all six aggregate and all 60 per-head strict gates; predictor-scaled dS falls to four of six and 42 of 60, exactly matching the real kernel. On the two sigma-900 aggregates, Q/K reconstruction gives dQ/dK relative error `1.15%/0.75%` and `1.14%/1.21%`; exact-max dS gives `1.53%/0.79%` and `2.16%/2.82%`; prediction raises them to `9.58%/3.19%` and `11.88%/19.06%`. P and dOutput quantization remain secondary and dV passes every aggregate.
- A scalar guard sweep from `0.25x` through `4.0x` does not pass either sigma-900 aggregate; the best setting passes only five of 20 heads. Multiplying the bound by the already-computed P maximum improves the best point to nine of 20 heads and one of two aggregates, but no coefficient passes both. Lower scales trade zero rate for hundreds to thousands of clipping ppm, while higher scales increase zeros and dK error. Neither scalar-only formulation is promoted.
- A second-order Q32/K64 sketch is materially better calibrated than the retained separable bound. The tested 2x4 formulation summarizes Q/K and dOutput/V centroids, isotropic RMS radii, mean LSE, and Delta dispersion, then evaluates eight centroid pairs per 2,048-value dS block. Its raw predicted/true scale ratio is `0.484x/1.095x/1.851x` at p10/p50/p90, versus `2.024x/4.530x/8.582x` for the active guarded predictor. A `3x` sketch guard improves exact-4096 strict passes from four of six aggregates and 42 of 60 heads to four of six and 47 of 60; the two sigma-900 dQ/dK errors become `7.21%/3.02%` and `6.67%/10.25%`. The calibration gain is real, but both hard aggregates still fail.
- Sparse exact representatives protect the sketch's miss tail but do not reach exact-scale accuracy. Selecting eight Q rows and eight K rows from LSE/dOutput/Delta and K/V norm features evaluates 64 of 2,048 pairs. `max(1.5 * sketch, 4 * sparse_max)` reaches five of six aggregates and 52 of 60 heads; its sigma-900 dQ/dK errors are `5.69%/2.95%` and `5.24%/7.84%`. Expanding to 16x16 representatives, 256 of 2,048 pairs, still reaches only five of six and 52 of 60, with the remaining dK error `6.98%`. Even an oracle that replaces the 32 worst-predicted K64 scales out of 64 per Q block reaches only five of six aggregates, although it reaches 58 of 60 heads. The remaining error is distributed rather than confined to a small top-K set.
- Held-out log-scale regression confirms that the sketch features contain useful information without establishing a production policy. Training on one capture and evaluating the other, an eight-feature model using the retained bound, 2x4/4x8 moments, and 32/64-pair sparse lower bounds produces p10/p50/p90 predicted/true ratios `0.591x/1.025x/1.424x` and p90 absolute log2 error `0.819`, versus `3.101` for the active predictor. Mean guards and conditional `0.65-0.90` quantile models still reach at most five of six aggregates in reconstructed gradients. Two captures are also insufficient evidence for learned coefficients, so no learned runtime dependency or coefficient set is promoted.
- Generic 8/16/32-dimensional random projections are rejected for the score term. Projection error is exponentiated by softmax; all tested guards pass zero of six aggregates, and the unguarded maximum dS zero rates are approximately `84.9%`, `67.4%`, and `46.6%`. Random projections may still summarize the linear dOutput/V residual, but they are not a viable approximation to QK logits here.
- The sketch cost is much lower than exact pairwise preprocessing but is not strictly linear. A 2x4 centroid combine evaluates `8/2048 = 0.39%` of the exact block-pair count; 8x8 and 16x16 sparse representatives evaluate `3.125%` and `12.5%`. Materializing one FP32 Q32-by-K64 scale table for sequence 8192 and 16 heads requires approximately 2 MiB. A future candidate must count that table, the block-pair combine, and selection overhead in end-to-end timing rather than describing it as the current `O(N*D)` predictor.

The obsolete block-omission experiment is no longer part of the maintained data path.

### Measured Bottlenecks

The retained fixed-scale pipeline is dominated by P/dS reconstruction, INT8-to-FP32 scale application, fragment and packet handoffs, and synchronization around cross-warp ownership. The final current-build profile below is the authoritative profiling result for the swizzled image and its pre-swizzle control; CUDA-event measurements remain the latency promotion gate.

#### Final Swizzled Workspace Profile

Paired current-build Nsight Compute raw captures used the same `B=1, H=16, N=8192, D=64` launch, `Q32/K64`, CTA `M64xN128`, eight warps, and one filtered `fused_mma_kernel_k128_8warp` invocation for the pre-swizzle control and retained swizzled image. NHD and HND use the same binary pair and replay settings.

| Metric | NHD control | NHD swizzled | HND control | HND swizzled |
|---|---:|---:|---:|---:|
| Kernel duration (ms) | 19.396 | 18.887 | 19.375 | 18.884 |
| Executed warp instructions (M) | 1,590.177 | 1,560.609 | 1,590.177 | 1,560.609 |
| Shared-load instructions (M) | 119.079 | 102.302 | 119.079 | 102.302 |
| Shared-store instructions (M) | 43.319 | 26.542 | 43.319 | 26.542 |
| Shared-load wavefronts (M) | 285.606 | 268.829 | 285.606 | 268.829 |
| Shared-store wavefronts (M) | 44.090 | 26.299 | 44.159 | 26.362 |
| Global RED instructions (M) | 16.777 | 16.777 | 16.777 | 16.777 |
| Barrier-stall share (%) | 16.01 | 14.72 | 15.99 | 14.80 |
| Registers/thread | 243 | 233 | 243 | 233 |
| Dynamic shared memory (KiB) | 57.600 | 49.408 | 57.600 | 49.408 |

The global reduction sector count is also unchanged at `67.109M` in both layouts. Global L2 and DRAM traffic moves only at measurement noise scale, and the launch remains register/shared-memory limited to one resident CTA in all four captures. The swizzle therefore removes on-chip staging and instruction work; it does not reduce global dQ atomics. These Nsight Compute durations are profiler-instrumented architectural evidence, not normal latency measurements.

#### Reduction Status

The exact on-the-fly `abs(dS)` maximum reduction is no longer present. The active kernel still pays for the local P scale reduction: scalar `fmaxf` updates build `p_max_abs`, followed by one warp-level `__reduce_max_sync` before P is quantized. This is a real non-tensor cost, but it is narrower than the removed dS maximum reduction and should be treated as one part of the P reconstruction path rather than as the main remaining dS-scale mechanism.

The Q32/K64 predictor summaries are computed in Triton with `O(N*D)` reductions. They do not create the native shared-store traffic. Replacing those reductions is lower priority unless an end-to-end phase breakdown shows that preprocessing, rather than the CUTLASS launch, has become material.

#### P and dS Reconstruction

Each K128 iteration reconstructs P and dS around the int8/fp16 MMA results. The path includes int32-to-fp32 conversion, score and softmax scaling, `expf`, LSE and Delta subtraction, dS formation, absolute-value/max tracking, and scale arithmetic. P and dS are then quantized with reciprocal-scale application and integer conversion. This combination explains a substantial part of the excess non-tensor instruction activity and includes the remaining P warp reduction.

#### Shared-Memory Handoffs

The producer-local canonical P/dS score-pair handoff is removed. The active path still materializes several fragments through shared memory:
- dS mirror stores into one swizzled `dS_dKV` parent per N32 pair so the dQ path can consume both reconstructed M halves through static 16x32 MMA-A views.
- Packed Q/dO fragment stores and later loads for the dK/dV MMAs.
- Packed K fragment stores and later loads for the dQ MMAs.
- Swizzled global dQAccum storage for direct atomic accumulation; the inverse permutation is deferred to Triton conversion, so the active kernel has no shared dQ transpose.
- dK/dV epilogue staging before coalesced global stores.

#### Scalar Scale Application and Fragment Plumbing

The dV, dK, and dQ accumulation paths apply FP32 products involving P, dS, dOutput, Q-block, K-block, and output scales around the integer MMA fragments. Manual packed-fragment indexing, address arithmetic, tail predicates, conversions, and repeated synchronization add further non-tensor instructions and scheduler stalls. These operations also constrain attempts to keep P/dS values in registers because producer lifetimes already approach the resource limit.

#### Remaining-Work Classification

| Stage | Status under the retained contract | Consequence |
|---|---|---|
| dV/dK conversion and scaled accumulation | Exhausted locally | Independent P/dS, Q, and dOutput scales require conversion and FP32 products; grouped, FP16-outer, and custom-conversion candidates did not win |
| P/dS reconstruction | Locally reducible only at small margin | QK, exponentiation, P, and dS are required unless forward artifacts or the numeric contract change |
| Quantization, C-to-A handoff, and dS publication | Mostly exhausted | Producer-local P/dS stores are removed; mirrored dS remains required for cross-warp dQ |
| dQ workspace and atomic accumulation | Swizzled physical workspace retained | Direct MMA-TV global atomics remove shared dQ staging while preserving FP32 grouping and the global contribution count |
| Q/dO packet and next-pair traffic | Tested local schedules exhausted | Current prefetch overlaps dQ; early-dO and tail-prefetch variants regressed |
| Forward-owned Q/K or P metadata | Contract-changing | Both forward head dimensions use Q32/K64, but the public forward still discards Q/K artifacts; head-128 backward and full P checkpointing remain separate contracts |

The practical conclusion is that large-margin gains are exhausted inside the current fixed four-INT8 pipeline. Remaining local ideas have ceilings bounded by small stage shares or require changing quantized values. Larger gains must be evaluated as architectural or end-to-end product changes.

## Completed Work

This section records the decisions behind the current source and prevents closed experiments from being repeated without new evidence. An implementation defect invalidates only that candidate build and its measurements, not the underlying design. An experiment is closed only after the defect is fixed and the corrected candidate is validated, or when independent design-level accuracy or contract evidence rules out that specific formulation.

### Completed Audits

- The pre-swizzle resource audit reported 243 registers per thread for tail and aligned variants, zero stack/local spill, and 57,600 bytes of dynamic shared storage. The retained pair-interleaved raw-K N128 image remains exactly 235/234 registers and 2,192/2,024 instructions for tail/aligned, with 49,408 bytes dynamic shared memory and zero stack/local spill; smooth K uses 237/234 registers and 2,344/2,088 instructions, also with zero stack/local spill. Both retain four static barriers. N256 retains the specialization-local 128-register cap so 512 threads fit the SM86 65,536-register CTA limit. The optimized raw-K N256 image uses 86,272 bytes shared, 136/112 bytes stack, and 2,520/2,344 instructions for tail/aligned; smooth K uses the same shared allocation, 160/120 bytes stack, and 2,696/2,400 instructions. Both shapes still permit only one resident CTA.
- The post-handoff line-info SourceCounters profile maps all 2,088 active SASS instructions to `qk_int8_sv_f16_bwd_kernel_cutlass_sm80.cuh`. The largest instruction families are `FMUL.FTZ` (`220.2M`), `FFMA.FTZ` (`211.8M`), `I2FP.F32.S32` (`201.3M`), `F2I.S8.NTZ` (`67.1M`), `PRMT` (`84.0M`), and `MOVM`/`SHFL.IDX` (`33.6M` each). `MUFU.EX2` accounts for `33.6M` exponent operations. Source attribution is therefore available when the extension is built with `-lineinfo`; the normal retained binary has identical machine instructions without the debug correlation metadata.
- The barrier audit established four dependency boundaries. Removing the startup K/V publication barrier produced 4,096 Racecheck hazards between asynchronous shared writes and cross-warp `LDSM` reads. The pre-dK/dV barrier publishes Q/dO packets and dS. The post-dK barrier before dQ is safely removed, and the end barrier protects next-pair Q/dO and row-state publication.
- The historical canonical dS audit established that dQ needs one K32 MMA-A fragment assembled from two N16 producer halves. Direct gathering from two 1 KiB-separated N16 slots was closed under CuTe's vectorization contract; the retained pair-interleaved mirror instead packs both temporal M16 halves into one swizzled 16x64 int8 parent per N32 pair and keeps the native 16x32 `LDSM.x4` loads.
- The canonical two-source proof was corrected before promotion. The first standalone probe accidentally applied `make_transposed_tensor` to both the canonical score-pair destination and the mirror reference; production intentionally stores canonical dS transposed and the dQ mirror non-transposed. With the exact production orientations and source-distinct pseudorandom fragments, neither normal nor `.trans` `LDSM.x2` plus any uniform lane shift matched the MMA-A reference.
- Exhaustive low-instruction mapping checks closed the direct two-source x2 mirror formulation. None of 3,840 lane-bit-permutation/XOR address maps matched either M half. Byte-signature mapping showed that a normal load scatters each eight-byte target across eight source lanes, while a transposed load scatters it across four. A modeled hardware register transpose requires at least three `movmatrix`, two lane-permutation, and two local-byte-permutation stages per source fragment. That remains noncompetitive for eliminating the cross-warp dQ mirror, but it does not apply to the producer-local C-pair handoff used by dV/dK.
- A separate producer C-pair proof found the exact four-stage bit network `MOVM -> lane rotate -> byte 1/2 swap -> MOVM`. Coordinate-tagged data match all 512 bytes of the canonical MMA-A fragment. Integrating that mapping removes the canonical P/dS stores and reloads while retaining the dQ mirror and four-barrier ownership schedule.
- Matched sequence-8192 retained-image attribution shows `0.9940x` profiled duration, `0.9132x` LDSM execution, `0.9445x` shared-load wavefronts, and `0.3976x` shared-store wavefronts versus the saved pre-handoff control. Shared-store bank conflicts fall `12.8%`, eligible warps rise from `0.42` to `0.43` per cycle, and tensor instructions are identical. Total executed instructions rise `2.1%` because of the register transpose, while the measured latency still improves.
- The long-shape telemetry and error-decomposition audit is complete. `data/evaluate_cutlass_bwd_captures.py` now reports aggregate and per-head gradient cosine, relative error, maximum absolute error, finite and strict-gate status, Q/K/dOutput/P/dS INT8 zero rates, P/dS endpoint saturation and clipping, P/dS scale percentiles, true-dS/predicted-dS percentiles, and fixed log2 histograms. It also compares Q/K reconstruction, exact-max dS, predicted dS, local P quantization, dOutput quantization, and the real INT8 kernel. Its default matrix covers both captures and 512/513/4096/8192 with explicit `slice`, `exact`, and `tiled` provenance labels.
- The packet and lifetime audit found no free compaction margin. Q/dO/K packet allocations exactly match their fragment counts; the K packet is `16 fragments * 4 words * 32 lanes * 4 bytes = 8,192 bytes`. Lowering shared storage alone cannot create a second resident block, so double buffering requires a proven dead region or a different ownership schedule.

### Completed Design Reviews

- dQ ownership and Flash comparison. Sage and Flash both produce one dQ partial per CTA-owned K/N block. Sage's four even producer warps cover D16 slices and atomically accumulate into one fp32 workspace. The retained swizzle changes only the internal physical address and deferred conversion; Flash's deterministic split pointer is an alternate workspace/reduction contract, not evidence that Sage should add a split reduction.
- Local dQ staging comparison. Same-build NCU duration improved from `180,896 ns` for the shuffle path to `168,448 ns` for the bank-aware shared transpose. Shared-load/store bank-conflict counts moved from `271,226/9,361` to `282,969/57,887`, showing that the faster result comes from the complete staging/coalescing schedule rather than conflict count alone. Same-build timing favored shared transpose in all four controls by approximately `4.5-4.8%`: NHD `23.41548 -> 22.28124 ms` at 8192 and `5.96204 -> 5.69924 ms` at 4096; HND `23.19785 -> 22.18089 ms` and `5.92862 -> 5.67287 ms`.
- Forward/backward artifact compatibility. CUTLASS forward uses the same Q32/K64 Q/K scale domain at head dimensions 64 and 128, including K32 CTAs sharing one K64 scale and Q16 warps sharing one Q32 scale. Native backward consumes that contract only at head dimension 64. The forward wrapper still discards those quantized tensors after producing output/LSE, so the backward-reuse benchmark remains a product-contract proxy rather than an integrated training invocation.
- Forward-owned P artifacts. Forward computes P transiently inside online softmax and publishes only output/LSE. A saved P maximum can remove only the local P maximum reduction; it cannot remove QK recomputation, exponentiation, P reconstruction, or dV's P operand. Saving quantized P itself is `O(N^2)` and is not feasible under the current memory contract.
- dS scale-fusion review. No existing scale-cancellation implementation was found. An unscaled-dS representation could move `sm_scale` in real arithmetic, but the scale floor, reciprocal, rounding, saturation, and dequantization products change the quantized contract. This remains a bounded offline hypothesis, not a source-level optimization.
- Q/dO lifetime review. The current next-pair `load_q_dO_pair()` is issued after current dK/dV has consumed the packed Q/dO operands, while dQ no longer reads the raw `q_i8`, `dO_i8`, or `dO_fp16_pair` buffers. This safely overlaps asynchronous loads with dQ. Moving the load after the end barrier is race-safe only by giving up that overlap; moving it across an earlier dependency requires a new publication proof. Raw double buffering is not justified with the remaining shared-memory headroom.
- Sixteen-warp feasibility and lifetime audit. The generated N256 specialization proves that the shape is representable and runnable: its 128-register cap fits 512 threads exactly within the SM86 CTA register limit. The first implementation kept four FP16 V MMA-B fragments live across score reconstruction and spilled 208/160 bytes per thread. Keeping V in dedicated N256 shared storage and loading one D16 fragment only for its dP use reduces the retained stack to 136/112 bytes without changing arithmetic, scales, barriers, or ownership. The phased pair-interleaved dS and dK/dV epilogue scratch share one 8 KiB region, leaving 86,272 bytes dynamic shared memory. N128 compiles through the original register-resident V path and remains instruction-for-instruction identical to its saved control.

### Retained Changes

- Unified Q32/K64 CUTLASS forward contract. The normal configured/public paths for head dimensions 64 and 128 quantize with flat Q32/K64 metadata and launch one fixed scale path while retaining all four CTA/warp execution instantiations. The public API, eager autotune, and compile autotune remain supported. Sequence-129 NHD/HND tests cover both head dimensions and all four execution tiles with and without LSE.
- Native backward dead-surface cleanup. The internal operator no longer accepts `output` or final fp16 `dQ`, because Triton preprocessing consumes output before launch and Triton postprocessing converts the fp32 dQ workspace afterward. Their unused launch strides and rejected, unreferenced dQ/scale/transpose helpers were removed. Generic head-dimension and tile traits plus generated-instantiation plumbing remain available for a future dedicated D128 design.
- K128 main-kernel baseline and N256 experimental config. QBlock=32/KBlock=64 CTA M64xN128 with eight warps remains the public default. CTA M64xN256 with sixteen warps is generated and dispatched only when callers explicitly request `(32,64,64,256)`. N256 preserves one warp per N16 producer tile, expands dQ from four to eight N32 pairs and from two to four K64 domains, and halves cross-CTA dQ contributions. No backward autotuner selects between them.
- Triton utility phases. Delta/dOutput preprocessing, Q32/K64 predictor summaries, workspace allocation, and fp32 dQ-to-fp16 conversion moved out of the native wrapper and into autotuned Triton kernels. The native operator now launches the fused CUTLASS main kernel with explicit, typed workspaces.
- Native smooth K. Backward accepts the forward-compatible raw-domain LSE, centers K in the same Q32/K64 artifact domain, specializes the native kernel on smoothing, accumulates exact reconstructed-dS row sums without a new barrier, and fuses the K-mean correction into swizzled dQ conversion. The predictor audit found no reason to change the retained `1.5x` policy. The raw N128 specialization retains the control's instruction counts, register allocation, stack use, and barrier count. The 36-case focused matrix covers raw/smooth N128/N256, NHD/HND, and aligned/tail inputs; smooth Racecheck and Synccheck pass N128/N256 NHD/HND at 512/513.
- Unconditional separable dS prediction. The retained performance predictor is `1.5 * (softmax_scale / N) * (max_row ||dO||2 * max_row ||V||2 + max_row |Delta|)`. It is `O(N*D)`, uses Q32/K64 summaries, and does not allocate an `O(N^2)` scale table or select a runtime policy. It is not the strict-accuracy oracle.
- Quantization constants and conversion. The accepted reciprocal is `0x1.010122p-7`, the scale floor is `2^-126`, and C++ INT8 conversion is saturating. Triton dOutput quantization uses the shared unclamped `_round_to_int8` helper because its exact max-abs scale preserves the intended range.
- Guard selection. The fixed `1.5x` dS guard is retained as the performance reference. A `1.25x` guard was faster on the six retained captures but was less robust on the sequence-65 correctness control; `2.0x` increased quantization error. The expanded scalar sweep confirms that no global guard or simple P-max multiplier clears both sigma-900 records, so guard tuning is closed without a new predictor structure.
- Training-derived validation data. Two independent SDXL-derived sequence-4096 captures with ten heads and sigma indices `300/700/900` are retained with reusable capture, simulation, benchmark, and evaluator scripts under `data/`. The full replay is finite, but exact sequence 4096 passes only four of six aggregate and 42 of 60 per-head strict gates. Mean dQ/dK relative error is `5.49%/4.42%`, worst error is `11.88%/19.06%`, and sparse dS clipping does not by itself predict those failures; this remains exploratory rather than strict-promotion quality.
- Validation baseline. The active build passes the parameterized N128/N256 focused matrix. Source-distinct N256 comparisons cover NHD/HND at 256/257/512/513: dK/dV are bitwise identical to N128 and dQ differs by at most `1.22e-4` from the changed FP32 atomic grouping. After the N256 V-storage ownership change, Racecheck and Synccheck again pass aligned/tail NHD/HND at 512/513; tail Memcheck had already passed both layouts. The completed N128 validation also includes the 16/32-head final Flash benchmark, swizzled-workspace ABBA timing, SASS/resource inspection, paired current-build NCU captures, `pre-commit`, compilation, and whitespace checks.
- N256 initial timing baseline. The normal benchmark CLI defaults to both generated configs and also accepts either through `--block-configs`. At `B=1,H=16,D=64`, 10 warmups and 50 repeats give N128/N256 kernel-plus-conversion medians of `3.9797/4.9833 ms` and `15.0134/20.0980 ms` for NHD 4096/8192, and `3.9951/5.9597 ms` and `15.1460/25.2964 ms` for HND. N128/N256 ratios are `0.7986x`, `0.7470x`, `0.6704x`, and `0.5987x`; the geometric mean is `0.6995x`, so the initial N256 path is `1.430x` slower overall. Raw rows are in `build/bench_sagebwd_n128_n256_{NHD,HND}_initial.csv`.
- N256 on-demand V reload. The retained N256 path keeps V in dedicated shared storage and reloads one D16 MMA-B fragment immediately before its two dP MMAs, rather than carrying four fragments across QK, exponentiation, and P/dS reconstruction. Pair-interleaved dS and the dK/dV epilogue reuse the same phased 8 KiB scratch region. An order-controlled final/initial/initial/final run at `B=1,H=16,D=64`, 10 warmups, and 50 repeats reports initial/final medians of `4.9971/4.2404 ms` and `19.8963/15.2748 ms` for NHD 4096/8192, and `5.0145/4.2813 ms` and `25.2410/15.5597 ms` for HND. Speedups are `1.178x`, `1.303x`, `1.171x`, and `1.622x`, or `1.307x` geometrically. In the final same-build N128/N256 comparison, medians are `3.9076/4.1636 ms`, `14.9452/15.1127 ms`, `3.9178/4.1850 ms`, and `14.9632/15.3507 ms`; optimized N256 is only `1.042x` slower geometrically but still loses every row, so N128 remains the public default. Raw final rows are in `build/bench_sagebwd_n128_n256_v_reload_final_{NHD,HND}.csv`; order-controlled rows are in `build/bench_n256_v_reload_final_abba_*.csv`.
- Smooth-K timing. At `B=1,H=16,D=64`, N128 raw/smooth kernel-plus-conversion medians are `3.8128/4.1656`, `14.7865/16.2719`, `3.8205/4.1681`, and `14.8567/15.8874 ms` for NHD/HND 4096/8192. End-to-end medians are `4.1047/4.4021`, `15.1905/16.6333`, `4.0776/4.4344`, and `15.1694/16.1992 ms`. Smooth/raw geometric ratios are `1.088x` kernel-plus-conversion and `1.081x` end-to-end. The added exact row sums improve quantized-gradient accuracy but are not free; raw rows remain the performance control. Results are in `build/bench_sagebwd_cutlass_smooth_k_{NHD,HND}_4096_8192.csv`.
- Scale/index hoists. The active kernel caches the CTA-invariant K-domain index, reuses the Q-block index for predictor metadata, caches the dK scale product and dOutput scale-index base, and specializes K-scale lifetime so only aligned launches retain the CTA-invariant scale across M-pairs. With `-lineinfo` on native SM86, aligned static SASS falls from 2096 to 2088 instructions while the tail function is instruction-identical to baseline; registers remain 243 aligned and 227 tail with zero stack/local memory. On identical NHD/HND inputs at sequence 513/4096, dK/dV are bitwise identical and dQ differs only in 7-314 FP16 elements by at most `6.10e-5` with relative L2 below `1e-5`. Kernel-only timing moved by less than the laptop GPU clock variation in the paired candidate/baseline runs, so this is retained as a low-risk control rather than a promotion-quality speedup claim.
- Post-dK barrier removal. The synchronization point between dK/dV accumulation and dQ staging is removed. Native SM86 SASS now has four static CTA barriers instead of five; relative to the five-barrier scale/index-hoist build, tail SASS is seven instructions shorter and aligned SASS removes one `BAR.SYNC.DEFER_BLOCKING` while adding one `NOP`. `__maxnreg__(243)` preserves the 227-register tail and 243-register aligned allocations with zero stack/local spill. Racecheck and Synccheck pass NHD/HND at sequence lengths 512 and 513. Paired kernel-only timing at 4096/8192 was mixed and within laptop clock/thermal drift, so the cleanup is retained without a performance-win claim.
- Packed-K dQ reuse. The dQ phase now loads each packed K fragment once per even warp and consumes it for both M16 halves, rather than reloading it for each half. It also computes each K-domain predictor scale once per pair. Native SM86 SASS falls by 16 `LDS` instructions and 22/13 total instructions in tail/aligned variants; the static CTA barrier count remains four in the retained image. Both variants use 243 registers with zero stack/local spill. Racecheck and Synccheck pass NHD/HND at 512/513. Paired new-contract timing is clock-sensitive in absolute milliseconds, but same-run Flash-normalized ratios improved by approximately `0.7-1.2%` for NHD and `0.7-3.2%` for HND across 4096/8192 kernel-only and end-to-end rows, so the change is retained.
- Historical pre-swizzle bank-aware shared dQ transpose. Before the swizzled workspace, the `dQ_stage_offset()` layout staged each 16x16 fp32 accumulator in a private dead score-pair slot and issued contiguous global atomics. It preserves the same cross-CTA reduction count as the prior shuffle path but wins every same-build NHD/HND control by approximately `4.5-4.8%`. Direct, row-major, affine, fp16, int32, and alternate transposed staging variants did not beat it. That path is superseded by the swizzled workspace; revisit dQ only through a materially different ownership contract.
- Swizzled global dQ workspace and deferred inverse mapping. The retained implementation replaces per-fragment shared dQ staging with a bijective internal MMA-TV physical layout for `dQAccum`. The native kernel atomically updates the workspace directly from MMA fragments, and `convert_dq` applies the inverse permutation. It removes the targeted shared dQ staging while preserving the global contribution count, FP32 accumulation, aligned/tail NHD/HND mapping, and coalesced conversion reads. The workspace proof, sanitizers, zero-spill resource inspection, current-build NCU counters, and paired end-to-end timings all pass the promotion requirements; this candidate is complete and retained.
- Pair-interleaved dS mirror. The maintained dS workspace now uses `dS_slot=n_pair`; the two temporal M16 halves select static columns 0 and 1 of the swizzled 16x64 int8 parent, while each N16 producer retains `n_tile & 1` ownership. A source-distinct CUDA proof matches all 2,048 MMA-A bytes against the separate-slot reference. The refined candidate reuses one MMA-A fragment across both loads, adds no barrier or global atomic, and passes the 134-case CUTLASS suite plus Racecheck/Synccheck for NHD/HND at 512/513. Four-block paired timing gives `1.004x` native-plus-conversion and `1.006x` end-to-end geometric speedup; the candidate is retained as a speed result despite a small register increase.
- Direct producer P/dS register handoff. Each warp converts its two M16 producer C fragments into the N16xM32 MMA-A fragments used by dV and dK with a proven `MOVM`, lane-rotation, byte-permutation, `MOVM` network. The dQ mirror remains shared because it is cross-warp. Relative to the saved control, each specialization removes 32 static `STS` and two `LDSM` instructions and adds 16 `MOVM` and eight `SHFL`; tail PRMT count falls by four while aligned PRMT count rises by 28. Both variants remain at 243 registers, 57,600 bytes shared, four static CTA barriers, and zero stack/local spill. Focused tests pass; dK/dV are bitwise identical; Racecheck and Synccheck pass NHD/HND at 512/513. Paired same-build timing reports `0.9941x` aggregate raw latency, and matched NCU reports `0.9940x` duration with about 60% fewer shared-store instructions and wavefronts. The two candidate/control orders did not disagree enough to warrant separate result rows. The handoff is retained.
- Single K-scale load site. Both aligned and tail specializations call `load_k_block_scale()` at one source location inside the M-pair loop. Resources remain 243 registers with zero stack/local spill and tail SASS is unchanged; ptxas emits eight additional aligned instructions compared with the manually split lifetime. Short paired timing was mixed, so this remains a source simplification without a speedup claim and should not be reopened without a new scale-lifetime design.
- Dead shared P-scale publication removal. `SharedStorage2DWarp::p_scale` was an unreferenced trailing eight-float array; the active local `p_scale` arithmetic is unchanged. Removing the field reduces shared storage from 57,632 to 57,600 bytes. Tail and aligned native-SM86 instruction streams are exactly identical to the control, resources remain 243 registers with zero stack/local spill, and the focused four-case matrix passes. This is retained as a resource cleanup without a latency claim.

### Closed Experiments

#### Transposed dQ Operand-Role Mirror

The transposed-dQ experiment evaluated `dQ^T = K^T @ dS^T` without changing the public dQ layout, internal fp32 workspace size, scale arithmetic, four-barrier schedule, or global contribution count. Complete hierarchical CuTe TV-layout inspection disproved the initial flattened-layout assumption: score-C/A and INT8 MMA-B fragments contain the same scalar extent but have different lane/value ownership. A layout-only `retile_D`, `partition_A`, or `partition_B` cannot move register values across lanes.

The corrected implementation expressed the exchange as a custom CuTe `Copy_Atom`. Its source and destination TV layouts describe the score-C-to-MMA-B mapping, while the hardware operation uses four warp shuffles and two byte permutations per eight-byte producer fragment. A typed `(N32 pair, M half, lane, N16 producer, word)` CuTe shared layout then combines the two producers with one 64-bit store per producer and one 128-bit consumer load. `tests/cuda/test_qattn_cutlass_bwd_ds_transposed_packet.cu` uses source-distinct coordinate values and matches all 2,048 MMA-B bytes across four N32 pairs and both M halves. The production candidate also prepacked `K^T` directly as MMA-A; reinterpreting the existing MMA-B packet as A had failed because the full A/B TV layouts are not identical. After that correction, all six focused NHD/HND aligned/tail cases passed.

The candidate still failed the cost gate. Relative to the retained pair-interleaved control, tail/aligned resources changed from `235/234` to `228/236` registers per thread, with zero stack/local spill in both builds. Aligned SASS grew from 2,024 to 2,072 instructions and tail SASS from 2,192 to 2,456. In each specialization, eight dS `LDSM.x4` instructions became eight `LDS.128` instructions, eight narrow mirror stores became two `STS.64` instructions, `SHFL` increased by eight, and IMMA, four static CTA barriers, and 16 global dQ atomics remained unchanged. The transposed workspace-coordinate path accounts for much of the additional tail address and predicate code.

One same-build screen block per layout used 25 warmups and 100 repeats, with candidate/control order reversed between NHD and HND. Every row regressed; the margin is much larger than the laptop's sub-percent clock drift. Ratios below are candidate latency divided by control latency, so values above one are slower.

| Layout | Heads | Sequence | Native plus conversion | End to end |
|---|---:|---:|---:|---:|
| NHD | 16 | 4096 | `1.054x` | `1.064x` |
| NHD | 16 | 8192 | `1.078x` | `1.068x` |
| NHD | 32 | 4096 | `1.060x` | `1.048x` |
| NHD | 32 | 8192 | `1.063x` | `1.066x` |
| HND | 16 | 4096 | `1.087x` | `1.087x` |
| HND | 16 | 8192 | `1.085x` | `1.093x` |
| HND | 32 | 4096 | `1.088x` | `1.080x` |
| HND | 32 | 8192 | `1.084x` | `1.075x` |

The geometric-mean regression is `1.075x` for native plus conversion and `1.073x` end to end. Raw rows are in `build/bench_sagebwd_transposed_dq_screen.csv`. A four-block promotion run and candidate sanitizers were intentionally skipped after this decisive performance rejection. The pair-interleaved mirror is restored in production; only the standalone exact CuTe mapping proof remains as a reproducible design artifact. The rebuilt tail/aligned target functions are instruction-for-instruction identical to the saved pair-interleaved control at 2,192/2,024 SASS instructions and 235/234 registers with zero spill. The restored image passes all 134 CUTLASS forward/backward tests and the full NHD/HND 512/513 Racecheck/Synccheck matrix.

#### Shared-Memory Excessive-Wavefront Attribution

A fresh native-SM86 `-lineinfo` SourceCounters capture at `B=1, H=16, N=8192, D=64` separates real serialization from Nsight Compute's raw shared-conflict field. The ordinary `LDSM.16.M88.4` instructions carrying the repeated raw `8,388,608` value have observed wavefront counts equal to ideal and therefore contribute no excessive wavefronts. The retained kernel has `316,080,128` observed shared wavefronts, `303,431,680` ideal wavefronts, and `12,648,448` excessive wavefronts, or `4.002%` of observed traffic. Two Q/dO packet-preparation `LDSM.16.MT88.4` instructions contribute `4,194,304` each. `LDGSTS`/`cp.async` contributes the other `4,259,840`, dominated by two Q/dO publications at `2,088,960` each. Ordinary `LDSM.16.M88.2`, non-transposed `LDSM.16.M88.4`, scalar `LDS`, `LDS.128`, and all shared stores are at their ideal wavefront count. Bank analysis must therefore use `L1 Wavefronts Shared Excessive`, not the raw N-way conflict count.

An isolated source-layout probe established the local swizzle tradeoff. Across 64 `LDSM.x4.trans` executions, a plain layout and `Swizzle<2,0,2>` each require `1,024` wavefronts versus `256` ideal, the retained `Swizzle<2,0,3>` requires `512`, and `Swizzle<3,0,4>` reaches the ideal `256`. Applying the bank-clean layout to both Q and dO removes all `8,388,608` transposed-load excess, but makes four much hotter non-transposed Q score loads two-wave and adds `33,554,432` excessive wavefronts; total excess rises to `37,814,272`. A global Q/dO swizzle is therefore decisively worse even though it fixes the targeted instructions.

The only selective simple candidate was a dO-int8-only `Swizzle<2,0,4>`, because dO int8 has no non-transposed score-load consumer. It passes all six focused aligned/tail NHD/HND cases and removes one `4,194,304`-wavefront transposed site without adding another conflict. Total excessive wavefronts fall `33.2%`, from `12,648,448` (`4.002%`) to `8,454,144` (`2.711%`). The complete kernel cost moves in the opposite direction: tail/aligned SASS grows by 24 instructions from `2,192/2,024` to `2,216/2,048`, registers rise from `235/234` to `239/238`, and shared memory, four barriers, atomics, stack, and local spill are unchanged.

Four independent 25-warmup/100-repeat blocks per layout used alternating candidate/control order and median-of-four block medians. Ratios are control latency divided by candidate latency, so values below one are regressions.

| Layout | Heads | Sequence | Native plus conversion | End to end |
|---|---:|---:|---:|---:|
| NHD | 16 | 4096 | `0.9951x` | `0.9745x` |
| NHD | 16 | 8192 | `0.9783x` | `0.9868x` |
| NHD | 32 | 4096 | `1.0082x` | `0.9909x` |
| NHD | 32 | 8192 | `0.9955x` | `1.0042x` |
| HND | 16 | 4096 | `1.0048x` | `0.9973x` |
| HND | 16 | 8192 | `1.0001x` | `0.9971x` |
| HND | 32 | 4096 | `0.9930x` | `0.9962x` |
| HND | 32 | 8192 | `0.9986x` | `1.0008x` |

Geometric means are `0.9942x/0.9890x` native/end-to-end for NHD, `0.9991x/0.9979x` for HND, and `0.9967x/0.9934x` over both layouts. Raw rows are in `build/bench_sagebwd_do_only_swizzle_{NHD,HND}_all.csv`, and the source counters are in `build/ncu_sagebwd_qdo_swizzle_do_only_source_seq8192.csv`. The dO-only swizzle is rejected: reducing a low-frequency bank penalty does not offset its address-generation and register cost in the whole kernel. The retained pair-interleaved image and original Q/dO layout are restored.

- Fixed global P scaling. Rejected because it destroys long-sequence dV signal.
- Exact Q32/K128 single-domain geometry. Rejected after regressing the governing long-sequence controls.
- Mirror-only dS-to-dK handoff. Rejected after timing regression and increased resource pressure.
- Power-of-two dS scaling. Rejected as a maintained implementation. It provided only approximately `0.6-1.0%` gains and its specialization/dispatch was removed.
- Periodic dS scaling. Rejected on accuracy despite approximately `4.3-5.1%` main-kernel gains; its specialization/dispatch was removed.
- Sparse dS. Rejected as the primary path. Oracle selection needed approximately `90%` block retention and still reached worst dQ/dK relative errors of `11.39%/21.39%`.
- Tested preprocess-only second-order/sparse scale formulations. A 2x4 centroid/RMS sketch improves calibration and a 64-pair sparse correction reaches five of six aggregates and 52 of 60 heads, but neither it, 256-pair sampling, held-out mean/quantile regression, nor exact correction of the 32 worst scales clears all six aggregates. These exact formulations are not promoted. Keep the broader block-sketch design open only if a new feature supplies missing softmax-concentration information and first passes reconstruction without a CUDA integration.
- Random-projection QK scale prediction. Rejected because softmax exponentiates sketch error; 8/16/32-dimensional variants pass zero of six aggregate controls and create excessive dS zeros.
- True pre-forward prediction. Rejected because same-timestep dOutput RMS differed by approximately `28x`; same-RMS random dOutput produced p99 scale ratios up to `31.5x` and a maximum around `642x`.
- Runtime dS-policy switching. Rejected in favor of one unconditional predictor; dynamic, power-of-two, and periodic IDs and the `PREDICT_DS_SCALE` toggle are no longer part of the active API or source.
- Exact on-the-fly K64 dS scaling as an unchanged default replacement. The current-source control restored one `abs(dS)` warp reduction, eight shared scale slots, and one CTA barrier per M32 pair. It passed focused correctness, all 60 exact-4096 per-head strict gates, and the full 512/513 NHD/HND Racecheck/Synccheck matrix. It remained spill-free at 243/231 tail/aligned registers and 57,632 bytes shared; aligned SASS shortened from 2,088 to 2,064 instructions, but barriers rose from four to five and `REDUX.MAX` from two to three. Candidate/control/candidate timing regressed every 4096/8192 NHD/HND kernel-only and end-to-end row by `1.5-3.6%`, so the unchanged synchronization protocol is closed as a performance candidate despite its accuracy value.
- Coordinated dV/dK group accumulation. Plain delayed INT32 accumulation under the independent source-order scales remains closed because it removes neither `I2FP` nor scaled `FFMA` and has no common exact product. A two-Q32 carry group using exact current-pair dS products and ceil power-of-two products was reconstructed and implemented with exact dS maxima, shared scale metadata, INT32 group fragments, and an FP16 outer accumulator. It passed the exploratory CUDA format gate on all six aggregates and 60 heads; the hardest seed-2 sigma-900 aggregate measured dQ cosine/relative/max `0.99899/4.49%/0.237`, dK `0.99786/6.54%/0.084`, and dV `0.99986/1.70%/0.019`, so it missed the retained strict gates and would require an explicit relaxation. The corrected native implementation used 255/242 registers, added a cross-warp maximum reduction and barrier, removed 37 static dK FFMA instructions but added 30 FMUL and 16 HADD2 instructions, and regressed paired kernel-only latency by `1.7%/1.9%` NHD 4096/8192 and `0.6%/2.3%` HND 4096/8192. It is closed without a justified tolerance relaxation because it has no speed benefit.
- FP16 dK outer accumulation. The repository-style mixed path was replayed with FP16 dK accumulation of each independently scaled INT8 partial while retaining FP32 dV, dQ, and all source-order arithmetic. It preserved the retained predictor's `4/6` aggregate and `42/60` head decisions, and exact dS replay remained `6/6` and `60/60`; additional dK error was approximately `0.01%` on the retained captures. The native candidate reduced the two specializations to 223/217 registers with zero stack/local spill, but replaced 32 FFMA with 32 FMUL plus 16 HADD2 and regressed paired kernel-only latency by `1.8%/1.8%` NHD 4096/8192 and `2.6%/3.0%` HND 4096/8192. Lower register count did not overcome the extra conversion path.
- SM86 fixed-shift `I2F.F16`/`HFMA2` primitive. A native probe confirmed scalar `I2F.F16` and `HFMA2` exist, so a custom primitive shifted each bounded K32 INT32 partial by three, packed two half conversions, and folded the factor eight into a half2 scale. The bound was finite (`516128 >> 3 = 64516`), but representing the small dK scale as one FP16 factor underflowed on sigma-300; native dK cosine fell to `0.426`/`0.396` on the two captures. SASS also grew by 32 instructions in the aligned specialization, with 32 `SHF`, 32 `I2F.F16`, 16 `PRMT`, and 16 `HFMA2` replacing the FP32 fused path. The primitive is closed; a two-factor scale would add another packed multiply and has no credible latency case.
- Local dQ shuffle and alternate shared layouts. The historical bank-aware shared transpose beat the prior shuffle path in all same-build timing controls, but the later swizzled global workspace superseded both. Direct/transposed, row-major, affine-swizzled, fp16-stage, int32-stage, and synchronization variants either regressed, increased conflicts/resources, or failed to improve the full kernel. Reopen dQ only with a different ownership or global-workspace contract.
- Q/dO early-load and tail-prefetch schedules. Early dO FP16 loading was a small regression or mixed under same-build NHD/HND controls. The tested tail-prefetch schedule was clearly slower, and changing q32/k64 to q128/k128 did not rescue it. The current placement already overlaps next-pair asynchronous loads with dQ, so these source-local reorderings are closed.
- The tested broad Q/dO packetization, broad vector-packet, register-V, and split dV/dK formulations are closed because their validated builds were slower or resource-heavy. Revisit the broader ideas only with a materially different layout or ownership hypothesis and source/SASS evidence that targets a measured handoff cost.
- N256 rejected lifetime variants. Replacing `__maxnreg__` with `__launch_bounds__` did not alter allocation. Keeping even one V fragment resident increased spill, and serializing the two dQ M halves increased stack and local traffic. Publishing the 8 KiB quantized-P operand to shared memory before the existing barrier and reloading it for dV raised tail/aligned stack from 136/112 to 152/136 bytes and spill store/load bytes from 156/176 and 116/116 to 168/180 and 136/136. Reloading producer dS from the already-required pair-interleaved mirror after the barrier marginally reduced tail spill but raised aligned stack to 120 bytes and grew tail/aligned SASS by 88/16 instructions; the added shared loads and transpose network failed the compiler gate. Requesting 129 registers emitted 129 rather than rounding down, worsened the tail stack to 152 bytes, and cannot fit a 512-thread CTA within the SM86 register file. These source-local lifetime variants are closed; a further N256 gain needs a different accumulator or ownership structure.
- Dimension-owned dK/dV without Q/dO packets. The isolated implementation assigned four dK dimension owners and four dV dimension owners, captured one direct Q or dO operand per owner, and traversed all eight N16 score slots. dQ used four private 1 KiB regions in dead resident-V storage. Racecheck exposed an ordering bug in the first implementation: direct operand loads after the publication barrier could overlap the next-pair asynchronous Q/dO prefetch. That run was discarded. Capturing the fragments before the barrier and deferring prefetch until after dQ fixed the race; the corrected build then passed Racecheck and Synccheck for NHD/HND at 512/513 and focused correctness, with sequence-4096 differences below `6.10e-5` maximum and `1e-5` relative L2. Only then was the formulation closed because it used a 96-byte aligned stack frame and regressed serial 4096/8192 kernel-only and end-to-end controls by approximately `15-21%`.
- Plain 32-byte-row dS/dKV scratch. Rejected despite reducing each warp scratch slot from 1,024 to 512 bytes. A dedicated fragment probe showed zero mismatches between the existing C-fragment store/MMA-A load path and the compact layout, and the reused `16x16` FP16 dK/dV epilogue stage fit exactly. The candidate reduced dynamic shared memory from 57,632 to 53,536 bytes, tail registers from 243 to 223, and static SASS by 8/9 instructions in tail/aligned variants, with no stack/local spill. Focused tests and the full 512/513 NHD/HND Racecheck/Synccheck matrix passed; dK/dV were bitwise identical at sequence 4096 and dQ differed only by `3.05e-5` maximum with `7.43e-6` relative L2 from cross-run atomic ordering. Candidate/control/candidate timing tied NHD 4096 but regressed HND 4096 by approximately `1.7-3.2%` and the 8192 rows by approximately `1.1-3.7%`. Matched sequence-4096 NCU counters explained the loss: shared-load bank conflicts rose from `2,097,152` to `10,485,760`, store conflicts from `542,057` to `4,510,420`, and load/store wavefronts by approximately `11.1%/14.1%`. Keep the swizzled layout unless a compact replacement also proves its bank behavior.
- Pre-dK/dV publication-barrier removal. Rejected after an isolated same-order run. Native Racecheck reported zero hazards for NHD/HND at 512/513, but aligned registers increased from 243 to 249. NHD was effectively tied, while HND regressed from 16.1038 to 17.2969 ms at sequence 8192 and from 3.9695 to 4.2977 ms at sequence 4096. This is the barrier that publishes packed Q/dO and mirrored dS, and it remains in the active kernel; it is distinct from the removed post-dK barrier before dQ.
- Direct two-source canonical dS gather. Rejected as a low-instruction mirror replacement after correcting the standalone oracle. The invalid production candidate loaded canonical transposed dS as though it had the mirror's non-transposed orientation, causing all four focused dQ cases to fail while dK/dV remained unchanged. The corrected source-distinct oracle, all uniform lane shifts, normal/transposed x2 atoms, and all lane-bit-permutation/XOR address maps established that an exact fragment needs a substantial cross-lane byte transpose. No candidate timing or resource result from the invalid build is retained.

### Closed Design Constraints

- Keep the four current CTA barriers. The startup K/V publication barrier is Racecheck-proven, the pre-dK/dV barrier publishes Q/dO packets and dS, and the end barrier protects next-pair publication. Do not use barriers as ptxas register-allocation boundaries.
- Do not treat packet compaction as free storage. Active Q/dO/K packet allocations are exact for their fragment counts; double buffering needs a proven dead-lifetime alias or a new schedule.
- Do not retry the plain 32-byte-row dS scratch without a bank-aware layout: it saved 4 KiB and 20 tail registers but increased shared conflicts and regressed timing.
- Do not retry the eight-warp dimension-owned Q/dO-elimination schedule without a way to avoid all-slot score reloads and the aligned stack frame.
- Treat broad packetization, register-V, direct/transposed dQ, early-dO, tail-prefetch, split dV/dK, periodic/power-of-two dS, pre-dK barrier removal, unchanged exact-max K64 dS synchronization, coordinated dV/dK grouping, FP16 dK outer accumulation, and the fixed-shift SM86 conversion primitive as closed controls unless a materially different representation changes the measured cost model.
- Treat dV/dK INT32 conversion and scaled-FP32 accumulation as compulsory under the retained independent scale domains. Reopen only with a representation that proves a common product, bounded integer rescaling, accuracy, resource fit, and removal of hardware work rather than movement between FP16 and FP32.
- Keep direct dQ-mirror elimination closed until a producer-to-consumer proof beats the known three-MOVM/two-lane/two-byte reconstruction cost. The producer-local dV/dK handoff does not make the cross-warp dQ mapping cheaper. The corrected transposed operand-role formulation is also closed: its exact CuTe C-to-B `Copy_Atom` needs a material cross-lane shuffle/permutation network and regressed every measured row by `4.8-9.3%`.
- Keep the fixed predictor as the performance reference and exact-max reconstruction as the accuracy oracle. Do not retry a scalar guard, simple P-max multiplier, generic QK random projection, or the tested 2x4-plus-row-sample formulations.
- Use the retained producer-register handoff and swizzled global dQAccum workspace as the native local baseline. Do not restore canonical P/dS score-pair stores or treat score-pair slots as free storage; the shared score-pair staging used by the pre-swizzle dQ path is gone.
- Treat the four barriers and the `0.43` eligible-warps/cycle result as diagnostics, not permission to remove synchronization. Any new schedule must identify a different ownership dependency and pass the full native sanitizer matrix.
- Keep N256's 128-register cap. The current 512-thread CTA cannot request 129 registers, and uncapped allocation remains above the hardware limit. Reopen N256 lifetime work only with a materially different accumulator or phase-ownership design that compiles at no more than 128 registers and beats both the retained N256 image and unchanged N128.

## Next Work

The next cycle uses QBlock=32, KBlock=64, CTA M64xN128, eight warps, and `smooth_k=False` as the measured reference, with the optimized N256 config as the architectural comparison. Smooth K remains an explicitly selectable correctness/accuracy mode rather than the default benchmark result. Native smooth-K ownership and predictor evaluation are complete. The source-local N256 lifetime cycle is also complete: on-demand V reload is retained, while partial V caching, serialized dQ halves, shared-P staging, and register-cap alternatives are closed. Each new candidate must state whether it changes the artifact, accuracy, ownership, scale, precision, or product contract. Preserve the current arithmetic and synchronization contract when that category is not the subject of the experiment.

### Deferred

- Combined forward/backward state routing. The normal head-64 forward and native backward share the Q32/K64 scale contract, but the public forward still discards those tensors. A future state-owning forward, internal-quantization compatibility path, combined public API, and full-training benchmark must define and measure that contract together.
- Head dimension 128 native backward. The public CUTLASS forward uses Q32/K64 at head dimension 128, but the native backward specialization remains head-64 only. Reusable `HeadDim`/tile traits and generated-instantiation plumbing are retained; the historical generic `M64xN64`, four-warp D128 backward used 250 registers per thread and 41,424 bytes of shared memory and was not competitive. Resume with a dedicated D128 packet-ownership and register-lifetime design, using M64xN64 only as a bring-up geometry, rather than adding a generated instantiation to the D64 kernel.
- Split dQ workspace and separate reduction. This mirrors Flash's deterministic ownership option and may replace contended atomics with regular stores, but it adds `O(splits * B * H * N * D)` fp32 traffic, workspace, a launch, and a reduction-order change. Promote only on end-to-end latency, not main-kernel duration alone.
- dS-load-amortizing dQ ownership. The four even dQ warps reread the same dS mirror for different D16 output slices. A candidate may assign multiple D16 slices to one consumer warp, or make producer warps compute local dQ contributions followed by a hierarchical in-CTA reduction, so that a dS fragment is reused or never leaves the producer. The cost model must include additional dQ accumulators, producer-local K loads, shared FP32 partials, barriers, atomic contribution count, and register pressure at the current 235/234-register tail/aligned baseline. Eliminating the mirror is not a win if it merely creates more dQ partials or a larger reduction.
- Keep predictor work as a separate accuracy workstream. If it resumes, target missing softmax-concentration information rather than a larger generic guard. The strongest remaining hypothesis is a forward-owned row concentration statistic or compact top-block summary that the forward softmax can emit in `O(N)` storage; a preprocess-only alternative needs a new block-conditioned score feature that beats the tested centroid and 16x16 samples. Account explicitly for a Q32-by-K64 scale table and block-pair combine in end-to-end timing.
- Revisit quantization blocks only after the artifact contract and accuracy policy are settled. Generate every shape through `scripts/generate_cutlass_bwd_instantiations.py` and evaluate accuracy, resources, occupancy, shared counters, sanitizers, and both timing modes.
- Forward-owned P maxima or P-scale metadata are deferred as a narrow scale-reduction optimization. Saving one P scale per Q32/K64 block could remove the local P maximum reduction and related scale arithmetic, but it cannot remove QK recomputation, exponentiation, P reconstruction, or dV's P operand. The metadata is still an `O(N^2)` block table, so its forward store, backward load, workspace ownership, and end-to-end cost must be counted. Full quantized-P checkpointing remains outside the memory contract.
- INT8 V for dP. Quantize V with a bounded K64/D-block scale and use the existing INT8 dOutput with INT8 V for the dP MMA. This could remove the remaining FP16 dP HMMA path and the dO FP16 shared handoff, but it adds V quantization, a new scale product, and INT32-to-FP32 dP rescaling. Replay the retained captures offline first, including dS and final dQ/dK/dV accuracy, saturation, zero rates, and preprocessing cost. This is a precision/product-contract candidate, not a source-only handoff cleanup.
- INT4 remains deferred until the INT8 path has a stronger accuracy margin and a demonstrated end-to-end need.

### Promotion Requirements

Every promising candidate must pass finite-output and accuracy gates, native Racecheck/Synccheck for aligned and tail NHD/HND inputs, SASS/resource inspection, and interleaved kernel-only/end-to-end timing at 4096/8192 in both layouts. Ownership or synchronization changes additionally require an explicit producer/consumer proof. A compile, mapping, correctness, or sanitizer failure triggers diagnosis and a corrected rerun; it is not performance evidence. Higher register or shared-memory use is acceptable only when overall measured speed justifies it; stack/local spill in an otherwise correct build remains a rejection signal.

Report kernel-only and end-to-end latency separately. Forward-artifact, split-workspace, extra-reduction, or predictor-table candidates must include their producer, workspace, and launch costs in end-to-end timing. Source-level NCU work requires a temporary `-lineinfo` build and restoration of the retained binary. The Triton reduction tree remains lower priority because it is `O(N*D)` and outside the native shared-store bottleneck.

## Standard Commands

Build the normal retained image with four jobs:

```powershell
Remove-Item Env:NVCC_APPEND_FLAGS -ErrorAction SilentlyContinue
$Env:MAX_JOBS = '4'
python setup.py build_ext --inplace
```

Focused correctness:

```powershell
python -m pytest -q tests/test_sageattn_cutlass.py tests/test_sagebwd_cutlass.py
```

Rebuild and run the exact source-distinct CuTe C-to-B mapping proof retained from the transposed-dQ experiment:

```powershell
nvcc -std=c++20 -arch=sm_86 -Xcompiler=/Zc:preprocessor `
  -I. -Ithird_party/cutlass/include `
  tests/cuda/test_qattn_cutlass_bwd_ds_transposed_packet.cu `
  -o build/test_qattn_cutlass_bwd_ds_transposed_packet.exe
./build/test_qattn_cutlass_bwd_ds_transposed_packet.exe
```

The expected result is `transposed dS packet: all 2048 fragment bytes matched`.

Benchmark every Q32/K64 forward execution tile for both supported head dimensions against FlashAttention's library-selected kernel:

```powershell
foreach ($layout in @('NHD', 'HND')) {
  python bench/bench_qattn_cutlass.py `
    --batch-size 1 --num-heads 16 32 --seq-lens 4096 8192 `
    --head-dims 64 128 --layout $layout --warmup 25 --repeats 100 --mode all `
    --csv "build/bench_qattn_cutlass_rawk_${layout}_4096_8192.csv"
}
```

Native-only Racecheck and Synccheck matrix:

```powershell
$cases = @(
  @{ Layout = 'NHD'; SeqLen = 512 },
  @{ Layout = 'NHD'; SeqLen = 513 },
  @{ Layout = 'HND'; SeqLen = 512 },
  @{ Layout = 'HND'; SeqLen = 513 }
)

foreach ($case in $cases) {
  compute-sanitizer --target-processes application-only `
    --tool racecheck --racecheck-report all `
    --kernel-name 'kernel_substring=fused_mma_kernel_k128_8warp' `
    python scripts/racecheck_cutlass_bwd_kernel_only.py `
    --seq-len $case.SeqLen --layout $case.Layout
  if ($LASTEXITCODE -ne 0) {
    throw "Racecheck failed for $($case.Layout) $($case.SeqLen)"
  }

  compute-sanitizer --target-processes application-only `
    --tool synccheck `
    --kernel-name 'kernel_substring=fused_mma_kernel_k128_8warp' `
    python scripts/racecheck_cutlass_bwd_kernel_only.py `
    --seq-len $case.SeqLen --layout $case.Layout
  if ($LASTEXITCODE -ne 0) {
    throw "Synccheck failed for $($case.Layout) $($case.SeqLen)"
  }
}
```

Compute Sanitizer's kernel filter is a key/value expression, `kernel_substring=...`; it is not NCU's `regex:...` syntax. Keep `--target-processes application-only` and use the native-only helper so PyTorch setup, Triton compilation, and unrelated child-process kernels are outside instrumentation. Racecheck should end with `0 hazards displayed (0 errors, 0 warnings)` and Synccheck with `ERROR SUMMARY: 0 errors` for all four cases.

Benchmark both layouts and modes in four alternating-order blocks:

```powershell
foreach ($layout in @('NHD', 'HND')) {
  python bench/bench_sagebwd_cutlass.py `
    --batch-size 1 --num-heads 16 32 --head-dims 64 `
    --seq-lens 4096 8192 --layout $layout `
    --warmup 25 --repeats 100 `
    --block-configs 32,64,64,128 --mode all `
    --csv "build/bench_sagebwd_cutlass_final_swizzled_${layout}_a.csv"
  python bench/bench_sagebwd_cutlass.py `
    --batch-size 1 --num-heads 32 16 --head-dims 64 `
    --seq-lens 8192 4096 --layout $layout `
    --warmup 25 --repeats 100 `
    --block-configs 32,64,64,128 --mode all `
    --csv "build/bench_sagebwd_cutlass_final_swizzled_${layout}_b.csv"
  python bench/bench_sagebwd_cutlass.py `
    --batch-size 1 --num-heads 16 32 --head-dims 64 `
    --seq-lens 8192 4096 --layout $layout `
    --warmup 25 --repeats 100 `
    --block-configs 32,64,64,128 --mode all `
    --csv "build/bench_sagebwd_cutlass_final_swizzled_${layout}_c.csv"
  python bench/bench_sagebwd_cutlass.py `
    --batch-size 1 --num-heads 32 16 --head-dims 64 `
    --seq-lens 4096 8192 --layout $layout `
    --warmup 25 --repeats 100 `
    --block-configs 32,64,64,128 --mode all `
    --csv "build/bench_sagebwd_cutlass_final_swizzled_${layout}_d.csv"
}
```

Take the median of the four block medians for each row, then write the consolidated rows to the two `bench_sagebwd_cutlass_rawk_{NHD,HND}_4096_8192.csv` paths used above.

Profile one active main-kernel launch:

```powershell
python build/profile_sagebwd_once.py sage `
  --head-dim 64 --seq-len 512 --num-heads 16 --warmup 1 `
  --block-config 32,64,64,128
```

For a source-level native-only Nsight Compute capture, create a temporary `-lineinfo` build:

```powershell
$Env:NVCC_APPEND_FLAGS = '-lineinfo'
$Env:MAX_JOBS = '4'
python setup.py build_ext --inplace
```

Then capture the active kernel:

```powershell
ncu --set full --profile-from-start off `
  --target-processes application-only `
  --import-source on --source-folders C:\sageattention-autotune `
  --kernel-name 'regex:fused_mma_kernel_k128_8warp' `
  --launch-count 1 --export build/ncu_sagebwd_final_native_8192_source.ncu-rep `
  python scripts/racecheck_cutlass_bwd_kernel_only.py `
  --seq-len 8192 --num-heads 16 --layout NHD
```

After exporting the source profile, clear `NVCC_APPEND_FLAGS` and restore the retained non-lineinfo binary before timing.

For the current-build raw counter pair, use `--page raw --csv --print-units base` with `--profile-from-start off`, `--target-processes application-only`, `--kernel-name-base function`, and the filtered `fused_mma_kernel_k128_8warp` name. Query the metric definitions with `--query-metrics --chips ga103` when needed, but omit `--chips` from the profiling command in Nsight Compute 2026.2.1; that option is query-only in this release. The retained counter set includes `gpu__time_duration`, `smsp__inst_executed`, shared load/store instructions and wavefronts, global RED instructions and sectors, L2 read/write sectors, DRAM bytes, eligible warps, and barrier-stall share. Nsight Compute duration is architectural evidence only; use CUDA-event ABBA measurements for latency claims.

# QAttn CUTLASS Backward Kernel Plan

## Current State

The maintained backward path targets native SM86, fp16, dense non-causal fixed-length attention with head dimensions 64 and 128 and NHD/HND layouts. Its dS mirror uses one swizzled 16x64 int8 parent per N32 pair, with static 16x32 M-half views for dQ. Three explicitly configured specializations are generated:

```text
D64:  QBlock=32, KBlock=64, CTA M64xN128, eight warps (default)
D64:  QBlock=32, KBlock=64, CTA M64xN256, sixteen warps (experimental)
D128: QBlock=32, KBlock=64, CTA M64xN64, eight warps (maintained reference)
```

`_BWD_CONFIG` remains the D64 `(32, 64, 64, 128)` default, while `_BWD_CONFIG_HD128` is `(32, 64, 64, 64)`. Head-specific configuration tables also expose the D64 N256 experiment. There is no backward autotuner; the public wrapper selects the first maintained configuration for the padded head dimension. The generator emits one translation unit per launch specialization using the stable head/config filename, so the current three launches occupy three units and remain below the four-unit ceiling. The old filename/module `sm80` is a legacy identifier; `setup.py` emits only `compute_86,sm_86`. All configured paths use the fixed Q32/K64 metadata contract, the separable dS predictor, and the internal swizzled dQAccum workspace described below.

Both configured paths support raw and centered K. The public wrapper defaults to `smooth_k=True`, matching CUTLASS forward. Callers always provide the corrected natural-log LSE in the raw-score domain. For smoothing, Python recomputes the sequence K mean, quantizes `K - mean(K)`, and converts LSE to the centered-score domain with `LSE_centered = LSE_raw - softmax_scale * dot(Q, mean(K))`. The native kernel accumulates one exact reconstructed-dS row sum per query row, and Triton restores the missing dQ term `dS_sum * mean(K)` while inverting the swizzled dQ workspace. Raw K compiles out the row reduction and reuses `Delta` as an ignored operator placeholder, so it does not allocate or clear the padded dS-sum workspace.

The remainder of this plan is organized by contract, pipeline, correctness, final performance, bottlenecks, and completed or deferred work. The public forward still discards its quantized Q/K artifacts, so the backward timing is a reuse proxy rather than an integrated training benchmark; product and API limitations are collected under `Deferred`.

### Supported Contract

- Inputs: fp16 Q, K, V, output, and dOutput; float32 LSE.
- Layouts: NHD and HND.
- Attention: dense, non-causal, fixed length.
- K policy: `smooth_k=False/True`; LSE is natural-log and corrected to the raw-score domain at the public boundary.
- Head dimension: 64 or 128 after padding.
- Last dimension: contiguous; wrappers materialize contiguous inputs.
- Batch and head strides: arbitrary int32-compatible strides for the logical input layout.
- Workspace tensors: contiguous, explicitly typed, and stored in fixed `[batch, heads, sequence, dimension]` logical order where applicable.
- Out of scope: causal, variable length, GQA/MQA, and additional dtypes.

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

CUTLASS main kernel:
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

The D64 CTA schedule and shared-memory lifetimes are:

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

There are four static CTA barriers. The D64 resident-V region is aliased by the Q/dO packet publication only after V fragments have been captured in registers. Its pre-dK/dV barrier publishes the Q/dO packets and mirrored dS; P and dS for dV/dK no longer cross shared memory. There is no post-dK barrier before dQ. The dQ MMA fragments atomically update the internal swizzled global dQAccum workspace directly; Triton applies the inverse mapping during final conversion. The end-of-iteration barrier both publishes the next asynchronous Q/dO/row-state load and prevents the next iteration from overwriting shared mirror and packet storage still in use.

The D128 M64xN64 schedule also has four static barriers but uses head-specific ownership. Each N16 tile has two warps, one per M16 half. The two halves exchange one common P scale, publish canonical P/dS plus the pair-interleaved dS mirror, and split final ownership so the lower-M warp accumulates dV while the upper-M warp accumulates dK. Two even-N owners each cover four D16 dQ slices and both N32 pairs. Q/dO use two shared stages: the startup barrier jointly publishes K/V, packed K, and the first Q/dO pair; the inactive Q/dO stage receives the next asynchronous prefetch; and the end barrier publishes that stage. This double buffer prevents inactive tail loader warps from overwriting the current Q/dO operands while the remaining active warp pair finishes dK/dV.

The current backward wrapper still quantizes Q/K on each call with `per_block_int8`. Forward uses the same Q32/K64 quantization domain for both head dimensions, but its internal tensors are discarded; the backward benchmark therefore prepares the compatible Q/K INT8 tensors and scales outside the timed region as a saved-state reuse proxy. The timed backward path includes Triton preprocessing, workspace/output allocation, the CUTLASS main kernel, and dQ conversion.

The public CUTLASS forward uses `per_block_int8(QBlock=32,KBlock=64)` for head dimensions 64 and 128 and returns only output and optional LSE. Its Q/K scale tensors have the same flat artifact shape required by native backward, while attention CTA and warp dimensions remain independent execution choices. Native backward now consumes that contract at both head dimensions. No maintained public forward path routes saved Q/K artifacts into the native CUTLASS backward operator, and no maintained forward path publishes P, P maxima, or a backward dS scale table.

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

The grid maps `blockIdx.x` to one CTA-owned N block, `blockIdx.y` to head, and `blockIdx.z` to batch. D64 keeps one physical warp per N16 tile. N128 uses eight warps; its four even-`n_tile` warps cover the four D16 dQ slices and each owner sums all four N32 pairs. N256 uses sixteen warps; physical warps 0-3 own the four D16 dQ slices and each owner sums all eight N32 pairs across four K64 scale domains. N256 therefore halves the number of global dQ contributions relative to N128 without changing the public workspace or atomic element count per CTA.

D128 maps `n_tile=warp_id/2` and `m_half=warp_id&1`, so each N16 tile has a two-warp M-half pair. The two even N tiles own the lower and upper four D16 dQ slices; each M-half owner sums both N32 pairs and writes its own M16 rows. Every configuration atomically adds one accumulated M16 fragment per CTA directly into the common global fp32 dQ workspace. The internal workspace is in MMA TV order so each warp's fragment index is contiguous in global memory.

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
52 passed
```

The matrix covers raw/smooth K, NHD/HND, aligned/tail sequence lengths 64/65, the maintained D64 N128 and D128 N64 configurations, the experimental D64 N256 comparison, public head-specific selection, and the CuTe dQ workspace mapping and physical-to-logical conversion. During integration, two correctness issues were fixed:
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

These are the current full backward benchmark results for the accepted `406a481` image. They use `smooth_k=False` on an NVIDIA GeForce RTX 3080 Ti Laptop GPU with batch 1, 16/32 heads, sequence 4096/8192, and NHD/HND layouts. Each row used 20 warmups and 100 repeats in the same process as its freshly measured FlashAttention baseline. FlashAttention uses normal library dispatch and is not constrained to the Sage CTA shape.

`native+convert` is the CUTLASS launch plus Triton dQ conversion after prequantized Q/K and precomputed workspaces; it is named `kernel_only` in the raw CSV. `end-to-end` follows the backward-reuse contract: it reuses prequantized backward-format Q/K and includes Triton preprocessing, workspace/output allocation, the CUTLASS launch, and dQ conversion. Both scopes exclude Q/K quantization and the forward pass. Effective throughput uses `8 * batch * heads * head_dim * seq_len^2` backward FLOPs.

#### D64 Production Reference

| Layout | Heads | Seq | Flash median ms | Sage end-to-end ms | End-to-end speedup | Sage native+convert ms | Native+convert speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| NHD | 16 | 4096 | 5.123 | 4.329 | 1.183x | 4.174 | 1.227x |
| NHD | 16 | 8192 | 19.440 | 16.418 | 1.184x | 16.054 | 1.211x |
| NHD | 32 | 4096 | 9.968 | 8.488 | 1.174x | 8.319 | 1.198x |
| NHD | 32 | 8192 | 36.565 | 29.539 | 1.238x | 28.713 | 1.273x |
| HND | 16 | 4096 | 5.147 | 4.292 | 1.199x | 4.320 | 1.191x |
| HND | 16 | 8192 | 19.488 | 16.431 | 1.186x | 16.093 | 1.211x |
| HND | 32 | 4096 | 9.485 | 7.944 | 1.194x | 7.455 | 1.272x |
| HND | 32 | 8192 | 34.860 | 29.308 | 1.189x | 28.639 | 1.217x |

Across the eight D64 rows, end-to-end speedup is `1.174-1.238x` with a `1.1934x` geometric mean, and native+convert speedup is `1.191-1.273x` with a `1.2249x` geometric mean. D64 wins every row in both scopes.

#### D128 Maintained Reference

The current D128 M64xN64 reference was measured with the same `smooth_k=False`, 20-warmup, 100-repeat common-baseline protocol. It includes the retained FMA, P/dS hoist, and asynchronous dO-staging changes. FlashAttention still wins the aggregate, but the native kernel plus dQ conversion is now close to parity.

| Layout | Heads | Seq | Flash median ms | Sage end-to-end ms | End-to-end speedup | Sage native+convert ms | Native+convert speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| NHD | 16 | 4096 | 9.304 | 9.757 | 0.954x | 9.411 | 0.989x |
| NHD | 16 | 8192 | 34.707 | 36.290 | 0.956x | 35.771 | 0.970x |
| NHD | 32 | 4096 | 18.094 | 18.917 | 0.957x | 18.231 | 0.992x |
| NHD | 32 | 8192 | 69.902 | 73.450 | 0.952x | 72.214 | 0.968x |
| HND | 16 | 4096 | 9.499 | 10.045 | 0.946x | 9.771 | 0.972x |
| HND | 16 | 8192 | 35.437 | 36.777 | 0.964x | 36.062 | 0.983x |
| HND | 32 | 4096 | 18.364 | 19.041 | 0.964x | 18.341 | 1.001x |
| HND | 32 | 8192 | 70.480 | 73.524 | 0.959x | 72.405 | 0.973x |

Across the eight D128 rows, geometric speedup is `0.9563x` end to end and `0.9810x` native+convert. The HND H32/S4096 native+convert row is marginally ahead at `1.001x`; the other rows remain below Flash. D128 is retained for complete Q32/K64 support and as the head-specific optimization reference, while D64 N128 remains the production performance path.

Current raw rows are `C:/tmp/sageattn/experiments/sagebwd_planned_opts/fresh_406a481_flash_hd{64,128}_{NHD,HND}.csv`. Candidate/control timing and profile evidence used to accept or reject individual changes is recorded under `Retained Changes` and `Closed Experiments`; those ratios are not combined with this current Flash-normalized result.

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

The retained fixed-scale pipeline is dominated by P/dS reconstruction, INT8-to-FP32 scale application, fragment and packet handoffs, and synchronization around cross-warp ownership. Full historical Nsight tables are not maintained in this overview. CUDA-event results in `Performance Against FlashAttention` are the current latency reference; selected counters remain in the retained and closed experiment entries only when they explain a decision.

The stable conclusions from the profiling work are:
- The remaining gap is on-chip rather than DRAM-bound: scalar reconstruction, conversion, shared-memory issue, and fragment plumbing dominate the excess work.
- Sage and Flash use the same global dQ reduction count in the matched ownership comparison, so reducing atomics is not the current priority.
- Both D128 kernels had one resident eight-warp CTA in the matched attribution. A new occupancy claim requires a compiled schedule with lower register and shared-memory use.
- The accepted D64 workspace swizzle, D128 P/dS hoist, D128 asynchronous dO staging repair, and D64 lane-major packet handoff remove specific on-chip costs. Their resource, counter, and candidate/control evidence is recorded under `Retained Changes` and `Closed Experiments`.

#### Fresh Matched D64/D128 Profile

The current matched profile is under `C:/tmp/sageattn/experiments/d128_roofline_20260815`. It uses raw-K NHD, batch 1, sequence 4096, 16 heads, three warmups, one captured main-kernel launch, kernel replay, and the normal FlashAttention dispatch. NCU duration is architectural evidence only; the CUDA-event tables above remain the latency authority.

| Metric | Sage D64 | Flash D64 | Sage D128 | Flash D128 |
|---|---:|---:|---:|---:|
| NCU main-kernel duration | `4.508 ms` | `5.747 ms` | `11.390 ms` | `11.300 ms` |
| Total warp instructions | `374.45 M` | `155.68 M` | `811.04 M` | `278.73 M` |
| HMMA instructions | `8.39 M` | `41.94 M` | `16.78 M` | `83.89 M` |
| IMMA instructions | `16.78 M` | `0` | `33.55 M` | `0` |
| ALU instructions | `125.40 M` | `26.70 M` | `277.39 M` | `59.71 M` |
| FMA-pipe instructions | `131.87 M` | `37.00 M` | `199.13 M` | `43.58 M` |
| LSU-pipe instructions | `54.94 M` | `39.34 M` | `199.25 M` | `78.80 M` |
| Shared-load instructions | `9.86 M` | `1.59 M` | `24.71 M` | `3.19 M` |
| Shared-store instructions | `5.12 M` | `8.52 M` | `22.87 M` | `8.65 M` |
| Tensor-pipe active | `34.25%` | `44.72%` | `27.08%` | `45.52%` |
| Active warps per SM | `7.99` | `8.00` | `8.00` | `7.99` |
| Dynamic shared memory | `49,408 B` | `73,728 B` | `78,112 B` | `81,920 B` |
| Registers per thread | `239` | `255` | `252` | `255` |

The total-instruction ratio is `2.405x` for D64 and `2.910x` for D128. D128 executes `4.65x` Flash's ALU instructions, `4.57x` its FMA-pipe instructions, `2.53x` its LSU instructions, `7.75x` its explicit shared-load instructions, and `2.64x` its shared-store instructions. The D128 MIO-throttle ratio is `1.509` stalled warps per active issue cycle versus Flash's `0.560`; D64 is `0.428` versus `0.476`. This is the clearest head-dimension split: D64's retained schedule tolerates its scalar work and wins, while D128 loses tensor issue efficiency and multiplies shared/scalar plumbing.

D128 is not bandwidth-bound. The fresh launch reads `165.10 MB` and writes `102.68 MB` in `11.39 ms`, about `23.5 GB/s`; Flash moves slightly more DRAM data. A same-process NHD phase diagnostic also places the native launch at about `94%` of D128 end-to-end time at sequence 4096 and `97%` at 8192. Preprocessing plus dQ conversion is important for reporting scope, but it is not the first optimization target.

| Head dim / sequence | Preprocess | Native launch | dQ conversion | End-to-end |
|---|---:|---:|---:|---:|
| D64 / 4096 | `0.177 ms` | `3.617 ms` | `0.087 ms` | `3.815 ms` |
| D64 / 8192 | `0.233 ms` | `14.294 ms` | `0.123 ms` | `14.221 ms` |
| D128 / 4096 | `0.306 ms` | `8.670 ms` | `0.172 ms` | `9.203 ms` |
| D128 / 8192 | `0.712 ms` | `35.062 ms` | `0.342 ms` | `36.092 ms` |

These are independent diagnostic medians rather than additive timing brackets, so small ordering inconsistencies are expected. They quantify the ceiling: even eliminating D128 preprocessing and conversion entirely would not close the current native schedule gap.

#### D128 Shared-Memory Attribution

The accepted sequence-8192 SourceCounters report contains `754,974,720` LDSM wavefronts versus `620,756,992` ideal, with exactly `134,217,728` excessive wavefronts (`17.778%`). SASS/source-order mapping identifies all excess as two eight-site `LDSM.16.MT88.4` families: transposed INT8 dO loads used by dV and transposed INT8 Q loads used by dK. Each site executes `2,097,152` instructions and takes `16,777,216` wavefronts versus `8,388,608` ideal. Ordinary LDSM, scalar LDS/STS, and the accepted asynchronous LDGSTS families are otherwise at or near ideal.

An isolated D128 `32 x 8` uint128 layout probe screened plain storage and seven swizzles for both ordinary MMA-A and transposed MMA-B reads. The retained `Swizzle<3,0,3>` is Pareto-best: ordinary reads take `1,024/1,024` ideal wavefronts, while transposed reads take `2,048/1,024`. Every screened layout that kept the minimum transposed penalty made ordinary reads two- or four-wave. A global Q/dO layout substitution is therefore closed; the open target is the transposed copy path itself. Reproducible probe source, CSV, and summary are in the fresh profile directory.

#### Practical Rooflines

- Arithmetic peak is not the current limit. Flash backward performs five FP16 matrix products; Sage performs one FP16 and four INT8 products. At the SM86 dense `4:1` INT8-to-FP16 peak ratio, the ideal tensor-arithmetic floor is `2.5x` faster than five FP16 products. The compiled Sage kernels execute `0.60x` as many tensor warp instructions as Flash, which gives a more conservative `1.67x` equal-issue instruction ceiling. D128's `2.91x` total instruction count and lower tensor utilization erase that advantage.
- Matching the paper's approximately `1.25x` D128 backward ratio from the local `0.981x` native-plus-conversion position would require about a `1.274x` speedup, or a `21.5%` latency reduction. Raising effective D128 tensor utilization from `27.1%` to roughly the retained D64 level of `34-35%` has that arithmetic magnitude, but only if shared and scalar work are reduced enough to feed the tensor pipe. This is a scheduling target, not a prediction.
- Removing all currently measured D128 LDSM excess would reduce LDSM wavefronts by `17.8%`, but it touches only one on-chip family. It is a credible low-single-digit kernel candidate, not a complete route to the paper ratio.
- Two resident eight-warp D128 CTAs require at most about `50 KiB` shared memory and `128` registers per thread. A six-warp two-CTA design requires at most about `170` registers per thread and the same shared-memory bound. A four-warp two-CTA design permits the current register scale but still exposes only eight active warps, so it offers barrier staggering rather than more warp concurrency. These are hard compiler gates.
- Register allocation is kernel-wide per thread, not averaged across warp roles. A low-register helper role cannot compensate for a dKV-owner path above the 12-warp `170`-register or 16-warp `128`-register CTA limit.
- The current persistent dK/dV fragment is `8 D16 slices x 8 FP32 accumulator values = 64` registers per warp before score, dP, scale, and dQ state. Any dQ reuse or role-specialized schedule must account for this persistent image explicitly; otherwise a nominally lower-load design will spill.

#### Reduction Status

The exact on-the-fly `abs(dS)` maximum reduction is no longer present. The active kernel still pays for the local P scale reduction: scalar `fmaxf` updates build `p_max_abs`, followed by one warp-level `__reduce_max_sync` before P is quantized. This is a real non-tensor cost, but it is narrower than the removed dS maximum reduction and should be treated as one part of the P reconstruction path rather than as the main remaining dS-scale mechanism.

The Q32/K64 predictor summaries are computed in Triton with `O(N*D)` reductions. They do not create the native shared-store traffic. Replacing those reductions is lower priority unless an end-to-end phase breakdown shows that preprocessing, rather than the CUTLASS launch, has become material.

#### P and dS Reconstruction

Each K128 iteration reconstructs P and dS around the int8/fp16 MMA results. The path includes int32-to-fp32 conversion, score and softmax scaling, `expf`, LSE and Delta subtraction, dS formation, P-maximum tracking, and scale arithmetic. P and dS are then quantized with reciprocal-scale application and integer conversion. This combination explains a substantial part of the excess non-tensor instruction activity and includes the remaining P warp reduction.

#### Shared-Memory Handoffs

The D64 producer-local canonical P/dS score-pair handoff is removed. D128 still publishes canonical P and dS pairs because its lower-M dV owner and upper-M dK owner each consume both reconstructed M halves. The active path materializes several fragments through shared memory:
- D128 canonical P/dS pair stores and owner reloads for dV/dK. The proposed half-pair handoff targets only this exchange.
- dS mirror stores into one swizzled `dS_dKV` parent per N32 pair so the dQ path can consume both reconstructed M halves through static 16x32 MMA-A views.
- D64 packed Q/dO fragment stores and loads, and D128 transposed reads from the raw Q/dO stages, for the dK/dV MMAs.
- Packed K fragment stores and later loads for the dQ MMAs.
- Swizzled global dQAccum storage for direct atomic accumulation; the inverse permutation is deferred to Triton conversion, so the active kernel has no shared dQ transpose.
- dK/dV epilogue staging before coalesced global stores.

Per temporal D128 M32 pair, the canonical P/dS exchange logically stores about `4 KiB` and reloads about `4 KiB`: four N16 tiles each publish `512 B` of P and `512 B` of dS, then four dV owners load one P pair and four dK owners load one dS pair. Keeping the local C-fragment half in registers and publishing only the remote half reduces both sides to about `2 KiB`. The lower-M producer publishes its `256 B` dS half, the upper-M producer publishes its `256 B` P half, and the existing proven C-to-A helper fills the corresponding half of the owner fragment. The dS mirror remains intact for dQ.

#### Scalar Scale Application and Fragment Plumbing

The dV, dK, and dQ accumulation paths apply FP32 products involving P, dS, dOutput, Q-block, K-block, and output scales around the integer MMA fragments. Manual packed-fragment indexing, address arithmetic, tail predicates, conversions, and repeated synchronization add further non-tensor instructions and scheduler stalls. These operations also constrain attempts to keep P/dS values in registers because producer lifetimes already approach the resource limit.

#### Remaining-Work Classification

| Stage | Status under the retained contract | Consequence |
|---|---|---|
| dV/dK conversion and scaled accumulation | Exhausted locally | Independent P/dS, Q, and dOutput scales require conversion and FP32 products; grouped, FP16-outer, and custom-conversion candidates did not win |
| P/dS reconstruction | Locally reducible only at small margin | QK, exponentiation, P, and dS are required unless forward artifacts or the numeric contract change |
| Quantization, C-to-A handoff, and dS publication | Head-specific | D64 producer-local P/dS stores are removed; D128 retains canonical cross-half P/dS plus the mirrored dS required for dQ |
| dQ workspace and atomic accumulation | Swizzled physical workspace retained | Direct MMA-TV global atomics remove shared dQ staging while preserving FP32 grouping and the global contribution count |
| Q/dO packet and next-pair traffic | Existing schedules retained; early-dO and tail-prefetch variants regressed | The phased D128 Q/dO packet alias passed resource/sanitizer checks but regressed CUDA-event latency; the accepted narrow producer-layout repair removes recurrent `LDGSTS` excess without changing the publication barrier or tail safety |
| Forward-owned Q/K or P metadata | Contract-changing | Both forward and backward head dimensions use Q32/K64, but the public forward still discards Q/K artifacts; routed saved state and full P checkpointing remain separate contracts |

The exact phased packet cache, score-side K caches, and full dQ dS-mirror hoist remain closed. The fresh attribution does, however, narrow two materially different D128 screens that were not covered by those rejections: a bank-clean transposed-B copy path that keeps the current Q/dO layout, and a half-pair P/dS handoff that publishes only the remote half needed by each dV/dK owner. These do not reopen the rejected full Q/dO packet cache or the unstable all-slice dQ hoist.

### SageBwd Paper Comparison

#### Evidence boundary

The SageBwd papers disclose the numerical algorithm and benchmark figures, but not the SageBwd kernel source or benchmark harness. The primary references are <https://arxiv.org/html/2603.02170v1> and <https://arxiv.org/html/2505.11594v3>. The public `thu-ml/SageAttention` `main` branch at commit `d1a57a5` contains forward/inference code and no SageBwd kernel or backward API; the upstream release issue remains open at <https://github.com/thu-ml/SageAttention/issues/229>. Therefore, the matrix split and quantization choices below are paper facts. Exact Triton tile sizes, warp assignment, register use, pipeline stages, workspace ownership, baseline version, and preprocessing timing are not established paper facts.

#### Benchmark scope

The SageBwd paper's Figures 2 and 3 are labeled forward plus backward kernel throughput. They are not backward-only measurements. The `1.67x` headline is the D64 causal 8K combined point: the plotted SageBwd and FlashAttn(CUDA) bars are approximately `248` and `149` TOPS, or `248 / 149 = 1.664x`. The earlier SageAttention3 source includes separate backward figures: at D128 non-causal 4K/8K, SageBwd is approximately `174/139 = 1.25x` and `182/145 = 1.26x` against FlashAttn(CUDA); at D64 it is approximately `229/142 = 1.61x` and `246/151 = 1.63x`. Against the faster FlashAttn(Triton) bars, the D64 4K/8K ratios are approximately `1.46x` and `1.49x`.

The paper source does not publish batch/head shapes, the FlashAttention commit or package version, the exact combined-throughput formula, or whether Q/K/V and dO quantization, smoothing, dQ conversion, and workspace initialization are inside each plotted region. The maintained SageAttention benchmark README says its published TOPS exclude quantization and smoothing, but that is repository evidence rather than a released SageBwd timing harness. Backward comparisons must therefore use the separate paper backward figures when available, and must keep kernel-only, native-plus-conversion, and end-to-end scopes separate. The local `2.5x` backward-FLOP convention is a useful accounting convention, not evidence of the paper's hidden harness.

#### Disclosed SageBwd path

The INT8 SageBwd design quantizes six of the seven attention matrix multiplications. Forward uses INT8 QK and INT8 PV, with per-block Q/K/V artifacts and per-token P quantization. It reuses the online-softmax maxima for the P scale instead of running a separate P maximum reduction. Backward recomputes tiled S and P, keeps `dP = dO @ V^T` in FP16, and quantizes the other four backward matrix multiplications: dV, dQ, and dK use INT8 operands, while INT8 dS feeds the dQ/dK products. The paper specifies per-block P and dO quantization in backward, per-block dS quantization, K-smoothing, and the smooth-K dQ correction `rowsum(dS) * K_mean`. It uses a FlashAttention-style tiled, on-the-fly dataflow and implements SageBwd in Triton.

The maintained native path already contains the main mathematical speed mechanism: INT8 QK, INT8 dV/dK/dQ, FP16 dP, on-the-fly P/dS reconstruction, one-time dO preprocessing, fused dV/dK/dQ ownership, FP32 dQ accumulation, and smooth-K correction. It is not otherwise identical to the paper algorithm. The retained forward is QK-INT8 plus FP16 PV, whereas the paper's forward PV is INT8; the native backward keeps the fixed Q32/K64 artifact contract and uses a predicted dS scale rather than the paper pseudocode's exact per-block `max(abs(dS))`. Those are contract or accuracy choices, not missing low-level instructions in the current backward kernel.

The paper's statement that its implementation prioritizes correctness and stability over aggressive fusion is also important. The FP4-forward section of SageAttention3 describes layout permutation, online-softmax max reuse, and producer-warp ping-pong, but those are FP4 inference optimizations and must not be presented as disclosed SageBwd backward optimizations.

#### Remaining optimizations under the fixed contract

The highest-value remaining work is representation and scheduling identified by the fresh Sage/Flash profile, not a new matrix-multiply algorithm. The paper discloses the low-precision products, block quantization, FP16 dP, smoothing, and dQ correction, but it does not disclose a backward tile, warp schedule, stage count, or handoff representation. The local plan therefore uses the paper only as a numerical and performance target.

- First remove the measured D128 transposed-Q/dO shared penalty without changing the ordinary consumer layout. The preferred screen is a non-transposed LDSM plus exact register permutation into the INT8 MMA-B fragment. The fallback is an aliased bank-clean mirror in the consumed FP16 dO stage, populated after dP and synchronized by the existing P/dS barrier. Neither path may increase `78,112` bytes, add a barrier, or disturb the two-stage tail lifetime.
- Next test the D128 half-pair P/dS handoff. Each dV owner already has its local P C fragment and needs only the remote M16 P half; each dK owner has its local dS C fragment and needs only the remote dS half. Keeping the local half in registers and publishing one aligned remote packet per warp can remove half the canonical P/dS stores and replace a full pair load with a remote-half load. The dS mirror for dQ remains unchanged. This is smaller and more directly substitutive than the rejected phased Q/dO cache.
- Then revisit dS-load amortization only as a bounded ownership experiment. Across the four dQ owners, the current loop executes `32` stable dS `LDSM.x4` copies per temporal M32 pair: two fragments inside each of four owned D16 slices. Grouping two D16 accumulators reduces that to `16` copies but adds about eight live accumulator registers, which the `252/254` aligned raw/smooth images cannot absorb naively. The rejected full hoist reduced the same count to `8` but was timing-unstable. A role-specific or schedule-level lifetime reduction must be proven before implementation.
- Broader tile and warp changes are justified only if they can cross a residency or issue-concurrency threshold. M64xN32 and M32xN64 remain one-CTA shapes and add replication or loop boundaries. M32xN32 is the only straightforward two-CTA storage candidate, but it must also compile below the register limits above and account for doubled Q/dO traversal. M64xN128 exceeds the shared-memory budget without phasing and doubles dK/dV ownership pressure.

The current evidence does not prioritize reducing DRAM traffic or dQ reduction count: Sage moves slightly less DRAM data than Flash and both execute the same global dQ reduction count. The remaining candidates target on-chip instructions, shared-memory issue, and scalar/MIO work. A larger Q block, different scale granularity, a separate dQ reduction, or a new saved-artifact format is outside the retained Q32/K64 and workspace contracts and should be measured as a separate branch.

#### SM89 ratio assessment

Ada does not provide a special dense INT8 advantage over Ampere for this comparison. NVIDIA's RTX 4090 table gives dense peak `660.6 TOPS` INT8 and `165.2 TFLOPS` FP16 Tensor Core with FP32 accumulation; the RTX 3090 Ti gives `320 TOPS` and `80 TFLOPS`. Both normalize to approximately `4:1`, or about `2048` INT8 operations/cycle/SM versus `512` FP16 operations/cycle/SM. Ada's relevant architectural change is primarily FP8 support here, not a larger INT8-to-FP16 ratio. The official architecture references are <https://docs.nvidia.com/cuda/ada-tuning-guide/index.html> and <https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html>.

The RTX 4090's 72 MiB L2 versus GA102's 6 MiB L2 remains a plausible workload-dependent ratio effect. The relevant main-kernel read-set estimate is smaller for Sage, but it must include the FP16 dO stream used by dP:

| Head/sequence | Sage read set per head | Flash read set per head |
|---|---:|---:|
| D64, 4K / 8K / 16K / 32K | 1.75 / 3.50 / 7.00 / 14.00 MiB | 2.00 / 4.00 / 8.00 / 16.00 MiB |
| D128, 4K / 8K / 16K / 32K | 3.50 / 7.00 / 14.00 / 28.00 MiB | 4.00 / 8.00 / 16.00 / 32.00 MiB |

These estimates count Sage `Q8 + K8 + dO8 + V16 + dO16` and Flash `Q16 + K16 + V16 + dO16`; scale, LSE, Delta, and output/workspace traffic are not included. The saved forward artifact subset is smaller (`Q8 + K8 + V16`), but it is not the complete backward read set. For a B=1 grid, multiplying the table by 16 or 32 heads gives:

| Shape | Sage H16 / H32 | Flash H16 / H32 |
|---|---:|---:|
| D64, 4K | 28 / 56 MiB | 32 / 64 MiB |
| D64, 8K | 56 / 112 MiB | 64 / 128 MiB |
| D64, 16K | 112 / 224 MiB | 128 / 256 MiB |
| D64, 32K | 224 / 448 MiB | 256 / 512 MiB |
| D128, 4K | 56 / 112 MiB | 64 / 128 MiB |
| D128, 8K | 112 / 224 MiB | 128 / 256 MiB |
| D128, 16K | 224 / 448 MiB | 256 / 512 MiB |
| D128, 32K | 448 / 896 MiB | 512 / 1024 MiB |

This is a capacity comparison, not a claim that the whole grid is resident: CTA interleaving, launch order, output stores, and other allocations determine actual reuse. It does show where a 72 MiB L2 can materially exceed the 6 MiB GA102 cache: D64 H16 at 8K is near the boundary (Sage 56 MiB, Flash 64 MiB), while D64 H32 and all larger grids exceed it; D128 H16 at 4K is also near the boundary (Sage 56 MiB, Flash 64 MiB). The Sage grid is N-block-owned and repeatedly reads Q/dO, while normal Flash backward is Q-block-owned and repeatedly reads K/V, so a larger cache can change which repeated operand stream is retained. However, the local matched profile already shows Sage with slightly less DRAM traffic and a much larger on-chip instruction/MIO cost, and the exact sm86/sm89 probes produce identical machine instruction words and resources for both Sage and Flash. L2 may change absolute time and can change the ratio at a particular shape, but it is not evidence for a missing Ada-specific Sage optimization and cannot explain the measured SM86 D128 deficit by itself.

## Completed Work

This section records the decisions behind the current source and prevents closed experiments from being repeated without new evidence. An implementation defect invalidates only that candidate build and its measurements, not the underlying design. An experiment is closed only after the defect is fixed and the corrected candidate is validated, or when independent design-level accuracy or contract evidence rules out that specific formulation.

### Completed Audits

- The pre-swizzle resource audit reported 243 registers per thread for tail and aligned variants, zero stack/local spill, and 57,600 bytes of dynamic shared storage. The retained pair-interleaved, lane-major-packet raw-K N128 image uses 242/239 registers and 2,176/2,008 instructions for tail/aligned, with 49,408 bytes dynamic shared memory and zero stack/local spill; smooth K uses 242/237 registers and 2,320/2,072 instructions, also with zero stack/local spill. Both retain four static barriers. N256 retains the specialization-local 128-register cap so 512 threads fit the SM86 65,536-register CTA limit. The optimized raw-K N256 image uses 86,272 bytes shared, 136/112 bytes stack, and 2,520/2,344 instructions for tail/aligned; smooth K uses the same shared allocation, 160/120 bytes stack, and 2,696/2,400 instructions. The retained D128 image after explicit FMA, P/dS hoisting, and asynchronous dO staging uses 78,112 bytes shared, 241/252 registers and 3,248/3,224 instructions for raw tail/aligned, with zero stack/local spill; smooth uses 242/254 registers and 3,312/3,256 instructions. All three shapes retain four static barriers and permit one resident CTA.
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
- Forward/backward artifact compatibility. CUTLASS forward uses the same Q32/K64 Q/K scale domain at head dimensions 64 and 128, including K32 CTAs sharing one K64 scale and Q16 warps sharing one Q32 scale. Native backward consumes that contract at both head dimensions. The forward wrapper still discards those quantized tensors after producing output/LSE, so the backward-reuse benchmark remains a product-contract proxy rather than an integrated training invocation.
- Forward-owned P artifacts. Forward computes P transiently inside online softmax and publishes only output/LSE. A saved P maximum can remove only the local P maximum reduction; it cannot remove QK recomputation, exponentiation, P reconstruction, or dV's P operand. Saving one P-scale value per Q32/K64 block is still an `O(N^2)` metadata table whose forward stores and backward loads must be counted, while saving quantized P is substantially larger. Neither is feasible under the current memory and forward-state contract.
- dS scale-fusion review. No existing scale-cancellation implementation was found. An unscaled-dS representation could move `sm_scale` in real arithmetic, but the scale floor, reciprocal, rounding, saturation, and dequantization products change the quantized contract. This remains a bounded offline hypothesis, not a source-level optimization.
- Q/dO lifetime review. D64 issues the next-pair `load_q_dO_pair()` after current dK/dV has consumed the packed Q/dO operands, while dQ no longer reads the raw buffers. D128 needs different ownership because inactive tail loader warps can reach prefetch before the remaining active pair finishes dK/dV. Its retained two-stage Q/dO buffer permits that asynchronous prefetch without an extra barrier; the end barrier publishes the inactive stage. Racecheck and Synccheck cover aligned and tail launches for both layouts and K policies.
- Sixteen-warp feasibility and lifetime audit. The generated N256 specialization proves that the shape is representable and runnable: its 128-register cap fits 512 threads exactly within the SM86 CTA register limit. The first implementation kept four FP16 V MMA-B fragments live across score reconstruction and spilled 208/160 bytes per thread. Keeping V in dedicated N256 shared storage and loading one D16 fragment only for its dP use reduces the retained stack to 136/112 bytes without changing arithmetic, scales, barriers, or ownership. The phased pair-interleaved dS and dK/dV epilogue scratch share one 8 KiB region, leaving 86,272 bytes dynamic shared memory. N128 compiles through the original register-resident V path and remains instruction-for-instruction identical to its saved control. This completes the source-local N256 lifetime cycle: on-demand V reload is retained, while partial V caching, serialized and split-owner dQ halves, shared-P staging, and register-cap alternatives are closed.
- D128 profiling-driven local cycle. Explicit outer FMA, P/dS fragment hoisting, and the narrow asynchronous dO producer repair are retained. CTA-wide dS-scale publication, base-2 reconstruction, every score-side K cache width, full dQ dS-mirror hoisting, and the phased Q/dO packet cache are closed by resource or latency evidence. Fresh matched profiling narrows two new source-specific screens: changing the transposed-B copy mechanism while keeping the current Q/dO layout, and publishing only the remote P/dS half required by each dV/dK owner. Broader work still requires an execution tile, warp, or stage redesign.
- D128 Q/dO layout microprobe. For the actual `32 x 8` uint128 stage, the retained `Swizzle<3,0,3>` is conflict-free for ordinary MMA-A loads and two-wave for the transposed MMA-B loads. Seven alternate layouts either retain that transposed penalty or make the hotter ordinary path two- to eight-wave. This closes a simple global swizzle change and directs the next experiment to the copy atom or an aliased mirror.
- Generic arithmetic and helper cycle. The common lane-major packet screen retained only the D64 N128 specialization; D128 and N256 failed their resource gates. Base-2 reconstruction is closed across D64 because N128-only arithmetic violated N256 compatibility and the common formula worsened N256 spill. Score-side K caching, cross-warp scale publication, and paired dK/dV epilogue staging are also closed after valid spill-free forms lost their promotion timing. Detailed evidence remains under `Retained Changes` and `Closed Experiments`.
- D64 schedule cycle. The final N128 sixteen-warp split-dV/dK schedule preserved N16 producer ownership, the pair-interleaved dS mirror, swizzled dQ, and four barriers, but all raw/smooth tail/aligned images spilled materially under the mandatory 128-register limit. D64 N128 therefore remains M64xN128/eight-warp; no D64 execution-schedule candidate remains open under the retained contract.

### Retained Changes

- Unified Q32/K64 CUTLASS forward contract. The normal configured/public paths for head dimensions 64 and 128 quantize with flat Q32/K64 metadata and launch one fixed scale path while retaining all four CTA/warp execution instantiations. The public API, eager autotune, and compile autotune remain supported. Sequence-129 NHD/HND tests cover both head dimensions and all four execution tiles with and without LSE.
- Native backward dead-surface cleanup. The internal operator no longer accepts `output` or final fp16 `dQ`, because Triton preprocessing consumes output before launch and Triton postprocessing converts the fp32 dQ workspace afterward. Their unused launch strides and rejected, unreferenced dQ/scale/transpose helpers were removed. The generic head-dimension and tile traits now support both maintained kernel bodies.
- K128 main-kernel baseline and N256 experimental config. QBlock=32/KBlock=64 CTA M64xN128 with eight warps remains the public default. CTA M64xN256 with sixteen warps is generated and dispatched only when callers explicitly request `(32,64,64,256)`. N256 preserves one warp per N16 producer tile, expands dQ from four to eight N32 pairs and from two to four K64 domains, and halves cross-CTA dQ contributions. No backward autotuner selects between them.
- Dedicated D128 backward. QBlock=32/KBlock=64 CTA M64xN64 with eight warps is the maintained D128 reference. Each N16 tile has two M-half warps; dV/dK ownership is split between them, while two even-N pairs own four D16 dQ slices each. A 16 KiB Q/dO double buffer prevents asymmetric-tail prefetch races while preserving four barriers. The source-order quantized model matches native dQ to `2.4e-7` and dK/dV to fp16 rounding on the N65 bring-up case. It is correct and spill-free but slower than normal FlashAttention dispatch, so it is retained as support and an optimization baseline rather than a performance default.
- D128 explicit FP32 outer FMA. The two D128 dK/dV accumulator updates now use `fmaf(int32_partial, scale, accumulator)`. Tail ptxas had already contracted the expression, so tail instruction counts remain `3,232/3,304` for raw/smooth. In aligned raw/smooth images, `FFMA` rises from 11 to 139 while `FMUL` falls from 222 to 94 and `FADD` falls by 64; static instructions fall by 72 to `3,192/3,232`, and registers fall from 255 to `252/254`, with zero stack/local spill. The change is D128-only: all eight D64 N128/N256 encoded instruction sequences remain exactly identical to the saved control. The 52-case focused suite passes. Direct FMA/control captures at sequence 1024/1025 in both layouts and K modes have maximum fp16 differences at or below `1.22e-4` and relative norms below `8.2e-6`; tail dK/dV are bitwise identical. The final FMA-only binary passes all 16 D128 Racecheck/Synccheck aligned/tail, layout, and K-policy cases. The two-bracket CUDA-event comparison improves all eight standard raw rows by `1.0077-1.0172x`, `1.0144x` geometrically, so the change is retained.
- D128 P/dS MMA-A fragment hoist. The dV-side P fragment and dK-side dS fragment are now copied once per owning M half before their eight D16 dimension slices, instead of being reloaded inside every slice. This removes 14 static `LDSM` instructions in each D128 image and lowers the FMA-reference resources from raw tail/aligned `248/252` to `240/250` registers and smooth `245/254` to `237/250`, with zero stack/local spill, unchanged `78,112` bytes of shared memory, and four barriers. Static SASS changes from `3,232/3,192` to `3,216/3,192` for raw tail/aligned and from `3,304/3,232` to `3,288/3,224` for smooth; the raw aligned image is instruction-count neutral because ptxas fills the shorter schedule with NOPs. The focused 52-case suite and all 16 filtered Racecheck/Synccheck cases pass. Direct NHD/HND raw/smooth captures at sequence 1024/1025 keep dK/dV bitwise identical to FMA-only and dQ within `1.22e-4` and `8.3e-6` relative norm. Two 20-warmup/100-repeat ABBA brackets per layout, aggregated over four medians per row, improve all eight standard native-plus-conversion rows by `1.0234-1.0349x`, `1.0291x` geometrically; the hoist is retained.
- D128 asynchronous Q/dO producer staging. The FP16 dO asynchronous copy contract now uses 16 lanes per logical row on D128, so each static copy covers two complete 256-byte FP16 rows instead of four separated 8-vector half-rows. The shared tensor layout, consumer-facing ordinary `LDSM` stream, Q and INT8 dO copies, two-stage lifetime, four barriers, and tail ownership are unchanged. The accepted image retains `78,112` bytes of shared storage and zero stack/local spill; raw tail/aligned resources are `241/252` registers and `3,248/3,224` instructions, while smooth uses `242/254` registers and `3,312/3,256` instructions. The matched native-plus-conversion CUDA-event ABBA run improves all eight H16/H32, S4096/S8192 NHD/HND rows by `1.0008-1.0116x`, `1.0073x` geometrically.
- Generated launch units. Backward emits one source per launch specialization with the stable `hd*_bm*_bn*_nw*_bq*_bk*` naming scheme. The two D64 launches and one D128 launch therefore compile as three independent units, within the four-unit ceiling.
- Triton utility phases. Delta/dOutput preprocessing, Q32/K64 predictor summaries, workspace allocation, and fp32 dQ-to-fp16 conversion moved out of the native wrapper and into autotuned Triton kernels. The native operator now launches the fused CUTLASS main kernel with explicit, typed workspaces.
- Native smooth K. Backward accepts the forward-compatible raw-domain LSE, centers K in the same Q32/K64 artifact domain, specializes the native kernel on smoothing, accumulates exact reconstructed-dS row sums, and fuses the K-mean correction into swizzled dQ conversion. The predictor audit found no reason to change the retained `1.5x` policy. The raw N128 specialization retains the control's instruction counts, register allocation, stack use, and barrier count. The 52-case focused matrix covers raw/smooth D64 N128/N256 and D128 N64, NHD/HND, aligned/tail inputs, both dQ workspace dimensions, and public head-specific default selection. D128 raw/smooth Racecheck and Synccheck pass NHD/HND at 64/65; the retained D64 sanitizer matrix remains clean at 512/513.
- Unconditional separable dS prediction. The retained performance predictor is `1.5 * (softmax_scale / N) * (max_row ||dO||2 * max_row ||V||2 + max_row |Delta|)`. It is `O(N*D)`, uses Q32/K64 summaries, and does not allocate an `O(N^2)` scale table or select a runtime policy. It is not the strict-accuracy oracle.
- Quantization constants and conversion. The accepted reciprocal is `0x1.010122p-7`, the scale floor is `2^-126`, and C++ INT8 conversion is saturating. Triton dOutput quantization uses the shared unclamped `_round_to_int8` helper because its exact max-abs scale preserves the intended range.
- Guard selection. The fixed `1.5x` dS guard is retained as the performance reference. A `1.25x` guard was faster on the six retained captures but was less robust on the sequence-65 correctness control; `2.0x` increased quantization error. The expanded scalar sweep confirms that no global guard or simple P-max multiplier clears both sigma-900 records, so guard tuning is closed without a new predictor structure.
- Training-derived validation data. Two independent SDXL-derived sequence-4096 captures with ten heads and sigma indices `300/700/900` are retained with reusable capture, simulation, benchmark, and evaluator scripts under `data/`. The full replay is finite, but exact sequence 4096 passes only four of six aggregate and 42 of 60 per-head strict gates. Mean dQ/dK relative error is `5.49%/4.42%`, worst error is `11.88%/19.06%`, and sparse dS clipping does not by itself predict those failures; this remains exploratory rather than strict-promotion quality.
- Validation baseline. The active build passes the parameterized D64/D128 focused matrix. Source-distinct N256 comparisons cover NHD/HND at 256/257/512/513: dK/dV are bitwise identical to N128 and dQ differs by at most `1.22e-4` from the changed FP32 atomic grouping. The D128 source-order model validates the exact quantized algorithm and exposed the now-fixed asymmetric-tail Q/dO prefetch race. D128 Racecheck and Synccheck pass all 16 raw/smooth aligned/tail layout cases. Resource/SASS inspection confirms zero D128 stack/local spill and four barriers without changing the retained D64 images. The completed D64 validation also includes the 16/32-head final Flash benchmark, swizzled-workspace ABBA timing, paired current-build NCU captures, `pre-commit`, compilation, and whitespace checks.
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

#### D128 Phased Q/dO MMA-B Packet Cache

The highest-ranked D128 packet experiment reused the dead per-stage 8 KiB FP16 dO region after P/dS reconstruction. Each warp converted its complete Q and int8 dO transposed-B fragment once, published aligned lane-major 16-byte packets before the existing pre-dK/dV barrier, and reloaded those packets for each D16 dK/dV consumer. It preserved Q32/K64 artifacts, two-stage asymmetric-tail protection, pair-interleaved dS, swizzled dQ, and four barriers. A word-major fallback was also compiled. Before explicit FMA, lane-major raw aligned used an 8-byte stack frame and word-major raw/smooth aligned used 40/32 bytes, so neither met the resource gate. Explicit FMA created enough headroom for raw packet variants; smooth aligned still needed a compile-time fallback to the unpacketized FMA path.

The final gated candidate was resource-clean: raw tail/aligned used 249/251 registers, smooth tail used 216, smooth aligned fallback used 254, and every variant had zero stack/local spill. Relative to FMA-only, raw static SASS fell from `3,232/3,192` to `3,064/3,112` instructions for tail/aligned. Each raw specialization removed 14 `LDSM`, 113 `SHFL`, and 56 `PRMT`, while adding 16 `LDS.128` and two `STS.128`; this proves that conversion work was removed rather than moved into an equivalent producer shuffle network. The 52-case suite passed. At sequence 1024/1025 in NHD/HND and raw/smooth modes, dK/dV were bitwise identical to FMA-only and dQ differed only by cross-run FP32 atomic order, at most `1.22e-4`. All 16 filtered Racecheck/Synccheck cases passed with zero hazards/errors.

The latency gate rejected the representation. One NHD `F-P-P-F` bracket used 20 warmups and 100 repeats at H16/H32 and sequence 4096/8192. FMA-only versus packet medians were `9.8721/9.9909`, `37.2811/37.7777`, `18.9982/19.2607`, and `74.1307/75.5857 ms`; control/candidate ratios were `0.9881x`, `0.9869x`, `0.9864x`, and `0.9808x`, or `0.9855x` geometrically with zero wins. The shared packet traffic and its issue cost outweigh the removed transposed-conversion instructions. HND promotion timing was skipped after the decisive four-row NHD regression. All packet helpers and staging changes were removed; the independent explicit D128 FMA and P/dS hoist remain, but no packet code does. Reopen this exact phased cache only with a different bank-proven packet representation or hardware access pattern that explains and removes the measured latency loss.

#### D128 P/dS MMA-A Fragment Hoist

The retained D128 FMA image reloaded the immutable P fragment for each dV D16 slice and the immutable dS fragment for each dK D16 slice. The accepted source-local change moves each MMA-A partition and copy before its owning `dim_base` loop. It does not alter quantized inputs, score arithmetic, producer ownership, the pair-interleaved dS mirror, the swizzled dQ workspace, the Q/dO double-buffer lifetime, or the four-barrier schedule.

The compiler result is resource-clean in all four raw/smooth and tail/aligned images. Relative to FMA-only, registers change `248/252 -> 240/250` raw and `245/254 -> 237/250` smooth, with zero stack/local spill and unchanged `78,112`-byte shared storage. `LDSM` falls from 76 to 62 in every image. Static instructions change `3,232/3,192 -> 3,216/3,192` raw and `3,304/3,232 -> 3,288/3,224` smooth; the raw aligned count is unchanged only because ptxas inserts NOPs after removing the loads. The focused 52-case suite passes, and the complete filtered D128 sanitizer matrix reports zero Racecheck hazards and zero Synccheck errors for NHD/HND, sequence 64/65, and raw/smooth K. Direct binary captures at sequence 1024/1025 show bitwise-identical dK/dV and only cross-run atomic-order dQ differences, at most `1.22e-4` with relative norm at most `8.3e-6`.

The latency gate uses native launch plus dQ conversion, excluding Q/K preparation, with two 20-warmup/100-repeat CUDA-event ABBA brackets per layout. Aggregating four control and four candidate medians per standard H16/H32, S4096/S8192 row gives:

| Layout | H16/S4096 | H16/S8192 | H32/S4096 | H32/S8192 |
|---|---:|---:|---:|---:|
| NHD | `1.0234x` | `1.0278x` | `1.0332x` | `1.0307x` |
| HND | `1.0234x` | `1.0296x` | `1.0349x` | `1.0302x` |

All eight rows improve; the geometric mean is `1.0291x`. Long D128 S4096/S8192 Flash-reference metrics are identical between FMA-only and the hoist for both layouts and K policies, including the known fixed Q32/K64 dQ/dK relative-error floor; the hoist adds no observed accuracy drift. This experiment is retained, and the next open D128 target is scalar and scale plumbing.

#### D128 Asynchronous Q/dO Producer Staging Repair

The original D128 Q/dO loader used the generic `Rows=4, Cols=16, LineLanes=8` FP16 copy contract. Each of its two static FP16 dO `LDGSTS` instructions therefore touched four row halves separated by the 256-byte physical row stride, producing the dominant shared-store wavefront excess. The accepted source-local repair parameterizes `GmemTiledCopyContract` and instantiates only D128 FP16 dO with `LineLanes=16`; the resulting `Rows=2` thread shape covers two complete rows per instruction and leaves the logical shared layout unchanged. Q, INT8 dO, K, V, consumer tensor views, asymmetric-tail ownership, and synchronization are not changed.

The current native-SM86 image remains resource-clean: dynamic shared storage is `78,112` bytes, all four static barriers remain, stack/local memory is zero, and the retained ordinary consumer stream still has 62 `LDSM` instructions. Raw tail/aligned resources are `241/252` registers and `3,248/3,224` instructions; smooth is `242/254` registers and `3,312/3,256` instructions. The eight D64 N128/N256 machine-code sequences are instruction-for-instruction identical to the retained control. A matched S8192 raw SourceCounters capture reduces D128 `LDGSTS` excessive wavefronts from the pre-repair `8,519,680` baseline to `262,144`; the residual four `65,536`-wavefront sites are one-time startup K/V copies, while the recurrent Q/dO sites are at their ideal count.

The 180-case focused CUTLASS files pass. Direct aligned/tail NHD/HND captures at sequence 1024/1025 and raw/smooth K keep dK/dV bitwise identical to the P/dS-hoist control; dQ differs only through cross-run FP32 atomic order, at most `6.10e-5` and below `7.9e-6` relative norm. All 16 filtered D128 Racecheck/Synccheck cases at sequence 64/65 report zero hazards/errors. Long S4096/S8192 D128 metrics preserve the known fixed-Q32/K64 accuracy floor. The complete collected test suite also passes all `2,659` node IDs in partitions: `1,008` CUDA-forward, `1,440` Triton-forward, and `211` compile/CUTLASS/backward/support cases.

Two inverse 20-warmup/100-repeat CUDA-event ABBA brackets per layout compare the candidate with the committed P/dS-hoist binary, excluding Q/K preparation and timing native launch plus dQ conversion:

| Layout | H16/S4096 | H16/S8192 | H32/S4096 | H32/S8192 |
|---|---:|---:|---:|---:|
| NHD | `1.0116x` | `1.0092x` | `1.0096x` | `1.0030x` |
| HND | `1.0008x` | `1.0075x` | `1.0099x` | `1.0064x` |

All eight rows improve, with geometric mean `1.0073x`; the producer-layout repair is retained.

#### D128 CTA-wide dS Predictor Publication

The D128 CTA-wide predictor candidate computed the same Q32/K64 dS predictor value in warp 0/lane 0 and published it through the existing P-scale `__syncthreads()` boundary. The logical quantization contract, arithmetic after publication, ownership, and barrier count were unchanged. Adding one FP32 field to `SharedStorageHD128` increased the aligned allocation from `78,112` to `78,128` bytes. The compiler produced raw tail/aligned resources of `255/242` registers and smooth `243/246`, with zero stack/local spill; static SASS was `3,232/3,256` raw and `3,320/3,296` smooth. The candidate passed the focused 180-case suite, all 16 Racecheck/Synccheck cases, direct aligned/tail output comparisons, and the long D128 accuracy matrix.

The promotion timing rejected the exchange. The full two-bracket NHD CUDA-event ABBA result was `0.9913x` geometrically, with only one of four H16/H32, S4096/S8192 rows improving (`0.9873x`, `0.9752x`, `0.9924x`, and `1.0108x`). The short screen and an inverse HND screen were not sufficient to overcome the decisive NHD promotion result. The extra shared scalar, scalar publication path, and altered live ranges cost more than the removed redundant predictor work. The candidate is closed; retain per-warp predictor computation and the `78,112`-byte staging layout.

#### D128 Base-2 Probability Reconstruction

The D128 base-2 candidate precomputed the score scale and row LSE in the log2 domain and replaced natural-domain `expf` reconstruction with `exp2f(fmaf(...))`. It preserved Q32/K64 artifacts, quantization scales, ownership, shared storage, and four barriers. Raw static SASS fell from the asynchronous-staging control by `8/16` tail/aligned instructions and smooth aligned fell by `16`; `FMUL` fell by `6/12`, registers remained `248/252` raw and `242/254` smooth, and stack/local traffic stayed zero. All eight D64 N128/N256 instruction streams were unchanged.

The changed FP32 order remained within the bounded numerical screen: direct captures reached at most `7.48e-4` fp16 absolute difference and `1.18e-4` relative norm, focused tests passed, all 16 sanitizers were clean, and the long D128 Flash-reference metrics were unchanged to the reported precision. The full two-bracket NHD CUDA-event ABBA result nevertheless lost every row, with geometric speedup `0.9869x` (`0.9832x`, `0.9877x`, `0.9852x`, `0.9915x`). The instruction reduction does not translate to latency on SM86, so this representation is closed without applying it to D64 or retaining its source changes.

#### D128 Score-Side K Fragment Hoisting

The score-side K experiment moved immutable K32 MMA-B fragments out of the temporal Q-pair loop. A full four-fragment cache adds 16 packed words per lane and failed the resource gate: raw tail/aligned compiled at `255/254` registers with 8 bytes of aligned stack, while smooth used `254/255` with 8 aligned stack bytes. Three cached fragments still used `254/254` raw with 8 aligned stack bytes and `251/254` smooth with 16 aligned stack bytes. Two fragments likewise spilled `8/16` aligned bytes. These forms were rejected before correctness or timing.

A final one-fragment cache was spill-free at raw `248/254` and smooth `245/254`, preserving `78,112` bytes of shared storage and four barriers. It caches one of four score K32 slices, so it removes only 25% of recurring score-side K loads after the first temporal tile. Aligned static instruction counts remain `3,224/3,256` raw/smooth, while tail expands from `3,248/3,312` to `3,280/3,320`. Focused tests passed, dK/dV remained bitwise identical, dQ stayed within `1.22e-4` and `8.8e-6` relative norm, all 16 sanitizers were clean, and long D128 accuracy was unchanged.

Timing did not meet the promotion criterion. The first two-bracket result was NHD `1.0032x` with three of four wins and HND `1.0058x` with three of four wins, `1.0045x` geometrically across the eight rows. An independent NHD repeat fell to `1.0008x` with two of four wins; aggregating both NHD runs gives about `1.0020x` and still regresses H16/S4096 to about `0.9970x`. The sub-percent result is not repeatable across shapes and does not justify higher registers or tail SASS. All score-side K cache forms are closed and their source changes are reverted.

#### D128 dQ dS-Mirror Fragment Hoisting

The dQ mirror candidate loaded both stable dS MMA-A fragments once before the four owned D16 output slices, replacing eight shared fragment loads per temporal step with two. It preserved the pair-interleaved dS producer, dQ workspace layout, K packet loads, source-order INT32 accumulation, Q/dO stages, shared storage, and barriers. The compile was clean: LDSM fell from 62 to 56 in all four images; raw tail/aligned used `241/254` registers and `3,248/3,208` instructions, while smooth used `242/246` and `3,312/3,256`. There was no stack/local traffic, shared storage remained `78,112` bytes, and all eight D64 instruction streams were unchanged.

The candidate passed the focused 180-case suite and all 16 Racecheck/Synccheck cases. Direct NHD/HND, S1024/1025, raw/smooth captures kept dK/dV bitwise identical and bounded dQ differences to `6.10e-5` and `7.4e-6` relative norm. Long D128 accuracy was unchanged to the reported precision. The short screen improved all eight rows at NHD `1.0105x` and HND `1.0089x` geometrically.

Longer timing was not stable enough to retain the change. The first full two-bracket runs were NHD `1.0010x` and HND `1.0041x`, each with three of four wins. A second run improved NHD to `1.0081x` and HND to `1.0028x`, both with four wins; aggregating the first two runs gave NHD `1.0046x` with four wins and HND `1.0035x` with three wins, the remaining H32/S4096 row effectively tied at `0.9998x`. A final HND bracket regressed to `0.9916x`; the three-run HND aggregate is `0.9995x`, with H16/S4096 at `0.9910x`. CUDA-event latency therefore does not meet the cross-shape stability gate despite the lower LDSM count. The candidate is closed and its source changes are reverted.

#### D64 N128 Lane-Major MMA-B Packet Handoff

The retained D64 N128 packet change re-lays out only the already-required packed K, Q, and dO MMA-B handoffs from `(tile, word, lane)` to aligned `(tile, lane, 16-byte packet)` storage. Producers and consumers use one `STS.128`/`LDS.128`-equivalent access instead of four scalar word accesses. Packet bytes, fragment order, ownership, shared allocation, and the four-barrier lifetime are unchanged. An initial all-specialization build was rejected because D128 raw aligned gained a 16-byte stack frame and N256 stack worsened. The final compile enables lane-major access only for D64 CtaN=128; all four D128 and all four N256 instruction streams remain exactly identical to the accepted asynchronous-staging control.

The D64 N128 image remains spill-free. Raw tail/aligned resources change from `235/234` registers and `2,192/2,024` instructions to `242/239` and `2,176/2,008`; smooth changes from `237/234` and `2,344/2,088` to `242/237` and `2,320/2,072`. Shared storage remains `49,408` bytes. Across each image, scalar/shared opcode counts fall by 36 LDS, 18 STS, and 16 PRMT sites without changing LDSM, IMMA, atomics, or barriers.

Matched S8192 SourceCounters profiles prove the bank contract. Control and candidate both execute `316,080,128` total shared wavefronts, `303,431,680` ideal wavefronts, and `12,648,448` excessive wavefronts; every new `LDS.128` and `STS.128` site has observed wavefronts equal to ideal. The candidate passed the focused 180-case suite and all 16 D64 Racecheck/Synccheck cases. Direct S1024/1025 NHD/HND raw/smooth comparisons keep dK/dV bitwise identical and bound dQ to `6.10e-5` and `5.6e-6` relative norm.

Two 20-warmup/100-repeat CUDA-event ABBA brackets per layout improve every standard D64 row. NHD H16/S4096, H16/S8192, H32/S4096, and H32/S8192 improve by `1.0513x`, `1.0361x`, `1.0277x`, and `1.0358x`; HND improves by `1.0420x`, `1.0304x`, `1.0278x`, and `1.0367x`. The combined geometric mean is `1.0360x` with eight of eight wins. The complete 2,659-test collection passes through bounded partitions; one unrelated BF16 causal forward tolerance case passed on exact rerun. The retained extension SHA256 is `7E47579C948951E9FC0836A1687EF5E482E8CC5D1DE14604C2174459B9E414AB`. The D64 N128-only lane-major packet layout is retained.

#### D64 Base-2 Probability Reconstruction

The D64 N128-only base-2 candidate precomputed score scales and row LSE in log2 space and used `exp2f(fmaf(...))`. It passed the initial compiler gate: raw tail/aligned SASS fell from `2,176/2,008` to `2,136/1,984`, aligned FMUL fell by 26, registers improved from `242/239` to `241/238`, and there was no spill. Smooth used `242/238` registers and `2,312/2,048` instructions. D128 and N256 machine code remained exact under the N128-only guard.

The arithmetic guard then failed all 16 focused N256-vs-N128 compatibility cases because the two schedules reconstructed and quantized P with different FP32 rounding; maximum gradient deltas reached about `1.9e-3`, beyond the existing cross-config contract. Applying the same base-2 formula to N256 restored a common arithmetic representation but failed its resource gate: raw tail/aligned stack increased from `136/112` to `176/128` bytes and smooth tail increased from `160` to `184`, with more local instructions. The D64 base-2 candidate is closed without timing, and the natural-domain formula is restored in both schedules.

#### D64 Score-Side K Fragment Hoisting

The D64 N128 score-K candidate cached both immutable K32 MMA-B fragments across the temporal query loop while leaving N256 and D128 machine code unchanged. It fit without spill but consumed the available register headroom: raw tail/aligned moved from `242/239` to `243/243`, smooth from `242/237` to `243/241`. Aligned static instruction counts were unchanged at `2,008/2,072` raw/smooth; tail grew by 8 raw and 16 smooth instructions. The focused 180-case suite passed.

A 10-warmup/50-repeat NHD CUDA-event bracket rejected the cache before broader promotion. H16/S4096, H16/S8192, H32/S4096, and H32/S8192 measured `0.9727x`, `0.9716x`, `0.9832x`, and `0.9844x`, or `0.9779x` geometrically with zero wins. Extending both fragment lifetimes at the 243-register ceiling costs more than the removed shared loads. HND timing and sanitizers were skipped after the decisive NHD result; the cache is closed and reverted.

#### D64 Domain-Scale Publication

A D64 N128-only candidate used the existing post-P/dS barrier to publish `dS_scale * q_block_scale` and `dS_scale * k_block_scale` once per K64 domain. This replaced per-warp dK multiplication and the four dQ owners' repeated domain-predictor calculations with shared loads. A CtaN=128-only storage base kept D128 and all N256 functions instruction-for-instruction identical; N128 shared storage increased 16 bytes to `49,424` with four barriers.

The compiler response was substantial: raw tail/aligned registers fell from `242/239` to `230/217`, smooth from `242/237` to `239/217`, and aligned static instructions fell from `2,008/2,072` to `1,984/2,056`. Every variant remained spill-free. The focused 180-case suite passed; direct S1024/S1025 NHD/HND raw/smooth comparisons retained bitwise dK/dV, with only cross-run dQ atomic-order deltas up to `6.10e-5` and relative norm below `6.9e-6`.

CUDA-event latency nevertheless rejected publication. A 10-warmup/50-repeat NHD bracket measured `0.9660x`, `0.9783x`, `0.9766x`, and `0.9775x` for H16/S4096, H16/S8192, H32/S4096, and H32/S8192, or `0.9746x` geometrically with zero wins. As with CTA-wide D128 dS publication, the shared producer/consumer path costs more than the redundant scalar work on SM86. HND timing and sanitizers were skipped after the decisive NHD loss; the source and retained binary are restored.

#### D64 Paired dK/dV Epilogue Staging

The D64 N128 candidate used the two disjoint 512-byte halves already available in each 1,024-byte warp scratch slot to stage dV and dK together. It converted and stored both FP16 fragments, issued one warp synchronization, copied both tiles globally, and issued one final warp synchronization. D128 already assigns dV and dK to separate M-half warps, and N256 has only one 512-byte epilogue stage per owner, so both retained their exact machine code.

The candidate added no shared storage or CTA barrier and remained spill-free. Raw tail/aligned resources stayed `242/239` registers and `2,176/2,008` static instructions; smooth stayed `242` tail and improved aligned from `237` to `235` registers, with `2,320/2,080` instructions versus `2,320/2,072` in control. The focused 180-case suite passed.

The short NHD CUDA-event gate rejected the pairing before broader promotion. A 10-warmup/50-repeat bracket measured `0.9757x`, `0.9866x`, `0.9902x`, and `0.9845x` for H16/S4096, H16/S8192, H32/S4096, and H32/S8192, or `0.9842x` geometrically with zero wins. The small epilogue cannot repay the altered store schedule and simultaneous half-fragment lifetime. HND timing and sanitizers were skipped; the candidate is closed and reverted.

#### D64 N256 Split dQ-Half Ownership

The N256-only compiler screen expanded dQ ownership from four warps to eight. Each owner handled one temporal M16 half for one D16 output slice, retaining the same total dQ atomic contributions while halving its dQ accumulator/sum fragments. The tradeoff was duplicated packed-K and dS-mirror consumption by the two M-half owners. D64 N128 and D128 machine code remained instruction-for-instruction identical.

The resource result was mixed and failed the predefined gate. Raw aligned stack improved from `112` to `96` bytes, and smooth tail/aligned improved from `160/120` to `152/96`; raw tail stack instead worsened from `136` to `184` bytes because the duplicated ownership expanded predicated tail state. Every variant still used the required 128-register cap, but a candidate that worsens either raw specialization does not solve the N256 spill problem. Correctness, sanitizers, and timing were skipped after the compiler rejection; the ownership split is closed and reverted.

#### D64 N128 Sixteen-Warp Split dV/dK Ownership

The final D64 schedule candidate launched 512 threads for the same M64xN128 CTA. Warps 0-7 retained one producer per N16 tile, reconstructed score/P/dS, owned dV and the existing dQ slices, and published the retained Q/dO packets plus pair-interleaved dS mirror. Warps 8-15 owned dK for the corresponding N16 tiles. Each helper reloaded its two temporal dS halves from the mirror and applied the existing C-to-A register transpose before consuming the matching Q packet. A single FP32 dK-or-dV accumulator image was reused by role. K/V staging remained cooperative, the existing pre-dK/dV barrier ordered producer publication before helper reads, the end barrier protected next-pair publication, and concurrent final dV/dK stores used the two disjoint 512-byte halves of each existing 1 KiB scratch slot. The candidate therefore kept `49,408` bytes of shared storage and four static CTA barriers without all-slot score reloads or a fifth barrier.

The mandatory 128-register gate rejected the schedule. Raw tail/aligned functions used 128 registers with `112/80` bytes of stack; smooth tail/aligned used 128 registers with `128/80` bytes. Static SASS grew from the eight-warp control's raw `2,176/2,008` and smooth `2,320/2,072` tail/aligned instructions to raw `2,864/2,576` and smooth `3,016/2,648`. The four candidate images contain `19/21/23/21` `LDL` and `35/19/39/19` `STL` sites for raw-tail/raw-aligned/smooth-tail/smooth-aligned. Splitting the persistent dK/dV accumulator is not enough to fit the producer's score, dP, resident-V, dV, and dQ lifetimes into 128 registers; helper mirror reconstruction and role control also expand the instruction stream.

Correctness, sanitizer, and timing work were skipped exactly as required by the compiler-first gate. The candidate source was reverted, the three stable generated units were restored, and the accepted extension was restored to SHA256 `7E47579C948951E9FC0836A1687EF5E482E8CC5D1DE14604C2174459B9E414AB`. The candidate binary, source patch, resources, and SASS are retained under `C:/tmp/sageattn/experiments/d64_split_dkv_n128_nw16`. This closes D64 schedule tuning under the current SM86, Q32/K64, public-layout, and four-barrier contract.

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
- The tested broad Q/dO packetization, extra packet caches, register-V, and split dV/dK formulations are closed because their validated builds were slower or resource-heavy. The narrow D64 N128 re-layout of existing exact-size packets is separately retained; revisit broader packet ideas only with a materially different ownership hypothesis and source/SASS evidence that targets a measured handoff cost.
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
- Keep D64 N128 at eight warps under the retained contract. The sixteen-warp N16-preserving split-dV/dK schedule still needs 80-128 bytes of stack per thread at the mandatory 128-register cap, despite halving the persistent dK/dV accumulator image.
- Treat broad packetization, the validated D128 phased Q/dO packet cache, register-V, direct/transposed dQ, early-dO, tail-prefetch, split dV/dK, periodic/power-of-two dS, pre-dK barrier removal, unchanged exact-max K64 dS synchronization, coordinated dV/dK grouping, FP16 dK outer accumulation, and the fixed-shift SM86 conversion primitive as closed controls unless a materially different representation changes the measured cost model.
- Keep the current D128 Q/dO `Swizzle<3,0,3>` for the ordinary stage. The screened plain and alternate swizzles cannot remove the transposed-B penalty without worsening ordinary loads. A non-transposed copy atom or consumed-stage mirror is a different experiment and remains open.
- Treat dV/dK INT32 conversion and scaled-FP32 accumulation as compulsory under the retained independent scale domains. Reopen only with a representation that proves a common product, bounded integer rescaling, accuracy, resource fit, and removal of hardware work rather than movement between FP16 and FP32.
- Keep direct dQ-mirror elimination closed until a producer-to-consumer proof beats the known three-MOVM/two-lane/two-byte reconstruction cost. The producer-local dV/dK handoff does not make the cross-warp dQ mapping cheaper. The corrected transposed operand-role formulation is also closed: its exact CuTe C-to-B `Copy_Atom` needs a material cross-lane shuffle/permutation network and regressed every measured row by `4.8-9.3%`.
- Keep the fixed predictor as the performance reference and exact-max reconstruction as the accuracy oracle. Do not retry a scalar guard, simple P-max multiplier, generic QK random projection, or the tested 2x4-plus-row-sample formulations.
- Use the retained D64 producer-register handoff and swizzled global dQAccum workspace as the native local baseline. Do not restore D64 canonical P/dS score-pair stores or treat its removed slots as free storage. D128 still owns canonical score-pair storage until a validated half-pair replacement removes it.
- Keep integrated forward/backward state routing outside this plan. The public forward still discards Q/K artifacts, and the current scope explicitly excludes a state-owning forward result, internal compatibility path, combined training API, and integrated training benchmark.
- Keep a split dQ workspace and separate reduction outside the retained ownership contract. It adds split-proportional FP32 workspace traffic, another launch, and a different reduction order; evaluate it only on a separate branch with complete end-to-end costs. The Triton reduction tree is `O(N*D)` and does not target the measured native shared-store bottleneck.
- Keep Q32/K64 fixed for this optimization cycle. Alternate quantization blocks require a separately generated and accuracy-qualified artifact contract rather than another candidate in the retained schedule.
- Treat the four barriers and the `0.43` eligible-warps/cycle result as diagnostics, not permission to remove synchronization. Any new schedule must identify a different ownership dependency and pass the full native sanitizer matrix.
- Keep N256's 128-register cap. The current 512-thread CTA cannot request 129 registers, and uncapped allocation remains above the hardware limit. Reopen N256 lifetime work only with a materially different accumulator or phase-ownership design that compiles at no more than 128 registers and beats both the retained N256 image and unchanged N128.

## Next Work

The measured references are Q32/K64 D64 M64xN128/eight-warp with `smooth_k=False` and D128 M64xN64/eight-warp. The fresh common-baseline position is `1.1934x/1.2249x` D64 and `0.9563x/0.9810x` D128 for end-to-end/native-plus-conversion geometric speedup against normal FlashAttention dispatch.

### Invariant Working Rules

- Treat `Current State` as the active contract and `Completed Work` as the exclusion registry. Do not reimplement a retained or closed candidate during a new experiment unless new source attribution supports a materially different representation.
- Declare before coding whether a candidate changes artifacts, accuracy, ownership, scale, precision, synchronization, generated units, workspace, or the public product contract. Preserve every category not explicitly under test.
- Isolate head- and tile-specific experiments until they pass independently. A D128-only experiment must not silently alter D64 or N256, and a D64 N128 experiment must not silently change N256 compatibility.
- Run the compiler/resource gate before correctness, sanitizers, or timing. Inspect raw/smooth and tail/aligned functions for registers, stack, local loads/stores, shared bytes, barriers, and static instructions. They may help identify bottlenecks, but the optimization target is still the overall speed.
- Require a source-distinct mapping proof before changing a fragment layout, copy atom, packet format, or producer/consumer handoff.
- Require aligned/tail NHD/HND correctness and long-shape accuracy for every surviving candidate. Arithmetic-order or precision changes need explicit numerical-delta bounds.
- Require Racecheck and Synccheck whenever ownership, publication, stage lifetime, or synchronization changes.
- Use interleaved CUDA-event ABBA for latency promotion. Report native-plus-conversion and end-to-end separately; NCU duration is attribution evidence only.
- Prove with SASS or counters that a source-specific optimization removed its intended cost instead of moving it into spills, extra shared traffic, atomics, or another launch.
- Restore the accepted generated units and production binary after rejected or `-lineinfo` experiments. Repair archived scripts only when they are selected for reuse.

### Changes To Try

The percentage ceilings below are planning estimates derived from measured instruction and wavefront families, not performance claims. Work in the listed order; do not start a broad D128 schedule rewrite before the two source-specific handoff experiments are resolved.

#### D128 Changes

| Order | Change | Why it is open | Estimated ceiling | First implementation gate |
|---:|---|---|---|---|
| 1 | Non-transposed LDSM plus exact register permutation for Q/dO MMA-B | Directly targets all `134,217,728` measured excessive LDSM wavefronts while retaining the conflict-free ordinary layout | About `1-4%` kernel latency | Exact byte mapping, ordinary-load wavefronts unchanged, `78,112` B, four barriers, zero stack/local in all four images |
| 1b | Aliased consumed-stage Q/dO transposed mirror | Fallback only if order 1 loses on register-permutation cost; reuse the dead FP16 dO stage after dP | About `1-3%`, net of added LDGSTS | No shared growth, fifth barrier, active-stage overwrite, ordinary-load degradation, or net shared-wavefront increase |
| 2 | Half-pair P/dS C-fragment handoff | Keep each owner's local half in registers and publish only the remote P or dS half | About `2-5%` | Halve canonical P/dS stores and loads, preserve the dS mirror, prove the packet mapping, and avoid smooth-aligned spill |
| 3 | Two-D16 grouped dQ consumption | Reduce stable dS mirror copies from `32` to `16` per temporal M32 pair without the rejected four-slice lifetime | About `0-3%` | Compiler-only scoped-lifetime screen first; reject on any stack/local traffic |
| 4 | Role-specialized dQ/dKV schedule | Remove overlap between dQ reuse state and the persistent 64-register dKV accumulator image | Potentially `5-12%`, high risk | Written ownership, register, packet-byte, atomic-count, and barrier-arrival model before implementation |
| 5 | M32xN32 two-CTA schedule | Only straightforward tile class with a plausible `20%+` issue-concurrency ceiling | High risk; useful only if it reaches two CTAs | Static model and compile at `<=50 KiB` shared and `<=128` registers for eight warps or `<=170` for six |

Order 1b is a fallback, not a parallel candidate. Order 3 is materially different from the completed all-slice hoist because it changes the accumulator grouping and lifetime. Order 2 is materially different from the completed phased Q/dO packet cache because it replaces an existing canonical exchange and transfers only one remote half.

#### D64 Changes

There is no immediate D64 schedule rewrite. Only these source-supported follow-ups remain actionable:

| Order | Change | Trigger | First implementation gate |
|---:|---|---|---|
| 1 | Apply the non-transposed MMA-B copy path to D64 | Only after the D128 implementation is proven and the same access family is shown material in D64 | Exact fragment proof, no register increase above the spill-free control, bank counters, then ABBA |
| 2 | Replace the N256 accumulator or ownership representation | Only with a design that removes its current spill rather than reducing it incrementally | Every raw/smooth tail/aligned image at `<=128` registers, zero stack/local traffic, and no all-slot score reloads |

#### Generic Changes

- Add a compiler-first summary that records registers, stack, local traffic, shared bytes, barriers, static instruction families, and generated-unit identity for all raw/smooth tail/aligned functions.
- Keep new copy-contract parameters specialization-local until the D64, D128, and N256 machine-code effects are independently understood.
- Keep the matched NCU and CUDA-event harnesses current, including one captured main-kernel launch, source-attributed `-lineinfo` reports, and production-image restoration.

### Deferred Work

Items recorded as retained or closed in `Completed Work` are not a deferred backlog. The following items are deferred because they need a different precision, artifact, workspace, API, hardware, or accuracy contract:

| Deferred item | Revisit condition |
|---|---|
| Native preprocessing and dQ conversion cleanup | Revisit after a native D128 win, or if matched phase timing exceeds about `10%` of end-to-end latency |
| dS predictor redesign | Keep offline until a new feature beats the retained capture matrix; include table construction, workspace, and producer cost before integration |
| INT8 dP path | Quantize V and use INT8 dO/V for dP only on a separate precision branch with full dS and final-gradient replay; the disclosed SageBwd path keeps dP FP16 |
| INT4 backward | Revisit only after INT8 has a stronger accuracy margin and a demonstrated end-to-end need |
| Alternate Q/K quantization blocks | Require separately generated and accuracy-qualified artifacts rather than changing the retained Q32/K64 schedule in place |
| Forward-owned Q/K or P metadata and integrated forward/backward routing | Require a separate state-owning training API and complete producer, storage, and end-to-end accounting |
| Split-dQ workspace and separate reduction | Require a separate ownership/workspace branch with reduction-order accuracy and launch-cost accounting |

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

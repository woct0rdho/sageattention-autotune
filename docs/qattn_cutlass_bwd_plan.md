# QAttn CUTLASS Backward Kernel Plan

## Current State

The maintained backward path targets native SM86, fp16, non-causal fixed-length attention with head dimension 64 and NHD/HND layouts. The only generated backward specialization is:

```text
QBlock=32, KBlock=64, CTA M64xN128, eight warps
```

`_BWD_CONFIG` remains `(32, 64, 64, 128)`. The old CTA-N=64 geometry and all runtime dS-policy IDs have been removed. The active kernel always uses one pre-backward separable dS-scale predictor; the maintained CUTLASS path has no runtime format toggle.

The old filename/module `sm80` is a legacy identifier. `setup.py` emits only `compute_86,sm_86`. The installed FlashAttention 2.9.1 wheel remains the reference implementation; a targeted SM80/SM86 source probe showed no material architecture-specific advantage for its relevant specialization.

The retained source includes the scalar/index hoists, four-barrier schedule, packed-K reuse across dQ M halves, direct producer-local P/dS register handoff, single K-scale load site, and dead shared P-scale cleanup. It contains none of the rejected dimension-owned or compact plain-scratch implementations.

Under the retained backward-reuse benchmark contract, Sage is faster than the FlashAttention 2.9.1 reference in every 4096/8192, 16/32-head, NHD/HND row. Final raw-K timing gives `1.085-1.115x` end-to-end backward speedup with a `1.099x` geometric mean, and `1.098-1.152x` kernel-only speedup with a `1.129x` geometric mean. This is a demonstrated result for the scoped native-SM86 workload, not a general claim across architectures, attention modes, or head dimensions.

Two qualifications remain before treating this as a drop-in training-path win. First, the normal head-64 CUTLASS forward now quantizes Q/K into the same Q32/K64 per-block domain as backward, but it still publishes only output/LSE; no public combined forward/backward autograd path routes saved Q/K artifacts into native backward, and the backward benchmark prepares them separately. Second, the retained dS predictor is finite and fast but passes only four of six exact-4096 aggregate controls and 42 of 60 heads under the strict gate. Exact K64 dS maxima pass all six aggregates and all 60 heads and remain faster than Flash in the retained controls, but cost `1.5-3.6%` relative to prediction.

The normal forward path keeps quantization fixed at Q32/K64 for head dimensions 64 and 128 while independently selecting `(CTA_Q, CTA_K, WARP_Q, WARP_K)` from `128x64x32x64`, `128x32x32x32`, `64x64x32x64`, and `128x64x16x64`. On the final raw-K 4096/8192, 16/32-head, NHD/HND grid, best-config forward measures `1.239-1.449x` end-to-end speedup with a `1.350x` geometric mean and `1.386-1.503x` kernel-only speedup with a `1.427x` geometric mean against FlashAttention's library-selected kernel.

Combining the independently measured head-64 forward and backward-reuse end-to-end medians gives `1.136-1.174x` forward-plus-backward speedup, a `1.151x` geometric mean, and a `1.153x` ratio of summed Flash time to summed Sage time across the eight rows. This derived result counts Q/K quantization once in forward and excludes it from backward. It is not a measurement of an integrated autograd function; smooth-K correction and a combined training API remain deferred.

The local review is now substantially closed. Under the current four-INT8-plus-one-FP16 product mix, fixed scale domains, arithmetic order, M64xN128/eight-warp ownership, and four dependency barriers, the largest remaining stages have either compulsory work or corrected negative candidates. A material gain beyond the current margin now requires a new ownership, artifact, scale, precision, or product contract rather than another isolated address, shuffle, or barrier cleanup.

### Supported Contract

- Inputs: fp16 Q, K, V, output, and dOutput; float32 LSE.
- Layouts: NHD and HND.
- Attention: dense, non-causal, fixed length.
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
  emit Q32 factors: max row ||dO||2 and max row |Delta|
  emit K64 factors: max row ||V||2

CUTLASS K128 main kernel:
  recompute QK with int8 MMA
  compute dP with fp16 MMA and fp32 accumulation
  reconstruct P and dS
  quantize P with a local scale
  synthesize one dS scale from Q32/K64 factors
  accumulate dV, dK, and dQ with int8 MMA
  write owned dK/dV and atomically accumulate dQ

Triton postprocess:
  convert fp32 dQ accumulation to fp16
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
  dS is mirrored by N32 pair for the cross-warp dQ consumer
    -> pre-dK/dV CTA barrier
  direct P/dS registers remain warp-local across the barrier
  each N16 warp accumulates owned dK/dV across all four D16 blocks
  loader warps prefetch the next Q/dO pair and row state
  even n_tile warps load each packed K fragment once and reuse it for both dQ M16 halves
  dQ uses the now-dead owned score-pair slot as a private transpose stage
    -> end-of-iteration CTA barrier
final dK/dV epilogues reuse the dead dS mirror slots
```

There are four static CTA barriers. The resident-V region is aliased by the Q/dO packet publication only after V fragments have been captured in registers. The pre-dK/dV barrier publishes the Q/dO packets and mirrored dS; P and dS for dV/dK no longer cross shared memory. There is no post-dK barrier before dQ. The end-of-iteration barrier both publishes the next asynchronous Q/dO/row-state load and prevents the next iteration from overwriting shared mirror, packet, and dQ staging storage still in use.

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

The active grid maps `blockIdx.x` to one N128 CTA block, `blockIdx.y` to head, and `blockIdx.z` to batch. Within each CTA, `n_tile=warp_id`, `n_pair=n_tile/2`, and `n_domain=n_tile/4`. Only even `n_tile` warps produce dQ; four producer warps cover the four D16 slices and atomically add their accumulated M16 fragments to the common global fp32 dQ workspace. The bank-aware shared transpose changes the register-to-atomic staging and coalescing, not the number of global dQ contributions.

FlashAttention's `compute_dq_dk_dv_1colblock` has the same ownership-level reduction: one CTA owns one K/N column block and contributes a dQ partial for that block. Its normal path accumulates into global `dq_accum`; deterministic mode adds `blockIdx.x * dq_accum_split_stride` and writes a separate split for a later reduction. A Sage split workspace, larger-N CTA, or separate reduction would therefore be a new ownership contract, not a local replacement for `dQ_stage_offset()`.

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
4 passed
```

The matrix covers NHD/HND and aligned/tail sequence lengths 64/65 for the single active configuration. During integration, two correctness issues were fixed:
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

These CUDA-event timings use the final retained image on an NVIDIA GeForce RTX 3080 Ti Laptop GPU with batch 1, head dimension 64, 16/32 heads, 25 warmups, and 100 repeats. FlashAttention uses its normal library dispatch and is not constrained to the Sage CTA shape. Effective backward throughput uses `8 * batch * heads * head_dim * seq_len^2` FLOPs and the median CUDA-event time.

`kernel_only` includes the CUTLASS launch and Triton dQ conversion after prequantized Q/K and precomputed workspaces. `end_to_end` follows the backward-reuse contract: it reuses prequantized backward-format Q/K and includes Triton preprocessing, workspace/output allocation, the CUTLASS launch, and dQ conversion. It excludes Q/K quantization and the forward pass.

| Layout | Heads | Seq | Flash TFLOPS | Sage end-to-end TFLOPS | End-to-end speedup | Sage kernel-only TFLOPS | Kernel-only speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| NHD | 16 | 4096 | 29.389 | 32.720 | 1.113x | 33.479 | 1.139x |
| NHD | 16 | 8192 | 31.400 | 34.744 | 1.107x | 36.182 | 1.152x |
| NHD | 32 | 4096 | 31.050 | 33.693 | 1.085x | 34.810 | 1.121x |
| NHD | 32 | 8192 | 32.311 | 35.076 | 1.086x | 35.473 | 1.098x |
| HND | 16 | 4096 | 29.245 | 32.601 | 1.115x | 33.338 | 1.140x |
| HND | 16 | 8192 | 31.097 | 34.371 | 1.105x | 35.728 | 1.149x |
| HND | 32 | 4096 | 30.558 | 33.161 | 1.085x | 34.264 | 1.121x |
| HND | 32 | 8192 | 31.771 | 34.746 | 1.094x | 35.394 | 1.114x |

The end-to-end backward range is `32.601-35.076 TFLOPS`; the kernel-only range is `33.338-36.182 TFLOPS`. Full rows are in `build/bench_sagebwd_cutlass_rawk_NHD_4096_8192.csv` and `build/bench_sagebwd_cutlass_rawk_HND_4096_8192.csv`.

#### Derived Forward Plus Backward

The following head-64 totals sum the independently measured forward end-to-end median and backward-reuse end-to-end median for each implementation. Sage Q/K quantization is therefore counted exactly once in forward. Total effective throughput uses `12 * batch * heads * head_dim * seq_len^2` FLOPs. No combined forward/backward function was implemented or timed.

| Layout | Heads | Seq | Flash total ms | Sage total ms | Flash total TFLOPS | Sage total TFLOPS | Overall speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| NHD | 16 | 4096 | 6.4250 | 5.5874 | 32.087 | 36.897 | 1.150x |
| NHD | 16 | 8192 | 24.1065 | 20.6361 | 34.208 | 39.961 | 1.168x |
| NHD | 32 | 4096 | 12.2203 | 10.7474 | 33.740 | 38.364 | 1.137x |
| NHD | 32 | 8192 | 47.2862 | 41.2436 | 34.878 | 39.988 | 1.147x |
| HND | 16 | 4096 | 6.4506 | 5.6289 | 31.959 | 36.625 | 1.146x |
| HND | 16 | 8192 | 24.4545 | 20.8301 | 33.721 | 39.589 | 1.174x |
| HND | 32 | 4096 | 12.3612 | 10.8810 | 33.356 | 37.893 | 1.136x |
| HND | 32 | 8192 | 47.7552 | 41.4622 | 34.536 | 39.778 | 1.152x |

Overall speedup spans `1.136-1.174x`; the geometric mean of row speedups is `1.151x`, while the ratio of all summed Flash medians to all summed Sage medians is `1.153x`. Absolute laptop timings remain clock- and temperature-sensitive, and sub-percent candidate decisions continue to require paired same-build controls rather than these throughput tables.

### Accuracy and Predictor Evidence

The predictor experiments use two independent SDXL-derived sequence-4096 captures with ten heads and sigma indices `300/700/900`. They established:
- A true pre-forward predictor is not viable. Same-timestep dOutput RMS differed by approximately `28x`; same-RMS random dOutput still changed block scales by p99 ratios up to `31.5x` and a maximum around `642x`.
- The pre-backward separable predictor is the useful asymptotic class. In the simulator, using dequantized Q/K and exact-dS references to isolate scale-policy error, the base formula measured mean/min dQ cosine `0.996286/0.993382`, dK `0.998667/0.994683`, mean/max dQ relative error `7.73%/11.50%`, and dK `4.10%/10.31%` on both captures and all ten heads.
- A dOutput-max-derived approximation measured mean/min dQ cosine `0.996062/0.992853`, dK `0.998681/0.994846`, mean/max dQ relative error `8.00%/11.95%`, and dK `4.10%/10.15%`; peak saturation was below `64 ppm` in that experiment.
- The active safety factor is `1.5`. The extended replay covers both captures, all ten heads, exact sequence 4096, sliced 512/513 controls, and a deterministic tiled 8192 control. The 8192 rows exercise the long kernel schedule but are synthetic extensions of sequence-4096 data, not independent natural-length captures.
- All 24 aggregate cases and all 240 per-head cases are finite. Fifteen of 24 aggregate rows and 144 of 240 per-head rows pass the strict promotion gate. At exact sequence 4096, four of six aggregates and 42 of 60 heads pass. Mean dQ/dK relative error remains `5.49%/4.42%`, worst relative error is `11.88%/19.06%`, and mean/worst dV relative error is `0.86%/1.70%`. Both sigma-900 records fail the strict aggregate gate; the seed-2 record also fails dQ and dK cosine, relative, and maximum-absolute gates.
- Reconstructed source-order telemetry matches the kernel's P `32x16` warp scale, dS `32x64` predictor scale, reciprocal multiplication, and saturating conversion. At sequence 4096, P has zero clipping, `0.024-0.105%` zeros, and approximately `0.207%` endpoint saturation, which is expected from local max scaling. dS has `22.35-42.67%` zeros, `7.47-29.34 ppm` endpoint saturation, and `7.31-28.94 ppm` clipping across the six aggregate records.
- Rare predictor misses are real but do not explain the accuracy ranking alone. The aggregate maximum true-dS/predicted-dS ratio reaches `30.04x`; however, the worst gradient record clips only approximately `10 ppm`, while the other sigma-900 record clips approximately `29 ppm`. Across all shape controls, aggregate P-scale medians span `4.48e-6` to `7.73e-5`, dS-scale medians span `2.10e-8` to `6.41e-5`, and the evaluator emits fixed log2 histograms plus min/p01/p10/p50/p90/p99/p999/max percentiles.
- The staged FP32 reconstruction matches the real predicted-scale kernel within `0.08%` relative error at exact sequence 4096. Q/K reconstruction alone and exact-max dS quantization each pass all six aggregate and all 60 per-head strict gates; predictor-scaled dS falls to four of six and 42 of 60, exactly matching the real kernel. On the two sigma-900 aggregates, Q/K reconstruction gives dQ/dK relative error `1.15%/0.75%` and `1.14%/1.21%`; exact-max dS gives `1.53%/0.79%` and `2.16%/2.82%`; prediction raises them to `9.58%/3.19%` and `11.88%/19.06%`. P and dOutput quantization remain secondary and dV passes every aggregate.
- A scalar guard sweep from `0.25x` through `4.0x` does not pass either sigma-900 aggregate; the best setting passes only five of 20 heads. Multiplying the bound by the already-computed P maximum improves the best point to nine of 20 heads and one of two aggregates, but no coefficient passes both. Lower scales trade zero rate for hundreds to thousands of clipping ppm, while higher scales increase zeros and dK error. Neither scalar-only formulation is promoted.
- A second-order Q32/K64 sketch is materially better calibrated than the retained separable bound. The tested 2x4 formulation summarizes Q/K and dOutput/V centroids, isotropic RMS radii, mean LSE, and Delta dispersion, then evaluates eight centroid pairs per 2,048-value dS block. Its raw predicted/true scale ratio is `0.484x/1.095x/1.851x` at p10/p50/p90, versus `2.024x/4.530x/8.582x` for the active guarded predictor. A `3x` sketch guard improves exact-4096 strict passes from four of six aggregates and 42 of 60 heads to four of six and 47 of 60; the two sigma-900 dQ/dK errors become `7.21%/3.02%` and `6.67%/10.25%`. The calibration gain is real, but both hard aggregates still fail.
- Sparse exact representatives protect the sketch's miss tail but do not reach exact-scale accuracy. Selecting eight Q rows and eight K rows from LSE/dOutput/Delta and K/V norm features evaluates 64 of 2,048 pairs. `max(1.5 * sketch, 4 * sparse_max)` reaches five of six aggregates and 52 of 60 heads; its sigma-900 dQ/dK errors are `5.69%/2.95%` and `5.24%/7.84%`. Expanding to 16x16 representatives, 256 of 2,048 pairs, still reaches only five of six and 52 of 60, with the remaining dK error `6.98%`. Even an oracle that replaces the 32 worst-predicted K64 scales out of 64 per Q block reaches only five of six aggregates, although it reaches 58 of 60 heads. The remaining error is distributed rather than confined to a small top-K set.
- Held-out log-scale regression confirms that the sketch features contain useful information without establishing a production policy. Training on one capture and evaluating the other, an eight-feature model using the retained bound, 2x4/4x8 moments, and 32/64-pair sparse lower bounds produces p10/p50/p90 predicted/true ratios `0.591x/1.025x/1.424x` and p90 absolute log2 error `0.819`, versus `3.101` for the active predictor. Mean guards and conditional `0.65-0.90` quantile models still reach at most five of six aggregates in reconstructed gradients. Two captures are also insufficient evidence for learned coefficients, so no learned runtime dependency or coefficient set is promoted.
- Generic 8/16/32-dimensional random projections are rejected for the score term. Projection error is exponentiated by softmax; all tested guards pass zero of six aggregates, and the unguarded maximum dS zero rates are approximately `84.9%`, `67.4%`, and `46.6%`. Random projections may still summarize the linear dOutput/V residual, but they are not a viable approximation to QK logits here.
- The sketch cost is much lower than exact pairwise preprocessing but is not strictly linear. A 2x4 centroid combine evaluates `8/2048 = 0.39%` of the exact block-pair count; 8x8 and 16x16 sparse representatives evaluate `3.125%` and `12.5%`. Materializing one FP32 Q32-by-K64 scale table for sequence 8192 and 16 heads requires approximately 2 MiB. A future candidate must count that table, the block-pair combine, and selection overhead in end-to-end timing rather than describing it as the current `O(N*D)` predictor.
- Restoring exact on-the-fly K64 dS maxima passes all six exact-4096 aggregates and all 60 heads, but loses every timing row. Averaging the two exact-scale runs around the predicted control gives `31.52-34.12` end-to-end and `33.16-34.91` kernel-only TFLOPS, versus `32.26-35.15` and `33.86-35.96` for prediction. Raw exact-scale medians regress `1.5-3.6%`. Exact scaling remains faster than same-run Flash by `1.063-1.084x` end to end and `1.092-1.140x` kernel-only, but the predictor reaches `1.090-1.134x` and `1.122-1.187x`.

The obsolete block-omission experiment is no longer part of the maintained data path.

### Measured Bottlenecks

In the matched-shape NHD sequence-8192 attribution capture, Sage took `19.54 ms` versus `22.85 ms` for the Flash backward main kernel, or `0.855x` Flash duration. Sage executed `1.613B` instructions versus Flash's `617.6M` (`2.61x`), `33.6M` HMMA instructions versus `167.8M` (`0.20x`), `285.6M` versus `336.1M` shared-load wavefronts (`0.85x`), and `44.7M` versus `36.3M` shared-store wavefronts (`1.23x`). Sage had higher eligible warps (`0.43` versus `0.13`) but higher barrier-stall share (`15.79%` versus `12.83%`).

A separate retained full Speed Of Light profile reports `20.61 ms`, `1.592B` executed instructions, `53.47%` compute throughput, `3.10%` DRAM throughput, `54.56%` L1 throughput, and `22.30%` L2 throughput. Capture-to-capture duration differences reflect profiler sections and run conditions; paired CUDA-event timing remains the promotion gate. Both profiles show that Sage's remaining internal cost is non-tensor reconstruction, conversion, address/fragment plumbing, and cross-warp staging rather than DRAM bandwidth or missing tensor-core work.

#### Reduction Status

The exact on-the-fly `abs(dS)` maximum reduction is no longer present. The active kernel still pays for the local P scale reduction: scalar `fmaxf` updates build `p_max_abs`, followed by one warp-level `__reduce_max_sync` before P is quantized. This is a real non-tensor cost, but it is narrower than the removed dS maximum reduction and should be treated as one part of the P reconstruction path rather than as the main remaining dS-scale mechanism.

The Q32/K64 predictor summaries are computed in Triton with `O(N*D)` reductions. They do not create the native shared-store traffic. Replacing those reductions is lower priority unless an end-to-end phase breakdown shows that preprocessing, rather than the CUTLASS launch, has become material.

#### P and dS Reconstruction

Each K128 iteration reconstructs P and dS around the int8/fp16 MMA results. The path includes int32-to-fp32 conversion, score and softmax scaling, `expf`, LSE and Delta subtraction, dS formation, absolute-value/max tracking, and scale arithmetic. P and dS are then quantized with reciprocal-scale application and integer conversion. This combination explains a substantial part of the excess non-tensor instruction activity and includes the remaining P warp reduction.

#### Shared-Memory Handoffs

The producer-local canonical P/dS score-pair handoff is removed. The active path still materializes several fragments through shared memory:
- dS mirror stores into `dS_dKV` so the dQ path can consume the same reconstructed values.
- Packed Q/dO fragment stores and later loads for the dK/dV MMAs.
- Packed K fragment stores and later loads for the dQ MMAs.
- Shared dQ transpose/staging before the global atomic accumulation.
- dK/dV epilogue staging before coalesced global stores.

The matched sequence-8192 NCU capture confirms that the register handoff reduces shared-store instructions from `110,428,160` to `43,319,296`, store wavefronts from `112,284,831` to `44,650,061`, load wavefronts from `302,383,104` to `285,605,888`, and LDSM executions from `48,300,032` to `44,105,728`. The fresh line-info SourceCounters profile then attributes the remaining shared work: dOutput FP16 fragment reads for dP consume approximately `84M` LDSM wavefronts; packed dO/Q consumer loads for dV/dK consume approximately `67M` LDS wavefronts; dQ's packed-K and mirrored-dS loads consume approximately `50M` LDS/LDSM wavefronts; dS mirror stores consume approximately `17M` STS.U16 wavefronts; and dQ transpose staging adds approximately `17M` loads plus `17M` stores. The only large excessive-wavefront sources are the two producer Q/dO packet transposes at approximately `8.39M` each and the next-pair Q/dO `cp.async` path at `20.89M` wavefronts, of which `4.18M` are excessive. Global dQ reduction sectors already match Flash, so an ownership change must be justified by a complete atomic/workspace cost model rather than by the local shared counters.

#### Scalar Scale Application and Fragment Plumbing

The dV, dK, and dQ accumulation paths apply FP32 products involving P, dS, dOutput, Q-block, K-block, and output scales around the integer MMA fragments. Manual packed-fragment indexing, address arithmetic, tail predicates, conversions, and repeated synchronization add further non-tensor instructions and scheduler stalls. These operations also constrain attempts to keep P/dS values in registers because producer lifetimes already approach the resource limit.

#### Executed-Stage Ranking

The source-correlated native profile groups executed instructions as dV/dK accumulation `24.96%`, P/dS reconstruction `23.53%`, quantization/C-to-A/dS mirror publication `17.55%`, dQ transpose/atomic accumulation `7.61%`, dQ scale and accumulation `5.59%`, next-pair prefetch/state `4.35%`, and Q/dO packet producers `4.29%`. The first three account for `66.04%`. Within dV/dK, approximately `201.3M` `I2FP.F32.S32`, `134.2M` scaled `FFMA`, `67.1M` LDS, and `33.6M` INT8 MMA instructions are executed. The conversion/scaled-accumulation path is therefore the largest isolated target, but the offline and native candidate results below close it under the current scale contract.

#### Remaining-Work Classification

| Stage | Status under the retained contract | Consequence |
|---|---|---|
| dV/dK conversion and scaled accumulation | Exhausted locally | Independent P/dS, Q, and dOutput scales require conversion and FP32 products; grouped, FP16-outer, and custom-conversion candidates did not win |
| P/dS reconstruction | Locally reducible only at small margin | QK, exponentiation, P, and dS are required unless forward artifacts or the numeric contract change |
| Quantization, C-to-A handoff, and dS publication | Mostly exhausted | Producer-local P/dS stores are removed; mirrored dS remains required for cross-warp dQ |
| dQ transpose and atomic accumulation | Local staging retained; ownership architectural | The shared transpose is the best tested local path and leaves global contribution count unchanged |
| Q/dO packet and next-pair traffic | Tested local schedules exhausted | Current prefetch overlaps dQ; early-dO and tail-prefetch variants regressed |
| Forward-owned Q/K or P metadata | Contract-changing | Both forward head dimensions use Q32/K64, but the public forward still discards Q/K artifacts; head-128 backward and full P checkpointing remain separate contracts |

The practical conclusion is that large-margin gains are exhausted inside the current fixed four-INT8 pipeline. Remaining local ideas have ceilings bounded by small stage shares or require changing quantized values. Larger gains must be evaluated as architectural or end-to-end product changes.

## Completed Work

This section records the decisions behind the current source and prevents closed experiments from being repeated without new evidence. An implementation defect invalidates only that candidate build and its measurements, not the underlying design. An experiment is closed only after the defect is fixed and the corrected candidate is validated, or when independent design-level accuracy or contract evidence rules out that specific formulation.

### Completed Audits

- The resource audit reports 243 registers per thread for tail and aligned variants, zero stack/local spill, and 57,600 bytes of dynamic shared storage. `__maxnreg__(243)` is the explicit register boundary. The historical 57,632-byte NCU image still correctly establishes one resident block, 16.7% theoretical occupancy, and approximately 0.40 eligible warps per cycle; the final 32-byte cleanup does not change occupancy.
- The post-handoff line-info SourceCounters profile maps all 2,088 active SASS instructions to `qk_int8_sv_f16_bwd_kernel_cutlass_sm80.cuh`. The largest instruction families are `FMUL.FTZ` (`220.2M`), `FFMA.FTZ` (`211.8M`), `I2FP.F32.S32` (`201.3M`), `F2I.S8.NTZ` (`67.1M`), `PRMT` (`84.0M`), and `MOVM`/`SHFL.IDX` (`33.6M` each). `MUFU.EX2` accounts for `33.6M` exponent operations. Source attribution is therefore available when the extension is built with `-lineinfo`; the normal retained binary has identical machine instructions without the debug correlation metadata.
- The barrier audit established four dependency boundaries. Removing the startup K/V publication barrier produced 4,096 Racecheck hazards between asynchronous shared writes and cross-warp `LDSM` reads. The pre-dK/dV barrier publishes Q/dO packets and dS. The post-dK barrier before dQ is safely removed, and the end barrier protects next-pair Q/dO and row-state publication.
- The historical canonical dS audit established that dQ needs one K32 MMA-A fragment assembled from two N16 producer slots. The current `LDSM.x4` atom cannot span the 1 KiB-separated mirror slots under CuTe's vectorization contract, and the initial scalar C-fragment/register shuffle mismatched all 32 lanes.
- The canonical two-source proof was corrected before promotion. The first standalone probe accidentally applied `make_transposed_tensor` to both the canonical score-pair destination and the mirror reference; production intentionally stores canonical dS transposed and the dQ mirror non-transposed. With the exact production orientations and source-distinct pseudorandom fragments, neither normal nor `.trans` `LDSM.x2` plus any uniform lane shift matched the MMA-A reference.
- Exhaustive low-instruction mapping checks closed the direct two-source x2 mirror formulation. None of 3,840 lane-bit-permutation/XOR address maps matched either M half. Byte-signature mapping showed that a normal load scatters each eight-byte target across eight source lanes, while a transposed load scatters it across four. A modeled hardware register transpose requires at least three `movmatrix`, two lane-permutation, and two local-byte-permutation stages per source fragment. That remains noncompetitive for eliminating the cross-warp dQ mirror, but it does not apply to the producer-local C-pair handoff used by dV/dK.
- A separate producer C-pair proof found the exact four-stage bit network `MOVM -> lane rotate -> byte 1/2 swap -> MOVM`. Coordinate-tagged data match all 512 bytes of the canonical MMA-A fragment. Integrating that mapping removes the canonical P/dS stores and reloads while retaining the dQ mirror and four-barrier ownership schedule.
- Matched sequence-8192 retained-image attribution shows `0.9940x` profiled duration, `0.9132x` LDSM execution, `0.9445x` shared-load wavefronts, and `0.3976x` shared-store wavefronts versus the saved pre-handoff control. Shared-store bank conflicts fall `12.8%`, eligible warps rise from `0.42` to `0.43` per cycle, and tensor instructions are identical. Total executed instructions rise `2.1%` because of the register transpose, while the measured latency still improves.
- The long-shape telemetry and error-decomposition audit is complete. `data/evaluate_cutlass_bwd_captures.py` now reports aggregate and per-head gradient cosine, relative error, maximum absolute error, finite and strict-gate status, Q/K/dOutput/P/dS INT8 zero rates, P/dS endpoint saturation and clipping, P/dS scale percentiles, true-dS/predicted-dS percentiles, and fixed log2 histograms. It also compares Q/K reconstruction, exact-max dS, predicted dS, local P quantization, dOutput quantization, and the real INT8 kernel. Its default matrix covers both captures and 512/513/4096/8192 with explicit `slice`, `exact`, and `tiled` provenance labels.
- The packet and lifetime audit found no free compaction margin. Q/dO/K packet allocations exactly match their fragment counts; the K packet is `16 fragments * 4 words * 32 lanes * 4 bytes = 8,192 bytes`. Lowering shared storage alone cannot create a second resident block, so double buffering requires a proven dead region or a different ownership schedule.

### Completed Design Reviews

- dQ ownership and Flash comparison. Sage and Flash both produce one dQ partial per CTA-owned K/N block. Sage's four even producer warps cover D16 slices and atomically accumulate into one fp32 workspace. Flash's deterministic split pointer is an alternate workspace/reduction contract, not evidence that Sage's shared transpose reduces global atomics.
- Local dQ staging comparison. Same-build NCU duration improved from `180,896 ns` for the shuffle path to `168,448 ns` for the bank-aware shared transpose. Shared-load/store bank-conflict counts moved from `271,226/9,361` to `282,969/57,887`, showing that the faster result comes from the complete staging/coalescing schedule rather than conflict count alone. Same-build timing favored shared transpose in all four controls by approximately `4.5-4.8%`: NHD `23.41548 -> 22.28124 ms` at 8192 and `5.96204 -> 5.69924 ms` at 4096; HND `23.19785 -> 22.18089 ms` and `5.92862 -> 5.67287 ms`.
- Forward/backward artifact compatibility. CUTLASS forward uses the same Q32/K64 Q/K scale domain at head dimensions 64 and 128, including K32 CTAs sharing one K64 scale and Q16 warps sharing one Q32 scale. Native backward consumes that contract only at head dimension 64. The forward wrapper still discards those quantized tensors after producing output/LSE, so the backward-reuse benchmark remains a product-contract proxy rather than an integrated training invocation.
- Forward-owned P artifacts. Forward computes P transiently inside online softmax and publishes only output/LSE. A saved P maximum can remove only the local P maximum reduction; it cannot remove QK recomputation, exponentiation, P reconstruction, or dV's P operand. Saving quantized P itself is `O(N^2)` and is not feasible under the current memory contract.
- dS scale-fusion review. No existing scale-cancellation implementation was found. An unscaled-dS representation could move `sm_scale` in real arithmetic, but the scale floor, reciprocal, rounding, saturation, and dequantization products change the quantized contract. This remains a bounded offline hypothesis, not a source-level optimization.
- Q/dO lifetime review. The current next-pair `load_q_dO_pair()` is issued after current dK/dV has consumed the packed Q/dO operands, while dQ no longer reads the raw `q_i8`, `dO_i8`, or `dO_fp16_pair` buffers. This safely overlaps asynchronous loads with dQ. Moving the load after the end barrier is race-safe only by giving up that overlap; moving it across an earlier dependency requires a new publication proof. Raw double buffering is not justified with the remaining shared-memory headroom.
- Sixteen-warp feasibility. Residual shared-storage support and a historical launch policy establish only that a 16-warp shape can be represented. No active generated `nw16` specialization or benchmark exists. SM86 allocates the same static register count to every warp, so the 243-register live set must approach roughly 128 registers/thread before a 16-warp design has a credible occupancy case; otherwise it spills or remains occupancy-limited.

### Retained Changes

- Unified Q32/K64 CUTLASS forward contract. The normal configured/public paths for head dimensions 64 and 128 quantize with flat Q32/K64 metadata and launch one fixed scale path while retaining all four CTA/warp execution instantiations. The public API, eager autotune, and compile autotune remain supported. Sequence-129 NHD/HND tests cover both head dimensions and all four execution tiles with and without LSE.
- Native backward dead-surface cleanup. The internal operator no longer accepts `output` or final fp16 `dQ`, because Triton preprocessing consumes output before launch and Triton postprocessing converts the fp32 dQ workspace afterward. Their unused launch strides and rejected, unreferenced dQ/scale/transpose helpers were removed. Generic head-dimension and tile traits plus generated-instantiation plumbing remain available for a future dedicated D128 design.
- K128 main-kernel baseline. The active generated specialization is QBlock=32, KBlock=64, CTA M64xN128, eight warps. The former backward K64 CTA is removed from the current generation and dispatch, but alternate quantization blocks and attention CTA/warp shapes remain valid future search dimensions if measurements justify them.
- Triton utility phases. Delta/dOutput preprocessing, Q32/K64 predictor summaries, workspace allocation, and fp32 dQ-to-fp16 conversion moved out of the native wrapper and into autotuned Triton kernels. The native operator now launches the fused CUTLASS main kernel with explicit, typed workspaces.
- Unconditional separable dS prediction. The retained performance predictor is `1.5 * (softmax_scale / N) * (max_row ||dO||2 * max_row ||V||2 + max_row |Delta|)`. It is `O(N*D)`, uses Q32/K64 summaries, and does not allocate an `O(N^2)` scale table or select a runtime policy. It is not the strict-accuracy oracle.
- Quantization constants and conversion. The accepted reciprocal is `0x1.010122p-7`, the scale floor is `2^-126`, and C++ INT8 conversion is saturating. Triton dOutput quantization uses the shared unclamped `_round_to_int8` helper because its exact max-abs scale preserves the intended range.
- Guard selection. The fixed `1.5x` dS guard is retained as the performance reference. A `1.25x` guard was faster on the six retained captures but was less robust on the sequence-65 correctness control; `2.0x` increased quantization error. The expanded scalar sweep confirms that no global guard or simple P-max multiplier clears both sigma-900 records, so guard tuning is closed without a new predictor structure.
- Training-derived validation data. Two independent SDXL-derived sequence-4096 captures with ten heads and sigma indices `300/700/900` are retained with reusable capture, simulation, benchmark, and evaluator scripts under `data/`. The full replay is finite, but exact sequence 4096 passes only four of six aggregate and 42 of 60 per-head strict gates. Mean dQ/dK relative error is `5.49%/4.42%`, worst error is `11.88%/19.06%`, and sparse dS clipping does not by itself predict those failures; this remains exploratory rather than strict-promotion quality.
- Validation baseline. The active build passes the four-case focused matrix, native-only Racecheck and Synccheck for NHD/HND at 512/513, the four-row new-contract timing baseline, SASS/resource inspection, aggregate NCU capture, `pre-commit`, compilation, and whitespace checks.
- Scale/index hoists. The active kernel caches the CTA-invariant K-domain index, reuses the Q-block index for predictor metadata, caches the dK scale product and dOutput scale-index base, and specializes K-scale lifetime so only aligned launches retain the CTA-invariant scale across M-pairs. With `-lineinfo` on native SM86, aligned static SASS falls from 2096 to 2088 instructions while the tail function is instruction-identical to baseline; registers remain 243 aligned and 227 tail with zero stack/local memory. On identical NHD/HND inputs at sequence 513/4096, dK/dV are bitwise identical and dQ differs only in 7-314 FP16 elements by at most `6.10e-5` with relative L2 below `1e-5`. Kernel-only timing moved by less than the laptop GPU clock variation in the paired candidate/baseline runs, so this is retained as a low-risk control rather than a promotion-quality speedup claim.
- Post-dK barrier removal. The synchronization point between dK/dV accumulation and dQ staging is removed. Native SM86 SASS now has four static CTA barriers instead of five; relative to the five-barrier scale/index-hoist build, tail SASS is seven instructions shorter and aligned SASS removes one `BAR.SYNC.DEFER_BLOCKING` while adding one `NOP`. `__maxnreg__(243)` preserves the 227-register tail and 243-register aligned allocations with zero stack/local spill. Racecheck and Synccheck pass NHD/HND at sequence lengths 512 and 513. Paired kernel-only timing at 4096/8192 was mixed and within laptop clock/thermal drift, so the cleanup is retained without a performance-win claim.
- Packed-K dQ reuse. The dQ phase now loads each packed K fragment once per even warp and consumes it for both M16 halves, rather than reloading it for each half. It also computes each K-domain predictor scale once per pair. Native SM86 SASS falls by 16 `LDS` instructions and 22/13 total instructions in tail/aligned variants; the static CTA barrier count remains four in the retained image. Both variants use 243 registers with zero stack/local spill. Racecheck and Synccheck pass NHD/HND at 512/513. Paired new-contract timing is clock-sensitive in absolute milliseconds, but same-run Flash-normalized ratios improved by approximately `0.7-1.2%` for NHD and `0.7-3.2%` for HND across 4096/8192 kernel-only and end-to-end rows, so the change is retained.
- Bank-aware shared dQ transpose. The active `dQ_stage_offset()` layout stages each 16x16 fp32 accumulator in a private dead score-pair slot and issues contiguous global atomics. It preserves the same cross-CTA reduction count as the prior shuffle path but wins every same-build NHD/HND control by approximately `4.5-4.8%`. Direct, row-major, affine, fp16, int32, and alternate transposed staging variants did not beat it. Keep this as the local dQ baseline; revisit dQ only through a materially different ownership contract.
- Direct producer P/dS register handoff. Each warp converts its two M16 producer C fragments into the N16xM32 MMA-A fragments used by dV and dK with a proven `MOVM`, lane-rotation, byte-permutation, `MOVM` network. The dQ mirror remains shared because it is cross-warp. Relative to the saved control, each specialization removes 32 static `STS` and two `LDSM` instructions and adds 16 `MOVM` and eight `SHFL`; tail PRMT count falls by four while aligned PRMT count rises by 28. Both variants remain at 243 registers, 57,600 bytes shared, four static CTA barriers, and zero stack/local spill. Focused tests pass; dK/dV are bitwise identical; Racecheck and Synccheck pass NHD/HND at 512/513. Paired same-build timing reports `0.9941x` aggregate raw latency, and matched NCU reports `0.9940x` duration with about 60% fewer shared-store instructions and wavefronts. The two candidate/control orders did not disagree enough to warrant separate result rows. The handoff is retained.
- Single K-scale load site. Both aligned and tail specializations call `load_k_block_scale()` at one source location inside the M-pair loop. Resources remain 243 registers with zero stack/local spill and tail SASS is unchanged; ptxas emits eight additional aligned instructions compared with the manually split lifetime. Short paired timing was mixed, so this remains a source simplification without a speedup claim and should not be reopened without a new scale-lifetime design.
- Dead shared P-scale publication removal. `SharedStorage2DWarp::p_scale` was an unreferenced trailing eight-float array; the active local `p_scale` arithmetic is unchanged. Removing the field reduces shared storage from 57,632 to 57,600 bytes. Tail and aligned native-SM86 instruction streams are exactly identical to the control, resources remain 243 registers with zero stack/local spill, and the focused four-case matrix passes. This is retained as a resource cleanup without a latency claim.

### Closed Experiments

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
- Local dQ shuffle and alternate shared layouts. The bank-aware shared transpose beats the prior shuffle path in all same-build timing controls. Direct/transposed, row-major, affine-swizzled, fp16-stage, int32-stage, and synchronization variants either regressed, increased conflicts/resources, or failed to improve the full kernel. The local dQ staging search is closed unless ownership changes.
- Q/dO early-load and tail-prefetch schedules. Early dO FP16 loading was a small regression or mixed under same-build NHD/HND controls. The tested tail-prefetch schedule was clearly slower, and changing q32/k64 to q128/k128 did not rescue it. The current placement already overlaps next-pair asynchronous loads with dQ, so these source-local reorderings are closed.
- The tested broad Q/dO packetization, broad vector-packet, register-V, and split dV/dK formulations are closed because their validated builds were slower or resource-heavy. Revisit the broader ideas only with a materially different layout or ownership hypothesis and source/SASS evidence that targets a measured handoff cost.
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
- Keep direct dQ-mirror elimination closed until a producer-to-consumer proof beats the known three-MOVM/two-lane/two-byte reconstruction cost. The producer-local dV/dK handoff does not make the cross-warp dQ mapping cheaper.
- Keep the fixed predictor as the performance reference and exact-max reconstruction as the accuracy oracle. Do not retry a scalar guard, simple P-max multiplier, generic QK random projection, or the tested 2x4-plus-row-sample formulations.
- Use the retained producer-register handoff and shared dQ transpose as the native local baseline. Do not restore canonical P/dS score-pair stores or treat score-pair slots as free storage; each slot is reused by dQ staging.
- Treat the four barriers and the `0.43` eligible-warps/cycle result as diagnostics, not permission to remove synchronization. Any new schedule must identify a different ownership dependency and pass the full native sanitizer matrix.

## Next Work

The next cycle uses QBlock=32, KBlock=64, CTA M64xN128, and eight warps as the measured reference, but it should not reopen closed source-local schedules. Each candidate must state whether it changes the artifact, accuracy, ownership, scale, precision, or product contract. Preserve the current arithmetic and synchronization contract when that category is not the subject of the experiment.

### Priority Order

- Close the real forward/backward reuse contract before making a broad end-to-end speed claim. The normal head-64 forward and native backward now share the Q32/K64 scale contract, but the remaining work is to define saved-state ownership and route those tensors from a state-owning forward into native CUTLASS backward while retaining an internal-quantization compatibility path. A combined public API and its full-training benchmark are intentionally deferred for now.
- If more main-kernel margin is required, select one dQ ownership experiment only after a static cost model compares larger-N CTA ownership with a split-workspace reduction. Count atomics, fp32 workspace traffic, reduction launches, numerical accumulation order, registers, shared memory, occupancy, and tail work. Do not treat the retained shared transpose as part of this search; it is the local baseline.
- Evaluate unscaled dS and `sm_scale` fusion only as a bounded numeric-contract study. Prove quantized-value behavior with the scale floor, reciprocal, saturation, and source-order dequantization; identify the exact removed SASS family and expected stage ceiling before a native candidate.

### Architectural Candidates

- Larger N-owned CTA. N256 ownership could reduce the number of global dQ contributions, but it must fit K/V, packed-K, dS mirrors, packet storage, barriers, and a smaller live set without reducing useful residency. A 16-warp implementation is not credible at the current 243 registers/thread.
- Split dQ workspace and separate reduction. This mirrors Flash's deterministic ownership option and may replace contended atomics with regular stores, but it adds `O(splits * B * H * N * D)` fp32 traffic, workspace, a launch, and a reduction-order change. Promote only on end-to-end latency, not main-kernel duration alone.
- Warp-specialized or 16-warp schedule. Proceed only after a source/SASS design reduces the common static allocation toward roughly 128 registers/thread and accounts for spills, shared memory, barriers, and occupancy. Residual trait support is not performance evidence.
- Changed scale or product representation. A candidate may reopen dV/dK conversion only by proving a common dequantization product, integer bounds, accuracy, and fewer executed conversion/accumulation instructions. A different FP16/INT8 product mix is a separate algorithm, not a cleanup of the retained kernel.

### Deferred

- Smooth-K native backward. Keep raw K and raw-score LSE as the maintained contract for now. Centered K requires an explicit LSE-domain contract and the dQ mean-correction path before it can share forward artifacts; do not silently pass centered K into the current kernel.
- Head dimension 128 native backward. The public CUTLASS forward uses Q32/K64 at head dimension 128, but the native backward specialization remains head-64 only. Reusable `HeadDim`/tile traits and generated-instantiation plumbing are retained; the historical generic `M64xN64`, four-warp D128 backward used 250 registers per thread and 41,424 bytes of shared memory and was not competitive. Resume with a dedicated D128 packet-ownership and register-lifetime design, using M64xN64 only as a bring-up geometry, rather than adding a generated instantiation to the D64 kernel.
- Keep predictor work as a separate accuracy workstream. If it resumes, target missing softmax-concentration information rather than a larger generic guard. The strongest remaining hypothesis is a forward-owned row concentration statistic or compact top-block summary that the forward softmax can emit in `O(N)` storage; a preprocess-only alternative needs a new block-conditioned score feature that beats the tested centroid and 16x16 samples. Account explicitly for a Q32-by-K64 scale table and block-pair combine in end-to-end timing.
- Revisit quantization blocks only after the artifact contract and accuracy policy are settled. Generate every shape through `scripts/generate_cutlass_bwd_instantiations.py` and evaluate accuracy, resources, occupancy, shared counters, sanitizers, and both timing modes.
- Forward-owned P maxima are deferred as a narrow scale-reduction optimization. Full P checkpointing remains outside the memory contract; a compact forward statistic is useful only if it removes measured backward work and pays for its forward store/load cost.
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
    python data/racecheck_cutlass_bwd_kernel_only.py `
    --seq-len $case.SeqLen --layout $case.Layout
  if ($LASTEXITCODE -ne 0) {
    throw "Racecheck failed for $($case.Layout) $($case.SeqLen)"
  }

  compute-sanitizer --target-processes application-only `
    --tool synccheck `
    --kernel-name 'kernel_substring=fused_mma_kernel_k128_8warp' `
    python data/racecheck_cutlass_bwd_kernel_only.py `
    --seq-len $case.SeqLen --layout $case.Layout
  if ($LASTEXITCODE -ne 0) {
    throw "Synccheck failed for $($case.Layout) $($case.SeqLen)"
  }
}
```

Compute Sanitizer's kernel filter is a key/value expression, `kernel_substring=...`; it is not NCU's `regex:...` syntax. Keep `--target-processes application-only` and use the native-only helper so PyTorch setup, Triton compilation, and unrelated child-process kernels are outside instrumentation. Racecheck should end with `0 hazards displayed (0 errors, 0 warnings)` and Synccheck with `ERROR SUMMARY: 0 errors` for all four cases.

Benchmark both layouts and modes:

```powershell
foreach ($layout in @('NHD', 'HND')) {
  python bench/bench_sagebwd_cutlass.py `
    --batch-size 1 --num-heads 16 32 --head-dims 64 `
    --seq-lens 4096 8192 --layout $layout `
    --warmup 25 --repeats 100 `
    --block-configs 32,64,64,128 --mode all `
    --csv "build/bench_sagebwd_cutlass_rawk_${layout}_4096_8192.csv"
}
```

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
  python data/racecheck_cutlass_bwd_kernel_only.py `
  --seq-len 8192 --num-heads 16 --layout NHD
```

After exporting the source profile, clear `NVCC_APPEND_FLAGS` and restore the retained non-lineinfo binary before timing.

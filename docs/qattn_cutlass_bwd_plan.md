# QAttn CUTLASS Backward Kernel Plan

## Current State

The maintained backward path targets native SM86, fp16, non-causal fixed-length attention with head dimension 64 and NHD/HND layouts. The only generated backward specialization is:

```text
QBlock=32, KBlock=64, CTA M64xN128, eight warps
```

`_BWD_CONFIG` remains `(32, 64, 64, 128)`. The old CTA-N=64 geometry and all runtime dS-policy IDs have been removed. The active kernel always uses one pre-backward separable dS-scale predictor; the maintained CUTLASS path has no runtime format toggle.

The old filename/module `sm80` is a legacy identifier. `setup.py` emits only `compute_86,sm_86`. The installed FlashAttention 2.9.1 wheel remains the reference implementation; a targeted SM80/SM86 source probe showed no material architecture-specific advantage for its relevant specialization.

The retained source includes the scalar/index hoists, four-barrier schedule, packed-K reuse across dQ M halves, single K-scale load site, and dead shared P-scale cleanup. It contains none of the rejected dimension-owned or compact plain-scratch implementations.

### Supported Contract

- Inputs: fp16 Q, K, V, output, and dOutput; float32 LSE.
- Layouts: NHD and HND.
- Attention: dense, non-causal, fixed length.
- Head dimension: 64 after padding.
- Last dimension: contiguous; wrappers materialize contiguous inputs.
- Batch and head strides: arbitrary int32-compatible strides for the logical input layout.
- Workspace tensors: contiguous, explicitly typed, and stored in fixed `[batch, heads, sequence, dimension]` logical order where applicable.
- Out of scope: causal, variable length, GQA/MQA, additional dtypes, and head dimension 128 dispatch.

## Active Pipeline

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
  Q/dO producer warps publish four packed MMA-B fragments each
  dS is stored canonically for dK and mirrored by N32 pair for dQ
    -> pre-dK/dV CTA barrier
  each N16 warp accumulates owned dK/dV across all four D16 blocks
  loader warps prefetch the next Q/dO pair and row state
  even N-pair warps load each packed K fragment once and reuse it for both dQ M16 halves
  dQ uses the now-dead owned score-pair slot as a private transpose stage
    -> end-of-iteration CTA barrier
final dK/dV epilogues reuse the dead dS mirror slots
```

There are four static CTA barriers. The resident-V region is aliased by the Q/dO packet publication only after V fragments have been captured in registers. There is no post-dK barrier before dQ. The end-of-iteration barrier both publishes the next asynchronous Q/dO/row-state load and prevents the next iteration from overwriting shared score and packet storage still in use.

The current backward wrapper still quantizes Q/K on each call with `per_block_int8`. The backward benchmark's end-to-end mode deliberately prepares those backward-compatible Q/K INT8 tensors and scales outside the timed region, modeling the intended forward-owned reuse without claiming that the current forward artifacts are already compatible. The timed backward path includes Triton preprocessing, workspace/output allocation, the CUTLASS main kernel, and dQ conversion.

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

## Reduction Audit

The Triton reductions follow the normal Triton reduction contract and are shaped intentionally:

- `tl.sum(..., axis=1)` reduces each `[BLOCK_M, HEAD_DIM]` tile to one scalar per row for Delta and dOutput L2 norms.
- `tl.sum(..., axis=1)` similarly reduces each `[BLOCK_N, HEAD_DIM]` V tile to row L2 norms.
- `tl.max` over the resulting row vector produces one Q32 or K64 summary value.
- Full-tile `tl.max` is appropriate for the dOutput and V maxima; reducing a smaller axis would change the predictor rather than improve the implementation.

The native CUTLASS kernel has a narrower reduction footprint than the historical dS-policy path. It does not reduce the exact maximum `abs(dS)` on the fly. Instead, it synthesizes the dS scale from the Q32/K64 summaries emitted by Triton. It does retain one cross-thread maximum reduction for the local P quantization scale: `p_max_abs` is accumulated while reconstructing P and passed through `warp_reduce_max`, which uses `__reduce_max_sync`. The pairwise `fmaxf` operations used to combine predictor factors are scalar maxima, not full-tile or cross-warp reductions.

Triton 3.8 lowers its preprocessing reductions through the standard reduction primitive. There is no general faster replacement that preserves these L2/max summaries. A different reduction tree would require a custom CUDA kernel or a weaker bound and must be evaluated end to end. These preprocessing reductions are `O(N*D)` and are not the explanation for the native kernel's shared-store wavefronts; the remaining native costs are P/dS reconstruction, quantization, fragment staging, and scalar scale application.

## Correctness

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

The long-shape promotion gates remain:

```text
1 - cosine < 2e-3
relative error < 0.06
maximum absolute error < 0.20
no NaN/Inf
```

For diffusion-training data, cosine and gradient direction are primary. Relative error below `0.10` is the target and below `0.20` is exploratory; these application metrics do not replace the stricter default-promotion gate.

## Fresh Timing

These are the latest retained-control CUDA-event medians on an NVIDIA GeForce RTX 3080 Ti Laptop GPU, batch 1, 16 heads, head dimension 64, with 15 warmups and 50 serial repeats. They were collected as the control in the compact-scratch candidate/control/candidate sequence. Removing the final trailing 32-byte unused shared field left the tail and aligned instruction streams exactly identical to that control, so these rows remain the current timing baseline.

`kernel_only` includes the CUTLASS launch and Triton dQ conversion after prequantized Q/K and precomputed workspaces. `end_to_end` follows the new contract: it reuses prequantized backward-format Q/K and includes Triton preprocessing, workspace/output allocation, the CUTLASS launch, and dQ conversion. It excludes Q/K quantization and the forward pass.

| Layout | Seq | Flash ms | Sage end-to-end ms | Sage kernel-only ms | End-to-end speed ratio | Kernel-only speed ratio |
|---|---:|---:|---:|---:|---:|---:|
| NHD | 4096 | 4.5317 | 4.2250 | 4.0044 | 1.073 | 1.132 |
| NHD | 8192 | 17.3425 | 15.6600 | 15.2852 | 1.107 | 1.135 |
| HND | 4096 | 4.5014 | 4.1364 | 3.9429 | 1.088 | 1.142 |
| HND | 8192 | 17.3041 | 15.4056 | 15.2985 | 1.123 | 1.131 |

The retained path is faster than the same-run Flash backward control by `7.3-12.3%` end to end and `13.1-14.2%` in kernel-only mode across these four rows. Absolute laptop timings remain clock- and temperature-sensitive; candidate promotion still requires interleaved controls or same-run Flash-normalized ratios rather than a sub-percent comparison against this table alone.

## Predictor Evidence

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
- Restoring exact on-the-fly K64 dS maxima passes all six exact-4096 aggregates and all 60 heads, but loses every timing row. Averaging the two exact-scale runs around the predicted control gives `31.52-34.12` end-to-end and `33.16-34.91` kernel-only TFLOPS, versus `32.26-35.15` and `33.86-35.96` for prediction. Raw exact-scale medians regress `1.5-3.6%`. Exact scaling remains faster than same-run Flash by `1.063-1.084x` end to end and `1.092-1.140x` kernel-only, but the predictor reaches `1.090-1.134x` and `1.122-1.187x`.

The obsolete block-omission experiment is no longer part of the maintained data path.

## Measured Bottlenecks

The completed native-SM86 attribution remains the motivation for the next optimization. At sequence 8192, the prior matched capture found approximately `0.600x` Flash tensor-instruction activity but `3.332x` non-tensor instruction activity for Sage. Shared-store wavefronts were approximately `3.331x` Flash, while global dQ reduction sectors already matched Flash. The active kernel therefore remains limited primarily by work around the MMA instructions, not by the number of MMA instructions or by DRAM bandwidth.

### Reduction status

The exact on-the-fly `abs(dS)` maximum reduction is no longer present. The active kernel still pays for the local P scale reduction: scalar `fmaxf` updates build `p_max_abs`, followed by one warp-level `__reduce_max_sync` before P is quantized. This is a real non-tensor cost, but it is narrower than the removed dS maximum reduction and should be treated as one part of the P reconstruction path rather than as the main remaining dS-scale mechanism.

The Q32/K64 predictor summaries are computed in Triton with `O(N*D)` reductions. They do not create the native shared-store traffic. Replacing those reductions is lower priority unless an end-to-end phase breakdown shows that preprocessing, rather than the CUTLASS launch, has become material.

### P and dS reconstruction

Each K128 iteration reconstructs P and dS around the int8/fp16 MMA results. The path includes int32-to-fp32 conversion, score and softmax scaling, `expf`, LSE and Delta subtraction, dS formation, absolute-value/max tracking, and scale arithmetic. P and dS are then quantized with reciprocal-scale application and integer conversion. This combination explains a substantial part of the excess non-tensor instruction activity and includes the remaining P warp reduction.

### Shared-memory handoffs

The active path materializes and moves several fragments through shared memory:

- P and dS stores into the score-pair staging area.
- dS mirror stores into `dS_dKV` so the dQ path can consume the same reconstructed values.
- Packed Q/dO fragment stores and later loads for the dK/dV MMAs.
- Packed K fragment stores and later loads for the dQ MMAs.
- Shared dQ transpose/staging before the global atomic accumulation.
- dK/dV epilogue staging before coalesced global stores.

These stores and reloads are the leading candidates for the historical shared-store-wavefront excess. The global dQ reduction sectors already match Flash, so changing global dQ ownership without first reducing the shared staging is unlikely to address the measured gap.

### Scalar scale application and fragment plumbing

The dV, dK, and dQ accumulation paths apply FP32 products involving P, dS, dOutput, Q-block, K-block, and output scales around the integer MMA fragments. Manual packed-fragment indexing, address arithmetic, tail predicates, conversions, and repeated synchronization add further non-tensor instructions and scheduler stalls. These operations also constrain attempts to keep P/dS values in registers because producer lifetimes already approach the resource limit.

## Completed Work

This section records the decisions behind the current source and prevents closed experiments from being repeated without new evidence. An implementation defect invalidates only that candidate build and its measurements, not the underlying design. An experiment is closed only after the defect is fixed and the corrected candidate is validated, or when independent design-level accuracy or contract evidence rules out that specific formulation.

### Completed Audits

- Native SM86 attribution found approximately `0.600x` Flash tensor-instruction activity but `3.332x` non-tensor activity and `3.331x` shared-store wavefronts for Sage at sequence 8192. Global dQ reduction sectors already matched Flash, so shared handoffs and surrounding instruction work remain the measured gap.
- The resource audit reports 243 registers per thread for tail and aligned variants, zero stack/local spill, and 57,600 bytes of dynamic shared storage. `__maxnreg__(243)` is the explicit register boundary. The historical 57,632-byte NCU image still correctly establishes one resident block, 16.7% theoretical occupancy, and approximately 0.40 eligible warps per cycle; the final 32-byte cleanup does not change occupancy.
- Sequence-8192 profiling measured 53.49% SM/compute throughput, 3.10% DRAM throughput, 29.95% tensor-pipeline throughput, 54.56% L1 throughput, 22.30% L2 throughput, 3% excessive shared wavefronts, and 16.12% L1 sector utilization. Source-line attribution remains incomplete because the standalone source-page export produced no rows in the current Windows Nsight Compute build.
- The barrier audit established four dependency boundaries. Removing the startup K/V publication barrier produced 4,096 Racecheck hazards between asynchronous shared writes and cross-warp `LDSM` reads. The pre-dK/dV barrier publishes Q/dO packets and dS. The post-dK barrier before dQ is safely removed, and the end barrier protects next-pair Q/dO and row-state publication.
- The canonical dS audit established that dK already consumes the score-pair representation, while dQ needs one K32 MMA-A fragment assembled from two N16 producer slots. The current `LDSM.x4` atom cannot span the 1 KiB-separated slots under CuTe's vectorization contract, and the initial C-fragment/register shuffle mismatched all 32 lanes.
- The canonical two-source proof was corrected before promotion. The first standalone probe accidentally applied `make_transposed_tensor` to both the canonical score-pair destination and the mirror reference; production intentionally stores canonical dS transposed and the dQ mirror non-transposed. With the exact production orientations and source-distinct pseudorandom fragments, neither normal nor `.trans` `LDSM.x2` plus any uniform lane shift matched the MMA-A reference.
- Exhaustive low-instruction mapping checks closed the direct x2 formulation. None of 3,840 lane-bit-permutation/XOR address maps matched either M half. Byte-signature mapping showed that a normal load scatters each eight-byte target across eight source lanes, while a transposed load scatters it across four. A modeled hardware register transpose requires at least three `movmatrix`, two lane-permutation, and two local-byte-permutation stages per source fragment. That is not a competitive replacement for the mirror stores. The defective production candidate was reverted without timing; this closes only the tested gather formulation, not every possible mirror-elimination ownership design.
- The retained-image attribution gate was not triggered because no canonical-handoff candidate survived fragment correctness. Refresh event, instruction, shared-bank, synchronization, scheduler, and L1/shared counters after the next viable handoff rather than profiling a known-invalid image.
- After reverting the invalid gather candidate, the native SM86 extension was rebuilt from the retained source and the focused NHD/HND sequence-64/65 matrix passed all four cases. The production kernel has no worktree diff, so subsequent telemetry uses the retained control image.
- The long-shape telemetry and error-decomposition audit is complete. `data/evaluate_cutlass_bwd_captures.py` now reports aggregate and per-head gradient cosine, relative error, maximum absolute error, finite and strict-gate status, Q/K/dOutput/P/dS INT8 zero rates, P/dS endpoint saturation and clipping, P/dS scale percentiles, true-dS/predicted-dS percentiles, and fixed log2 histograms. It also compares Q/K reconstruction, exact-max dS, predicted dS, local P quantization, dOutput quantization, and the real INT8 kernel. Its default matrix covers both captures and 512/513/4096/8192 with explicit `slice`, `exact`, and `tiled` provenance labels.
- The packet and lifetime audit found no free compaction margin. Q/dO/K packet allocations exactly match their fragment counts; the K packet is `16 fragments * 4 words * 32 lanes * 4 bytes = 8,192 bytes`. Lowering shared storage alone cannot create a second resident block, so double buffering requires a proven dead region or a different ownership schedule.

### Retained Changes

- K128 main-kernel baseline. The active generated specialization is QBlock=32, KBlock=64, CTA M64xN128, eight warps. The former backward K64 CTA is removed from the current generation and dispatch, but alternate quantization blocks and attention CTA/warp shapes remain valid future search dimensions if measurements justify them.
- Triton utility phases. Delta/dOutput preprocessing, Q32/K64 predictor summaries, workspace allocation, and fp32 dQ-to-fp16 conversion moved out of the native wrapper and into autotuned Triton kernels. The native operator now launches the fused CUTLASS main kernel with explicit, typed workspaces.
- Unconditional separable dS prediction. The accepted predictor is `1.5 * (softmax_scale / N) * (max_row ||dO||2 * max_row ||V||2 + max_row |Delta|)`. It is `O(N*D)`, uses Q32/K64 summaries, and does not allocate an `O(N^2)` scale table or select a runtime policy.
- Quantization constants and conversion. The accepted reciprocal is `0x1.010122p-7`, the scale floor is `2^-126`, and C++ INT8 conversion is saturating. Triton dOutput quantization uses the shared unclamped `_round_to_int8` helper because its exact max-abs scale preserves the intended range.
- Guard selection. The fixed `1.5x` dS guard is accepted. A `1.25x` guard was faster on the six retained captures but was less robust on the sequence-65 correctness control; `2.0x` increased quantization error. The expanded scalar sweep confirms that no global guard or simple P-max multiplier clears both sigma-900 records, so guard tuning is closed without a new predictor structure.
- Training-derived validation data. Two independent SDXL-derived sequence-4096 captures with ten heads and sigma indices `300/700/900` are retained with reusable capture, simulation, benchmark, and evaluator scripts under `data/`. The full replay is finite, but exact sequence 4096 passes only four of six aggregate and 42 of 60 per-head strict gates. Mean dQ/dK relative error is `5.49%/4.42%`, worst error is `11.88%/19.06%`, and sparse dS clipping does not by itself predict those failures; this remains exploratory rather than strict-promotion quality.
- Validation baseline. The active build passes the four-case focused matrix, native-only Racecheck and Synccheck for NHD/HND at 512/513, the four-row new-contract timing baseline, SASS/resource inspection, aggregate NCU capture, `pre-commit`, compilation, and whitespace checks.
- Scale/index hoist candidate. The active kernel now caches the CTA-invariant K-domain index, reuses the Q-block index for predictor metadata, caches the dK scale product and dOutput scale-index base, and specializes K-scale lifetime so only aligned launches retain the CTA-invariant scale across M-pairs. With `-lineinfo` on native SM86, aligned static SASS falls from 2096 to 2088 instructions while the tail function is instruction-identical to baseline; registers remain 243 aligned and 227 tail with zero stack/local memory. On identical NHD/HND inputs at sequence 513/4096, dK/dV are bitwise identical and dQ differs only in 7-314 FP16 elements by at most `6.10e-5` with relative L2 below `1e-5`. Kernel-only timing moved by less than the laptop GPU clock variation in the paired candidate/baseline runs, so this is retained as a low-risk control rather than a promotion-quality speedup claim.
- Post-dK barrier removal. The synchronization point between dK/dV accumulation and dQ staging is removed. Native SM86 SASS now has four static CTA barriers instead of five; relative to the five-barrier scale/index-hoist build, tail SASS is seven instructions shorter and aligned SASS removes one `BAR.SYNC.DEFER_BLOCKING` while adding one `NOP`. `__maxnreg__(243)` preserves the 227-register tail and 243-register aligned allocations with zero stack/local spill. Racecheck and Synccheck pass NHD/HND at sequence lengths 512 and 513. Paired kernel-only timing at 4096/8192 was mixed and within laptop clock/thermal drift, so the cleanup is retained without a performance-win claim.
- Packed-K dQ reuse. The dQ phase now loads each packed K fragment once per even warp and consumes it for both M16 halves, rather than reloading it for each half. It also computes each K-domain predictor scale once per pair. Native SM86 SASS falls by 16 `LDS` instructions and 22/13 total instructions in tail/aligned variants; the static CTA barrier count remains four in the retained image. Both variants use 243 registers with zero stack/local spill. Racecheck and Synccheck pass NHD/HND at 512/513. Paired new-contract timing is clock-sensitive in absolute milliseconds, but same-run Flash-normalized ratios improved by approximately `0.7-1.2%` for NHD and `0.7-3.2%` for HND across 4096/8192 kernel-only and end-to-end rows, so the change is retained.
- Single K-scale load site. Both aligned and tail specializations now call `load_k_block_scale()` at one source location inside the M-pair loop. Resources remain 243 registers with zero stack/local spill and tail SASS is unchanged; ptxas emits eight additional aligned instructions compared with the manually split lifetime. Short paired timing was mixed, so this is retained as source simplification without a speedup claim. Revisit only after the larger shared-memory handoffs are resolved.
- Dead shared P-scale publication removal. `SharedStorage2DWarp::p_scale` was an unreferenced trailing eight-float array; the active local `p_scale` arithmetic is unchanged. Removing the field reduces shared storage from 57,632 to 57,600 bytes. Tail and aligned native-SM86 instruction streams are exactly identical to the control, resources remain 243 registers with zero stack/local spill, and the focused four-case matrix passes. This is retained as a resource cleanup without a latency claim.

### Closed Experiments

- Fixed global P scaling. Rejected because it destroys long-sequence dV signal.
- Exact Q32/K128 single-domain geometry. Rejected after regressing the governing long-sequence controls.
- Mirror-only dS-to-dK handoff. Rejected after timing regression and increased resource pressure.
- Power-of-two dS scaling. Rejected as a maintained implementation. It provided only approximately `0.6-1.0%` gains and its specialization/dispatch was removed.
- Periodic dS scaling. Rejected on accuracy despite approximately `4.3-5.1%` main-kernel gains; its specialization/dispatch was removed.
- Sparse dS. Rejected as the primary path. Oracle selection needed approximately `90%` block retention and still reached worst dQ/dK relative errors of `11.39%/21.39%`.
- True pre-forward prediction. Rejected because same-timestep dOutput RMS differed by approximately `28x`; same-RMS random dOutput produced p99 scale ratios up to `31.5x` and a maximum around `642x`.
- Runtime dS-policy switching. Rejected in favor of one unconditional predictor; dynamic, power-of-two, and periodic IDs and the `PREDICT_DS_SCALE` toggle are no longer part of the active API or source.
- Exact on-the-fly K64 dS scaling as an unchanged default replacement. The current-source control restored one `abs(dS)` warp reduction, eight shared scale slots, and one CTA barrier per M32 pair. It passed focused correctness, all 60 exact-4096 per-head strict gates, and the full 512/513 NHD/HND Racecheck/Synccheck matrix. It remained spill-free at 243/231 tail/aligned registers and 57,632 bytes shared; aligned SASS shortened from 2,088 to 2,064 instructions, but barriers rose from four to five and `REDUX.MAX` from two to three. Candidate/control/candidate timing regressed every 4096/8192 NHD/HND kernel-only and end-to-end row by `1.5-3.6%`, so the unchanged synchronization protocol is closed as a performance candidate despite its accuracy value.
- The tested broad Q/dO packetization, broad vector-packet, register-V, direct/transposed dQ staging, and split dV/dK formulations are closed because their validated builds were slower or resource-heavy. Revisit the broader ideas only with a materially different layout or ownership hypothesis and source/SASS evidence that targets a measured handoff cost.
- Dimension-owned dK/dV without Q/dO packets. The isolated implementation assigned four dK dimension owners and four dV dimension owners, captured one direct Q or dO operand per owner, and traversed all eight N16 score slots. dQ used four private 1 KiB regions in dead resident-V storage. Racecheck exposed an ordering bug in the first implementation: direct operand loads after the publication barrier could overlap the next-pair asynchronous Q/dO prefetch. That run was discarded. Capturing the fragments before the barrier and deferring prefetch until after dQ fixed the race; the corrected build then passed Racecheck and Synccheck for NHD/HND at 512/513 and focused correctness, with sequence-4096 differences below `6.10e-5` maximum and `1e-5` relative L2. Only then was the formulation closed because it used a 96-byte aligned stack frame and regressed serial 4096/8192 kernel-only and end-to-end controls by approximately `15-21%`.
- Plain 32-byte-row dS/dKV scratch. Rejected despite reducing each warp scratch slot from 1,024 to 512 bytes. A dedicated fragment probe showed zero mismatches between the existing C-fragment store/MMA-A load path and the compact layout, and the reused `16x16` FP16 dK/dV epilogue stage fit exactly. The candidate reduced dynamic shared memory from 57,632 to 53,536 bytes, tail registers from 243 to 223, and static SASS by 8/9 instructions in tail/aligned variants, with no stack/local spill. Focused tests and the full 512/513 NHD/HND Racecheck/Synccheck matrix passed; dK/dV were bitwise identical at sequence 4096 and dQ differed only by `3.05e-5` maximum with `7.43e-6` relative L2 from cross-run atomic ordering. Candidate/control/candidate timing tied NHD 4096 but regressed HND 4096 by approximately `1.7-3.2%` and the 8192 rows by approximately `1.1-3.7%`. Matched sequence-4096 NCU counters explained the loss: shared-load bank conflicts rose from `2,097,152` to `10,485,760`, store conflicts from `542,057` to `4,510,420`, and load/store wavefronts by approximately `11.1%/14.1%`. Keep the swizzled layout unless a compact replacement also proves its bank behavior.
- Pre-dK/dV publication-barrier removal. Rejected after an isolated same-order run. Native Racecheck reported zero hazards for NHD/HND at 512/513, but aligned registers increased from 243 to 249. NHD was effectively tied, while HND regressed from 16.1038 to 17.2969 ms at sequence 8192 and from 3.9695 to 4.2977 ms at sequence 4096. This is the barrier that publishes packed Q/dO and mirrored dS, and it remains in the active kernel; it is distinct from the removed post-dK barrier before dQ.
- Direct two-source canonical dS gather. Rejected as a low-instruction mirror replacement after correcting the standalone oracle. The invalid production candidate loaded canonical transposed dS as though it had the mirror's non-transposed orientation, causing all four focused dQ cases to fail while dK/dV remained unchanged. The corrected source-distinct oracle, all uniform lane shifts, normal/transposed x2 atoms, and all lane-bit-permutation/XOR address maps established that an exact fragment needs a substantial cross-lane byte transpose. No candidate timing or resource result from the invalid build is retained.

## Next Work

The next cycle remains backward-only and uses QBlock=32, KBlock=64, CTA M64xN128, and eight warps as the measured reference. Preserve the score/softmax arithmetic order, unconditional `1.5x` predictor guard, INT8 conversion policy, accumulation types, and four proven dependency barriers unless an isolated design explicitly changes their ownership contract.

### Immediate sequence

- Keep the fixed predictor as the performance reference and exact-max reconstruction as the accuracy oracle. Do not retry a scalar guard or simple P-max multiplier; any new predictor must approach the exact-max stage's 60-of-60 per-head result without adding the rejected per-M32 CTA barrier.
- Resume direct `rP`/`rdS` producer-register handoff only after a standalone producer-to-MMA-A fragment proof. The implementation must remove canonical P/dS stores and reloads while preserving the retained dS mirror for dQ, stay spill-free, and keep the four-barrier schedule. The rejected canonical gather and ordinary mirror-only dK load are not fallbacks.
- Refresh retained-image attribution only after that handoff or another telemetry-driven candidate survives correctness and short timing. Capture event time, executed instructions, registers, shared load/store wavefronts and conflicts, synchronization stalls, eligible warps, and L1/shared throughput. Long-format NCU CSV parsing must skip the two `==PROF==` preamble lines and pivot by `Metric Name`. The exact-scale control was not profiled because governing event timing already rejected it.

### Deferred

- Define forward-owned Q/K artifacts independently of kernel geometry. The current forward emits per-thread metadata while backward consumes Q32/K64 per-block tensors. Introduce separate quantization, forward-attention, and backward-attention configuration fields; retain an internal-quantization compatibility path and describe benchmark preparation only as a backward-format reuse proxy until real artifacts are shared.
- Revisit quantization blocks or CTA/warp geometry only after handoff attribution and telemetry identify a concrete limitation. Generate all shapes through `scripts/generate_cutlass_bwd_instantiations.py` and collect accuracy, resources, occupancy, shared counters, sanitizer results, and both timing modes for every candidate.
- INT4 remains deferred until the INT8 path has a stronger accuracy margin and a demonstrated end-to-end need.

### Closed constraints

- Keep the four current CTA barriers. The startup K/V publication barrier is Racecheck-proven, the pre-dK/dV barrier publishes Q/dO packets and dS, and the end barrier protects next-pair publication. Do not use barriers as ptxas register-allocation boundaries.
- Do not treat packet compaction as free storage. Active Q/dO/K packet allocations are exact for their fragment counts; double buffering needs a proven dead-lifetime alias or a new schedule.
- Do not retry the plain 32-byte-row dS scratch without a bank-aware layout: it saved 4 KiB and 20 tail registers but increased shared conflicts and regressed timing.
- Do not retry the eight-warp dimension-owned Q/dO-elimination schedule without a way to avoid all-slot score reloads and the aligned stack frame.
- Treat broad packetization, register-V, direct/transposed dQ, early-dO, split dV/dK, periodic/power-of-two dS, pre-dK barrier removal, and unchanged exact-max K64 dS synchronization as closed controls unless new source/NCU evidence changes their cost model.

Every promising candidate must pass finite-output and accuracy gates, native Racecheck/Synccheck for aligned and tail NHD/HND inputs, SASS/resource inspection, and interleaved kernel-only/end-to-end timing at 4096/8192 in both layouts. A compile, mapping, correctness, or sanitizer failure triggers diagnosis and a corrected rerun; it is not performance evidence. Higher register or shared-memory use is acceptable only when overall measured speed justifies it; stack/local spill in an otherwise correct build remains a rejection signal. The Triton reduction tree remains lower priority because it is `O(N*D)` and outside the native shared-store bottleneck.

## Standard Commands

Build with four jobs:

```powershell
$Env:NVCC_APPEND_FLAGS = '-lineinfo'
$Env:MAX_JOBS = '4'
python setup.py build_ext --inplace
```

Focused correctness:

```powershell
python -m pytest -q tests/test_sagebwd_cutlass.py
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
python bench/bench_sagebwd_cutlass.py `
  --batch-size 1 --num-heads 16 --head-dims 64 `
  --seq-lens 4096 8192 --layout NHD `
  --warmup 25 --repeats 100 `
  --block-configs 32,64,64,128 --mode all
```

Profile one active main-kernel launch:

```powershell
python build/profile_sagebwd_once.py sage `
  --head-dim 64 --seq-len 512 --num-heads 16 --warmup 1 `
  --block-config 32,64,64,128
```

Native-only Nsight Compute capture:

```powershell
ncu --set full --profile-from-start off `
  --target-processes application-only `
  --import-source on --source-folders C:\sageattention-autotune `
  --kernel-name 'regex:fused_mma_kernel_k128_8warp' `
  --launch-count 1 --export build/ncu_sagebwd_final_native_8192_source.ncu-rep `
  python data/racecheck_cutlass_bwd_kernel_only.py `
  --seq-len 8192 --num-heads 16 --layout NHD
```

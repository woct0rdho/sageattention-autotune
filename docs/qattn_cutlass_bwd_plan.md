# QAttn CUTLASS Backward Kernel Plan

## Current State

The maintained backward path targets native SM86, fp16, non-causal fixed-length attention with head dimension 64 and NHD/HND layouts. The only generated backward specialization is:

```text
QBlock=32, KBlock=64, CTA M64xN128, eight warps
```

`_BWD_CONFIG` remains `(32, 64, 64, 128)`. The old CTA-N=64 geometry and all runtime dS-policy IDs have been removed. The active kernel always uses one pre-backward separable dS-scale predictor; the maintained CUTLASS path has no runtime format toggle.

The old filename/module `sm80` is a legacy identifier. `setup.py` emits only `compute_86,sm_86`. The installed FlashAttention 2.9.1 wheel remains the reference implementation; a targeted SM80/SM86 source probe showed no material architecture-specific advantage for its relevant specialization.

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

These are warmed serial 100-repeat CUDA-event medians on an NVIDIA GeForce RTX 3080 Ti Laptop GPU, batch 1, 16 heads, head dimension 64. The active configuration is the only Sage candidate. `kernel_only` includes the CUTLASS launch and Triton dQ conversion after prequantized Q/K and precomputed workspaces. The historical `end_to_end` rows below include Q/K quantization under the old benchmark contract; they are retained for historical comparison only and must be regenerated. Under the new contract, `end_to_end` reuses prequantized backward-format Q/K and includes preprocessing, workspace/output allocation, the main kernel, and dQ conversion, but not Q/K quantization or the forward pass.

| Order | Layout | Seq | Flash ms | Sage end-to-end ms | Sage kernel-only ms | End-to-end speed ratio | Kernel-only speed ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| forward | NHD | 4096 | 4.5818 | 4.5173 | 4.2742 | 1.014 | 1.072 |
| forward | NHD | 8192 | 17.3266 | 16.3051 | 15.9585 | 1.063 | 1.086 |
| forward | HND | 4096 | 4.7908 | 4.5393 | 4.3259 | 1.055 | 1.107 |
| forward | HND | 8192 | 17.7172 | 16.4588 | 15.9811 | 1.076 | 1.109 |
| reverse | NHD | 8192 | 17.7162 | 16.8566 | 16.0736 | 1.051 | 1.102 |
| reverse | NHD | 4096 | 4.5680 | 4.4851 | 4.3012 | 1.018 | 1.062 |
| reverse | HND | 8192 | 17.9876 | 17.0808 | 16.1326 | 1.053 | 1.115 |
| reverse | HND | 4096 | 4.5993 | 4.5082 | 4.3510 | 1.020 | 1.057 |

The historical active path beats the matched Flash backward control in all eight timing rows. Those end-to-end values include Q/K quantization and are not the replacement baseline for forward-owned reuse. Regenerate the table with the new benchmark contract before using end-to-end ratios for promotion decisions; forward timing remains intentionally outside this plan.

Timing artifacts:

- `build/bench_sagebwd_final_nhd_4096_8192.csv`
- `build/bench_sagebwd_final_hnd_4096_8192.csv`
- `build/bench_sagebwd_final_nhd_reverse_8192_4096.csv`
- `build/bench_sagebwd_final_hnd_reverse_8192_4096.csv`

## Predictor Evidence

The retained SDXL-derived captures remain under `data/`:

- `data/sdxl_periodic_ds_inputs_training_dout_latent128.pt`
- `data/sdxl_periodic_ds_inputs_training_dout_latent128_seed2.pt`

The experiments established:

- A true pre-forward predictor is not viable. Same-timestep dOutput RMS differed by approximately `28x`; same-RMS random dOutput still changed block scales by p99 ratios up to `31.5x` and a maximum around `642x`.
- The pre-backward separable predictor is the useful asymptotic class. In the simulator, using dequantized Q/K and exact-dS references to isolate scale-policy error, the base formula measured mean/min dQ cosine `0.996286/0.993382`, dK `0.998667/0.994683`, mean/max dQ relative error `7.73%/11.50%`, and dK `4.10%/10.31%` on both captures and all ten heads.
- A dOutput-max-derived approximation measured mean/min dQ cosine `0.996062/0.992853`, dK `0.998681/0.994846`, mean/max dQ relative error `8.00%/11.95%`, and dK `4.10%/10.15%`; peak saturation was below `64 ppm` in that experiment.
- The active safety factor is `1.5`. The current six-record CUTLASS replay is finite; mean dQ/dK relative error is `5.49%/4.42%` and worst relative error is `11.88%/19.06%`. This remains exploratory rather than strict-promotion quality: the worst dQ relative error exceeds the `10%` training target, and the evaluator does not yet record saturation, clipping, zero rates, scale histograms, or optimizer-step behavior.

The retained experiment outputs are:

- `data/ds_scale_precompute_scales.csv`
- `data/ds_scale_precompute_ratios.csv`

The old block-omission experiment and its generated CSV have been removed from the maintained data path.

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

### Resource and profiling constraints

The final `cuobjdump` SM86 resource report shows 243 registers per thread for both the tail and even-length variants, with zero stack and local-memory bytes. The kernel uses `__maxnreg__(243)` instead of retaining a synchronization barrier as a ptxas allocation boundary; removing the cap raised the barrier-removed tail specialization to 255 registers. The packed-K dQ reuse now reaches the explicit 243-register budget in the tail variant without a spill. `sizeof(SharedStorage2DWarp<BwdTileTraits<64, 64, 128, 8>>)` is 57,632 bytes, down 32 bytes after removing the obsolete shared dS-scale array. The final Nsight Compute capture reports 57.63 KiB dynamic shared memory, one resident block, 16.7% theoretical occupancy, and only 0.40 eligible warps per cycle on average. Register-spill traffic is zero. Resource increases are not rejected in isolation, but must be justified by measured kernel or end-to-end speed.

At sequence 8192, NCU reports 53.49% SM/compute throughput, 3.10% DRAM throughput, 29.95% instruction/tensor-pipeline throughput, 54.56% L1 throughput, and 22.30% L2 throughput. It flags 3% excessive shared-memory wavefronts and low L1 sector utilization at 16.12%. The wavefront ratio and the NCU excessive-wavefront percentage are different metrics; together with the source-level gap in non-tensor activity, they identify shared staging and instruction scheduling as the next measurement targets rather than proof that one particular store site is solely responsible.

Saved reports:

- `build/ncu_sagebwd_final_native_4096.ncu-rep`
- `build/ncu_sagebwd_final_native_8192.ncu-rep`
- `build/ncu_sagebwd_final_native_8192_source.ncu-rep`
- `build/ncu_sagebwd_final_native_8192_source_details.txt`
- `build/ncu_sagebwd_final_native_4096.csv`
- `build/ncu_sagebwd_final_native_8192.csv`
- `build/cuobjdump_final_resources.txt`
- `build/cuobjdump_final_sass.txt`

The aggregate NCU details are complete. The standalone source-page export produced no rows in the current Windows Nsight Compute build, so source-line attribution remains unverified even though the source-imported report is retained.

### Newly confirmed optimization ranking

The current source-level dependency audit changes the order of the next optimization cycle. The dominant opportunity remains non-tensor work around shared-memory handoffs, not additional MMA instructions or global dQ reduction.

- Treat the current four CTA barriers as dependency boundaries, not scheduling controls. The startup barrier publishes the cooperative K/V load and is required: removing it produced 4,096 Racecheck hazards between the asynchronous shared write at SASS offset `+0x740` and cross-warp `LDSM` reads around `+0x1960` through `+0x1a10`. The pre-dK/dV barrier publishes packed Q/dO and mirrored dS to their cross-warp consumers. The post-dK barrier before dQ has been removed because each dQ producer overwrites only its owned even-warp score slot; Racecheck and Synccheck are clean for aligned and odd-tail NHD/HND cases. Keep the end-of-iteration barrier that protects the next Q/dO and row-state publication.
- Test a canonical dS view. dS is written into the score-pair representation and copied again into `dS_dKV` for dQ. A CuTe view that lets dK and dQ consume the score-pair dS directly could remove mirror stores, reduce shared storage, and preserve the current numerical policy. The physical layout must be proven for both consumers; a generic single-domain rewrite is not sufficient.
- Narrow the packed handoffs. The packed Q/dO/K helpers move four scalar words per lane, and the final SASS contains scalar shared loads/stores plus byte and halfword P/dS stores. A lane-major layout with vector shared transactions is worth testing only if it reduces emitted instructions and shared wavefronts without adding repacking shuffles. Broad packetization, register-V retention, and direct/transposed dQ staging already regressed and are not first-line candidates.
- Consider resource-aware pipelining. The even-length variant uses 243 registers and 57.63 KiB dynamic shared memory, so lowering registers alone cannot create a second resident block. The active K packet allocation is larger than the four packed tiles it stores; an exact-size packet could free roughly 6 KiB for double-buffered staging and a producer/consumer pipeline. This is higher risk and should follow the barrier and canonical-dS experiments.
- Lower priority: scalar arithmetic and end-to-end workspace traffic. Hoist combined score and epilogue scale products only if SASS confirms fewer instructions. The compiler already emits fused score arithmetic and `MUFU.EX2`, so exp/scale approximations should not change the numerical policy. Loading both FP16 and INT8 dO remains a possible end-to-end preprocessing optimization, but the main kernel's approximately 3% DRAM utilization makes it unlikely to improve kernel-only time.

Candidate NCU CSVs are long-format metric rows after two `==PROF==` preamble lines; future aggregation must strip those lines and pivot by `Metric Name`. The instrumented native sequence-8192 capture is approximately 20.61 ms and must not be compared directly with normal benchmark timings near 16 ms.

## Completed Work

This is the experiment log behind the current design. Historical measurements remain here when they explain why an alternative is closed; only the accepted choices are active code. Historical controls, captures, and evaluator filenames may retain their original experiment names, but they are data artifacts only and are not dispatchable production policies.

### Accepted

- Native SM86 attribution. The governing gap is non-tensor work and shared-memory traffic rather than MMA count or DRAM bandwidth: the matched historical comparison measured approximately `0.600x` Flash tensor-instruction activity, `3.332x` non-tensor activity, and `3.331x` shared-store wavefronts for Sage, while global dQ reduction sectors already matched Flash. CTA M64xN128 with eight warps and Q32/K64 quantization blocks remains the reference baseline, not a permanent optimization constraint.
- K128 main-kernel baseline. The active generated specialization is QBlock=32, KBlock=64, CTA M64xN128, eight warps. The former backward K64 CTA is removed from the current generation and dispatch, but alternate quantization blocks and attention CTA/warp shapes remain valid future search dimensions if measurements justify them.
- Triton utility phases. Delta/dOutput preprocessing, Q32/K64 predictor summaries, workspace allocation, and fp32 dQ-to-fp16 conversion moved out of the native wrapper and into autotuned Triton kernels. The native operator now launches the fused CUTLASS main kernel with explicit, typed workspaces.
- Unconditional separable dS prediction. The accepted predictor is `1.5 * (softmax_scale / N) * (max_row ||dO||2 * max_row ||V||2 + max_row |Delta|)`. It is `O(N*D)`, uses Q32/K64 summaries, and does not allocate an `O(N^2)` scale table or select a runtime policy.
- Quantization constants and conversion. The accepted reciprocal is `0x1.010122p-7`, the scale floor is `2^-126`, and C++ INT8 conversion is saturating. Triton dOutput quantization uses the shared unclamped `_round_to_int8` helper because its exact max-abs scale preserves the intended range.
- Guard selection. The fixed `1.5x` dS guard is accepted. A `1.25x` guard was faster on the six retained captures but was less robust on the sequence-65 correctness control; `2.0x` increased quantization error.
- Training-derived validation data. Two independent SDXL-derived sequence-4096 captures with ten heads and sigma indices `300/700/900` were retained, with reusable capture, simulation, benchmark, and evaluator scripts under `data/`. The current CUTLASS replay is finite, with mean dQ/dK relative error `5.49%/4.42%` and worst error `11.88%/19.06%`; this remains exploratory rather than strict-promotion quality.
- Validation baseline. The active build passes the four-case focused matrix, native-only Racecheck and Synccheck for NHD/HND at 512/513, all eight paired backward timing rows, SASS/resource inspection, aggregate NCU capture, `pre-commit`, compilation, and whitespace checks.
- Scale/index hoist candidate. The active kernel now caches the CTA-invariant K-domain index, reuses the Q-block index for predictor metadata, caches the dK scale product and dOutput scale-index base, and specializes K-scale lifetime so only aligned launches retain the CTA-invariant scale across M-pairs. With `-lineinfo` on native SM86, aligned static SASS falls from 2096 to 2088 instructions while the tail function is instruction-identical to baseline; registers remain 243 aligned and 227 tail with zero stack/local memory. On identical NHD/HND inputs at sequence 513/4096, dK/dV are bitwise identical and dQ differs only in 7-314 FP16 elements by at most `6.10e-5` with relative L2 below `1e-5`. Kernel-only timing moved by less than the laptop GPU clock variation in the paired candidate/baseline runs, so this is retained as a low-risk control rather than a promotion-quality speedup claim.
- Post-dK barrier removal. The synchronization point between dK/dV accumulation and dQ staging is removed. Native SM86 SASS now has four static CTA barriers instead of five; relative to the five-barrier scale/index-hoist build, tail SASS is seven instructions shorter and aligned SASS removes one `BAR.SYNC.DEFER_BLOCKING` while adding one `NOP`. `__maxnreg__(243)` preserves the 227-register tail and 243-register aligned allocations with zero stack/local spill. Racecheck and Synccheck pass NHD/HND at sequence lengths 512 and 513. Paired kernel-only timing at 4096/8192 was mixed and within laptop clock/thermal drift, so the cleanup is retained without a performance-win claim.
- Packed-K dQ reuse. The dQ phase now loads each packed K fragment once per even warp and consumes it for both M16 halves, rather than reloading it for each half. It also computes each K-domain predictor scale once per pair. Native SM86 SASS falls by 16 `LDS` instructions and 22/13 total instructions in tail/aligned variants; the static CTA barrier count remains four in the retained image. Both variants use 243 registers with zero stack/local spill. Racecheck and Synccheck pass NHD/HND at 512/513. Paired new-contract timing is clock-sensitive in absolute milliseconds, but same-run Flash-normalized ratios improved by approximately `0.7-1.2%` for NHD and `0.7-3.2%` for HND across 4096/8192 kernel-only and end-to-end rows, so the change is retained.

### Rejected Or Deferred

- Fixed global P scaling. Rejected because it destroys long-sequence dV signal.
- Exact Q32/K128 single-domain geometry. Rejected after regressing the governing long-sequence controls.
- Mirror-only dS-to-dK handoff. Rejected after timing regression and increased resource pressure.
- Power-of-two dS scaling. Rejected as a maintained implementation. It provided only approximately `0.6-1.0%` gains and its specialization/dispatch was removed.
- Periodic dS scaling. Rejected on accuracy despite approximately `4.3-5.1%` main-kernel gains; its specialization/dispatch was removed.
- Sparse dS. Rejected as the primary path. Oracle selection needed approximately `90%` block retention and still reached worst dQ/dK relative errors of `11.39%/21.39%`.
- True pre-forward prediction. Rejected because same-timestep dOutput RMS differed by approximately `28x`; same-RMS random dOutput produced p99 scale ratios up to `31.5x` and a maximum around `642x`.
- Runtime dS-policy switching. Rejected in favor of one unconditional predictor; dynamic, power-of-two, and periodic IDs and the `PREDICT_DS_SCALE` toggle are no longer part of the active API or source.
- INT4. Deferred until the INT8 path has a stronger accuracy margin and a demonstrated end-to-end need.
- Direct `rP`/`rdS` producer-register handoff and geometry changes. Deferred until the long accuracy telemetry and source-level attribution gaps are closed, not rejected. Future searches may change Q/K quantization block sizes, CTA M/N tile sizes, K-side attention tile sizes, warp count, and warp arrangement. Every new shape needs a separate lifetime/resource design and full backward validation.
- Broad Q/dO packetization, broad vector packets, register-V retention, direct/transposed dQ staging, and split dV/dK variants. These were slower or resource-heavy in prior measurements. Revisit only with a narrower layout or ownership hypothesis and source/SASS evidence that targets a measured handoff cost.
- Pre-dK/dV publication-barrier removal. Rejected after an isolated same-order run. Native Racecheck reported zero hazards for NHD/HND at 512/513, but aligned registers increased from 243 to 249. NHD was effectively tied, while HND regressed from 16.1038 to 17.2969 ms at sequence 8192 and from 3.9695 to 4.2977 ms at sequence 4096. This is the barrier that publishes packed Q/dO and mirrored dS, and it remains in the active kernel; it is distinct from the removed post-dK barrier before dQ.

## Next Work

The next optimization cycle remains backward-only. Use QBlock=32, KBlock=64, CTA M64xN128, and eight warps as the measured baseline, but do not treat those dimensions as permanent constraints. Synchronization and representation changes should preserve the current predictor and numerical format. Each candidate must be measured in paired, same-build, same-order runs at sequence 4096 and 8192 in NHD and HND with both kernel-only and end-to-end backward timings. End-to-end backward reuses prequantized Q/K and includes Triton preprocessing, the CUTLASS launch, dQ conversion, and output/workspace allocation; it excludes Q/K quantization and the forward pass.

### Priority order

- Keep the four validated CTA barriers unless a new ownership or staging design changes their dependencies. Do not retain or insert barriers to influence register allocation; use an explicit register budget and require zero stack/local spill.
- Capture future shared-memory candidates at sequence 8192. Compare event time, executed instructions, registers, shared load/store wavefronts and conflicts, synchronization stalls, eligible warps, and L1/shared throughput.
- Test a canonical dS score-pair view without changing the predictor, INT8 scales, or accumulation format. Require exact layout/ownership reasoning for dK and dQ, then run the full correctness matrix before timing. The packed-K dQ handoff is already retained and should be the comparison baseline.
- Test vectorized packed Q/dO/K and P/dS handoffs only when generated SASS shows the intended vector transactions. Reject versions that trade shared stores for shuffles, bank conflicts, or higher register pressure.
- Investigate exact-size K packet storage and double-buffered producer/consumer staging only if the first four steps identify barrier or handoff latency as the limiting cost.
- Keep the scale/index-hoist candidate as the lower-priority arithmetic control and measure any follow-up dO workspace reuse end to end. Further scale changes require emitted-SASS evidence and must preserve the fixed dS guard and FP8/FP4 exclusion.
- Reconsider quantization blocks or CTA/warp geometry only after the reference handoffs are understood; the tested CTA-N=64, packet, register-V, and dQ transpose alternatives do not currently justify promotion.

- Complete long-shape telemetry before changing the dS representation, predictor, or quantization policy. Synchronization-only candidates may proceed after Racecheck because they should be numerically exact. Extend `data/evaluate_cutlass_bwd_captures.py` to report P and dS saturation/clipping, INT8 zero rates, scale histograms and percentiles, finite-output status, cosine, relative error, maximum absolute error, and strict-gate status for both retained captures, all heads, and 4096/8192 plus 512/513 tail controls. This establishes whether the predictor guard or quantization margin is constraining any later handoff or geometry change.
- Replace the missing source-line NCU view with a usable attribution report. Use line-info/source-counter extraction or an equivalent controlled build to separate P/dS reconstruction and quantization, packed Q/dO/K staging, dQ shared transpose, dK/dV staging, scale application, synchronization, and tail predicates. Report instruction classes, shared load/store wavefronts, L1 sector utilization, eligible warps, and register/shared-memory resources for each controlled variant.
- Establish isolated baselines for the remaining shared-memory stages. Keep the algorithm and the reference geometry fixed while measuring the P/dS canonical handoff, packed MMA-B stores/loads, dQ shared transpose, and dK/dV staging. A variant is useful only if paired event timing and source counters show a reduction in the targeted synchronization or traffic cost without moving work to another staging path.
- Investigate P/dS producer-consumer lifetime and scalar scale costs. After steps 1 and 2, test whether `rP`/`rdS` values or their scale products can be consumed directly by the next MMA path, partially reused, or staged in a cheaper representation. Also evaluate hoisting or reusing repeated FP32 scale products, combining scale application with fragment conversion where the generated SASS improves, and simplifying packed-fragment address arithmetic. Preserve the local P scale semantics and unconditional predicted dS scale. Reject any version that spills registers, loses Racecheck cleanliness, or regresses the overall kernel/end-to-end speed target; higher register/shared-memory use is acceptable when the timing improvement is real.
- Define forward-owned Q/K reuse as a separate data contract. The current CUTLASS forward path emits per-thread Q/K quantization metadata, while backward consumes the Q32/K64 per-block format. Future integration must either make the forward artifacts consumable by backward or retain a separately specified backward-compatible representation. Keep Q/K quantization block sizes, forward attention tile/warp configuration, and backward attention tile configuration as independent fields; do not encode them as one `BlockConfig` or infer one from another. The backward benchmark currently uses the backward-compatible format as a timing proxy, not as proof of true forward reuse.
- Sweep quantization block sizes when the telemetry identifies a scale or handoff limitation. Compare alternatives for QBlock and KBlock, including the current Q32/K64 baseline and larger or asymmetric domains where the workspace and predictor contract can support them. Measure predictor accuracy, scale distributions, P/dS saturation, shared traffic, register lifetime, and both backward timing measures. Do not assume that a larger quantization block is beneficial; it can reduce metadata and handoffs while increasing quantization error or register pressure.
- Sweep attention CTA and warp shapes when the source attribution points to tile ownership or scheduler imbalance. Compare CTA M/N tile sizes, K-side attention tile sizes, warp count, and warp arrangement, including whether a CTA-N=64-style shape or another N tile earns reconsideration. Regenerate declarations and dispatch through `scripts/generate_cutlass_bwd_instantiations.py`; do not hand-edit generated files. For each shape, collect SASS/resource data, occupancy, eligible warps, shared-store wavefronts, L1 utilization, Racecheck, long-tail accuracy, and backward timing.
- Revisit shared layouts only after the isolated handoff or geometry results identify a winning stage. Test dQ transpose and packed MMA-B layout changes against shared-store wavefronts, L1 sector utilization, register count, and kernel timing. The current 57.63 KiB shared-memory footprint and 243-register even-length variant are baselines for resource comparisons, not hard limits, but any increase must produce a clear backward end-to-end win with no spill.
- Re-run the complete backward validation matrix after each promising candidate. Require finite outputs, the strict default cosine/relative-error/maximum-absolute-error gates where applicable, the diffusion-training direction and relative-error targets, native Racecheck for aligned and tail NHD/HND cases, clean SASS/resource inspection, and paired same-build, same-order kernel-only/end-to-end backward timing at 4096/8192. Keep the current eight-row backward table as the performance baseline.

The idiomatic Triton reduction tree remains a lower-priority path. It is `O(N*D)`, outside the native shared-store problem, and should change only if the backward-only phase breakdown proves preprocessing dominates end-to-end time. The removed runtime policy switch and sparse path remain out of scope; geometry and quantization block exploration are allowed through generated, validated specializations.

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

## Key Artifacts

- Kernel: `csrc/qattn_cutlass/qk_int8_sv_f16_bwd_kernel_cutlass_sm80.cuh`
- Launch: `csrc/qattn_cutlass/qk_int8_sv_f16_bwd_launch_cutlass_sm80.cuh`
- Generated dispatch: `csrc/qattn_cutlass/generated/qk_int8_sv_f16_accum_f32_attn_bwd_cutlass_dispatch.cuh`
- Generator: `scripts/generate_cutlass_bwd_instantiations.py`
- Python wrapper: `sageattention/cutlass_bwd.py`
- Triton utility kernels: `sageattention/triton/cutlass_bwd.py`
- Triton quantization helpers: `sageattention/triton/quant_per_block.py`, `sageattention/triton/quant_per_thread.py`
- Benchmark: `bench/bench_sagebwd_cutlass.py`
- Profile helper: `build/profile_sagebwd_once.py`
- Native-only Racecheck and NCU helper: `data/racecheck_cutlass_bwd_kernel_only.py`
- Focused tests: `tests/test_sagebwd_cutlass.py`
- Capture helper: `data/capture_sdxl_training_dout.py`
- Capture accuracy evaluator: `data/evaluate_cutlass_bwd_captures.py`
- Predictor experiment: `data/experiment_ds_scale_precompute.py`
- Predictor benchmark: `data/benchmark_ds_scale_precompute.py`

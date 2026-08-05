# QAttn CUTLASS Backward Kernel Plan

## Status Snapshot

The correctness-first CuTe migration is complete. The current retained baseline is the public head-64 Q32/K64 `64x64x32x64` configuration, which selects an internal eight-warp `2 M-halves x 4 N-microtiles` kernel. It remains the source-level and long-shape A/B reference, but it is no longer the optimization destination. The primary program is now a clean head-64 `M64 x N128`, eight-warp backward redesign that attacks repeated on-chip movement, scalar materialization, synchronization, and dQ reduction together. Isolated small-margin K64 tuning is deferred until the structural redesign space is exhausted.

The latest focused kernel combines block-scaled Q/K, one logical dS quantization, CTA-local dQ fusion, compile-time quantization geometry, Ampere hardware max reductions, in-loop Q/dO tail prefetch, a resident-K MMA-B packet cache, a warp-local shared dQ transpose epilogue, and an aligned-sequence specialization. On the SM86 test GPU the retained Q32/K64 64x64 path:
- passes all 12 focused head-64 NHD/HND and tail cases across the three generated candidates.
- passes Compute Sanitizer Racecheck for an aligned Q32/K64 launch and odd-tail HND fallback launches for both the K64 and exploratory K128 CTAs, all with `0 hazards` (`0 errors`, `0 warnings`).
- uses `128` registers/thread and `33,088` bytes of dynamic shared memory, preserving two resident K64 CTAs on SM86.
- executes `8.769M` instructions at the 512-token profile shape, down from `10.909M` for the committed fully predicated kernel.
- reports `131,072` local-load sectors, `8,192` local-store sectors, and `524,288` global-reduction sectors.
- measures `5.3165/20.7345 ms` at 4096/8192 tokens in serial NHD timing. HND measures `5.3134/20.6602 ms`.

The shared dQ transpose removes the per-atomic source shuffles and improves matched long-shape Sage timing by `3.8-4.7%`. The aligned specialization then improves the retained kernel by approximately `6.5-6.8%` without changing registers, shared-memory allocation, or reduction traffic. It is selected only when `seq_len % BlockN == 0`; every other length uses the existing predicated implementation. A same-build direct-global versus shared-LDSM A/B selects the direct-global resident-K packet because it is `3.3-4.1%` faster at serial NHD/HND 4096/8192 despite using more shared storage. The fused main kernel, not preprocessing or allocation, remains the performance blocker.

The coordinated backward quantization rewrite is accepted, and Q32/K64 is the selected initial quantization geometry. Quantization, forward matmul, and backward matmul geometry may now vary independently. The backward API, generator, generated dispatch, Python wrapper, and benchmark labels now represent configurations as `(QBlock, KBlock, backward CTA M, backward CTA N)`; forward `warp_q`/`warp_k` metadata is no longer overloaded for backward selection. The dormant generic scale lookup was also corrected to index true Q/K block scales. The new backward target therefore uses two K64 quantization domains inside one N128 CTA rather than forcing KBlock to equal the CTA N extent. Q128/K64 and Q128/K128 remain historical comparison points, not geometries to carry indefinitely.

The geometry-decoupling prerequisite builds successfully and passes all 12 focused tests. A serial K64 diagnostic after the interface-only change measured `5.5863/20.4472 ms` at 4096/8192 NHD; the 4096 Flash sample was high-variance, so this run is an interface-regression check rather than new performance evidence. The exact K64 device body and launch bounds were not changed.

The active focused source generation contains three head-64 backward instantiations for direct comparison. A production rebuild and the full validation matrix are deferred until the focused kernel is complete; the previous multi-configuration generator and dispatch can be restored from git when broader coverage resumes.

## Scope And Contracts

- SM80/SM86, matching the CUTLASS forward extension.
- fp16 `q`, `k`, `v`, `out`, and `dout`.
- Non-causal, fixed-length, dense attention.
- HND and NHD layouts.
- Production scope remains head dimensions 64 and 128; the active focused dispatch currently supports padded head dimension 64 only.
- No GQA/MQA: Q, K, and V must have matching head counts.
- Standalone backward API only. Trainable wrapper and autotuning remain deferred.
- Internal Q/K quantization geometry and backward CTA geometry are independent compile-time choices. The focused experiment uses one explicit pair rather than an autotuned search.
- Forward optimization is out of scope. If quantization-state or interface compatibility requires a forward change, modify only the CUTLASS forward kernel and leave the CUDA and Triton forward kernels unchanged.
- Correctness reference is FlashAttention backward.
- Kernel code remains torch-free in `qk_int8_sv_f16_bwd_kernel_cutlass_sm80.cuh`. Stable-torch validation, allocation, and launch code remain in `qk_int8_sv_f16_bwd_launch_cutlass_sm80.cuh`.

The arithmetic contract is fixed:

| Path | MMA precision |
|---|---|
| QK recompute | INT8 `m16n8k32` |
| dV | INT8 `m16n8k32` |
| dK | INT8 `m16n8k32` |
| dQ | INT8 `m16n8k32` |
| `dP = dO @ V.T` | FP16 `m16n8k16`, FP32 accumulator |

Do not replace dV, dK, or dQ with FP16 MMA. The intended advantage is SM80 INT8 tensor-core throughput and the smaller INT8 operand/register/shared-memory footprint. A custom hardware atom such as `int8_transposed_ldsm_cute.cuh` is allowed only when existing CuTe atoms cannot express the required operation.

## Redesign Objective And Amdahl Budget

The measured Sage tensor-active cycle-time proxy is already `0.600x` Flash, so the intended three-versus-five tensor-work reduction is present. Let `f = 0.44` approximate Flash's long-shape tensor-active fraction, let Sage tensor work cost `0.60x` Flash tensor work, and let `b` be Sage non-tensor cost relative to Flash non-tensor cost. The activity-based planning proxy becomes:

```text
speedup = 1 / (0.60 * f + b * (1 - f))
```

| Sage non-tensor factor `b` | Whole-backward proxy |
|---:|---:|
| 1.00 | 1.21x |
| 0.80 | 1.40x |
| 0.70 | 1.52x |
| 0.60 | 1.67x |

This is not a wall-time decomposition because tensor and non-tensor execution overlap. It does establish the required scale of change: merely matching Flash's non-MMA path leaves the ceiling near `1.21x`; approaching `1.5-1.67x` requires Sage's non-MMA path to become 30-40% cheaper than Flash's or to be hidden under tensor execution. The redesign must make the INT8 representation reduce movement and control work as well as tensor work.

The fixed initial research configuration is:

| Axis | Selected focus |
|---|---|
| Head dimension and dtype | head 64, fp16 inputs/outputs |
| Q/K quantization | QBlock=32, KBlock=64 |
| Backward matmul CTA | M64 x N128, 8 warps / 256 threads |
| On-the-fly P/dS format | current signed INT8 block scaling first; physical packing may change |
| Forward matmul | current fixed CUTLASS path unless compatibility requires a CUTLASS-only change |

Large-margin structural changes take precedence over isolated instruction, swizzle, predicate, or epilogue changes. A small-margin experiment is justified before the structural program is exhausted only when it removes a resource blocker, enables the new schedule, or supplies attribution needed for a larger rewrite. Otherwise, sub-3% candidates wait until the K128, one-touch operand, and software-pipeline designs have been retained or rejected with evidence.

## Current Implementation

The launch sequence has three kernels:

```text
preprocess:
  Delta_i = sum(out_i * dout_i)
  zero the float dQ accumulation workspace
  reduce dO pair scales and emit packed int8 dO

fused KV-owned main kernel:
  keep one K/V CTA tile resident
  for each adjacent 32-row Q/dO pair:
    recompute QK with int8 MMA
    compute dP = dO @ V.T with fp16 MMA
    form P and dS from QK, LSE, dP, and Delta
    quantize P and one logical dS tile
    accumulate dV, dK, and dQ with int8 MMA
    apply Q/K dequantization factors in the dK/dQ FP32 accumulation
  write owned dK/dV directly, and atomically reduce dQ

postprocess:
  convert the float dQ workspace to fp16
```

### Storage And Lifetimes

- K and V are CTA-resident across the complete M loop.
- Producer warps also build one packed MMA-B K fragment per resident 32-row pair from the global K tensor. Complete pairs use a predicated global CuTe partition; incomplete tail pairs are zero-filled by predication. dQ consumers load the four packed words directly, while QK, dV, and dK retain the existing transposed shared-memory path.
- Q, fp16 dO, and preprocess-produced int8 dO use one canonical 32-row resident pair each.
- The dead int32 score fragment is recast in place for P. The dP fragment is overwritten by dS.
- `score_pair_i8` has four slots, one P/dS pair for each N microtile. After its MMA operands die, each slot is reused by one warp-local FP32 dQ transpose.
- `WarpScratchStorage` has four slots for the mirrored dS orientation and the half dKV epilogue. The accepted tail prefetch keeps the 32-row fp16 dO pair in a separate allocation because it remains live while dQ consumes the scratch slots.
- Each exact-schedule warp owns one persistent 32-float dV or dK fragment. Source-correlated SASS keeps it in `R5` through `R36`. It is not the source of the launch-bound local traffic.
- Q/int8-dO, K/V, P/dS, and persistent dKV state overlap in the schedule and cannot be blindly aliased.

The score, dV, dK, and dQ paths use CuTe MMA/copy contracts. Resident Q/K/int8-dO transposed-B reads use the custom CuTe-facing LDSM atom rather than explicit shared transposes. The dKV epilogue uses `SmemCopyAtomdKVC` and `GmemTiledCopydKV` for row-contiguous vector stores.

### Exact Head-64 Schedule

For public `64x64x32x64`, eight warps cover `(M-half, N-tile)` ownership:
- the two M-half owners of each N microtile combine their P maxima, giving one P scale for each 32x16 tile.
- all eight owners combine their dS maxima, giving one logical dS scale for the 32x64 CTA slice; each warp writes its P/dS fragment directly at the final scale.
- all old pair store/reload/re-round rescale passes are eliminated.
- one M-half owner accumulates dV and the other dK.
- only the even N-tile owner executes dQ for its adjacent N pair.

The kernel has `__launch_bounds__(256, 2)`. The 128-register limit permits two 256-thread CTAs on SM86. Unbounded compilation uses 146 registers and permits only one CTA, which is slower despite having no local traffic. The resident-K packet cache adds 4,096 bytes of shared storage; its direct-global startup pack avoids the repeated transposed LDSM conversion for dQ and remains within the two-CTA K64 resource budget.

The dQ epilogue uses a 4x8 destination permutation. CTA-local fusion accumulates both N pairs before this epilogue, reducing global-reduction sectors from the original `2,097,152` to `524,288`. Each owner warp first writes its scaled 16x16 FP32 fragment into its dead `score_pair_i8` slot through a bank-aware deinterleave/XOR layout, then reads the four-row/eight-column atomic groups directly; this removes the two source shuffles per atomic output. The stage uses distinct `n_tile + m_half` slots and warp-local synchronization only. The exact specialization computes one contiguous lane-specific workspace base per dQ MMA fragment. Generic kernels retain the CuTe workspace view.

For `seq_len % BlockN == 0`, the launcher selects a compile-time aligned kernel that removes the scalar score/dS row-column checks and the dQ output-row check. Q/dO/K/V copies, row-state loads, resident-K packet construction, and dK/dV stores remain predicated. Extending the aligned path through those operations increased compiler local traffic and regressed long-shape timing, so arbitrary-length safety and the smaller specialization are both retained.

After dV and dK finish for a pair, odd-N warps 2 and 3 prefetch the next Q/int8-dO/fp16-dO pair while even-N warps execute dQ MMA, shared transpose loads, and atomics; warp 6 loads the next LSE/Delta pair. The Q/dO reuse barrier is at the dKV/dQ boundary, so the prefetch adds no CTA barrier. Separating fp16 dO from the dS/dKV scratch allocation costs 4 KiB but preserves the K64 two-CTA occupancy threshold.

### Block-Scaled Quantization Contract

The accepted backward-specific contract uses block-scaled Q/K with compile-time block sizes. Q32/K64 is the selected initial format. The current exact K64 kernel sees one invariant Q scale per 32-row mainloop pair and one invariant K scale per resident tile. The redesign generalizes this contract so a backward CTA can contain an integer number of quantization domains: the target N128 CTA has two independent K64 domains, and its M64 slice has two Q32 domains. The score path remains INT8 QK, dV uses separately quantized P and INT8 dO, and dK and dQ share one quantized dS tile within each quantization domain.

The implementation defines `dS = P * (dP - Delta) * sm_scale`, so the softmax scale is already included before dS quantization:

```text
score = int8(Q) @ int8(K) * (q_scale * k_scale)
dS_i8, ds_scale = quantize(dS)
dK += int8(dS_i8).T @ int8(Q) * (ds_scale * q_scale)
dQ += int8(dS_i8) @ int8(K).T * (ds_scale * k_scale)
```

This preserves all four INT8 MMA paths, removes the former second dS maximum/materialization, and moves Q/K dequantization factors to the FP32 dK/dQ accumulation. The backward wrapper already produces the block-scaled Q/K representation through `per_block_int8`; the forward quantization path is unchanged. The template parameters name quantization and matmul geometry separately, but the current exact kernel still requires `QuantBlockK >= CtaN`. Removing that restriction and indexing multiple quantization domains per CTA is the first redesign prerequisite.

The numerical and timing gates are complete for the existing focused candidates under the original tolerances. Q128/K64 has dQ/dK relative error around `4.10-4.20%`, compared with `4.85-4.99%` for Q128/K128. Q32/K64 is slightly more accurate than Q128/K64 and is at least as fast in controlled long-shape timing, so it is the fixed starting point for the new matmul schedule. Internal geometry decoupling happens now; public quantization controls and autotuning remain deferred.

## Current Baseline And Redesign Target

The active generated dispatch contains three focused public configurations:

| Backward config `(QBlock,KBlock,CtaM,CtaN)` | Generated warps | Current implementation |
|---|---:|---|
| `32x64x64x64` | 4 | default block-scaled exact internal eight-warp head-64 kernel |
| `128x64x64x64` | 4 | coarser-Q comparison on the same internal eight-warp kernel |
| `128x128x64x128` | 8 | exploratory internal sixteen-warp head-64 kernel |

The prior `{64,128}` head-dimension cross product and three other public block configurations are intentionally out of the active build during focused optimization. Retrieve them from git before resuming broader coverage.

Current focused resource figures are:

| Quantization / CTA | Internal launch | Registers/thread | Dynamic shared memory | SM86 residency |
|---|---:|---:|---:|---:|
| Q32/K64, 64x64 | 8 warps / 256 threads | 128 | 33,088 B | 2 CTAs |
| Q128/K64, 64x64 | 8 warps / 256 threads | 128 | 33,088 B | 2 CTAs |
| Q128/K128, 64x128 | 16 warps / 512 threads | 128 | 49,536 B | 1 CTA |

The K64 resource contract is current-source evidence. The direct-global resident-K packet is retained, not provisional. Generic and head-128 resource tables are intentionally omitted because those generated paths are not in the focused build; refresh them only after broader production generation is restored.

The selected redesign target is intentionally one configuration:

| Quantization | Backward CTA | Internal launch | Resource policy |
|---|---:|---:|---|
| Q32/K64 | M64 x N128 | 8 warps / 256 threads | one resident CTA is acceptable; avoid compiler-local traffic |

The target may use `__launch_bounds__(256, 1)` and materially more than 128 registers/thread if register-resident V, persistent dK+dV accumulators, or resident K operand packets remove larger shared-memory and synchronization costs. Two-CTA occupancy remains a requirement for the retained K64 baseline, not an absolute rule for the K128 redesign.

During controlled A/B work, the generator may temporarily contain the retained K64 baseline and the new K128 candidate. Once the candidate passes the retention gates and wins long-shape timing, collapse generated dispatch and Python configuration to the single selected Q32/K64 plus M64xN128 geometry. Remove Q128/K64, Q128/K128, the old K64 exact path, temporal K128 code, and helpers used only by removed geometries rather than preserving inactive variants in the source tree. The arbitrary-length predicated implementation for the selected geometry must exist before the old fallback is removed. All removed paths remain recoverable from git.

## Critical Redesign Program

- **Completed prerequisite:** the backward API, generator, generated dispatch, Python wrapper, and benchmark labels now use explicit Q/K quantization blocks plus backward CTA M/N. Forward `warp_q`/`warp_k` metadata is no longer used to select backward geometry.
- Implement a clean aligned head-64 M64xN128, eight-warp KV-owned kernel. It should follow the successful structural choices of Flash's SM86 head-64 schedule: 256 threads, persistent dK and dV accumulators, V resident in registers when profitable, distinct score/dKV/dQ warp layouts, and double-buffered Q/dO state. Do not port the rejected temporal-eight-warp body.
- Build one canonical on-the-fly P/dS packet per score tile and consume it in normal and transposed orientations through CuTe copy atoms. The design objective is one scalar materialization and one packed store, with no mirrored dS tile and no repeated operand repacking. An interleaved 16-bit `{P_i8, dS_i8}` physical packet is the first layout to evaluate.
- Keep two K64 and two Q32 quantization domains inside the M64xN128 CTA. Compute and apply their scales independently, then combine scaled FP32 dQ contributions before one N128 atomic epilogue. This preserves the selected accuracy geometry while halving K-tile ownership and matching Flash's dQ reduction granularity.
- Software-pipeline QK/dP, scalar P/dS production, and gradient MMA consumption across two packet buffers. Prefetch Q8, dO8, fp16 dO, and row state for the next M slice while dV/dK/dQ consume the current packet. Group-level synchronization may replace lockstep CTA barriers when ownership and buffer lifetimes make it race-free.
- Keep the retained K64 kernel unchanged as the benchmark reference while these structural pieces are incomplete. Once the new schedule is retained, simplify the codebase to the selected geometry before doing lower-margin tuning.

## Accuracy Policy

Schedule, ownership, layout, staging, and synchronization changes that preserve Q/K block sizes and the mathematical P/dS quantization contract must pass the existing tolerances. Do not loosen accuracy to hide an indexing, race, scale, or accumulation bug.

Changing Q/K block sizes or the on-the-fly P/dS numerical format is an explicit speed/accuracy experiment and may use a separately documented tolerance tier. Before changing a threshold, record dQ/dK/dV cosine similarity, Frobenius relative error, and maximum absolute error for the retained Q32/K64 format and the candidate across fixed NHD/HND, aligned, tail, and long-shape cases. An initial experimental outer ceiling on the focused randomized tests is `1 - cosine < 5e-3`, Frobenius relative error `< 0.10`, and maximum absolute error `< 0.35` for each gradient, with no NaN or Inf. These are maximum guardrails, not target tolerances; tighten them to the observed retained candidate margin after selection.

Do not loosen a threshold in response to one failing seed. A format-changing candidate must show a repeatable long-shape speed gain large enough to justify its measured quality loss, and its accuracy tier must be visible in the plan, tests, and benchmark output. Only one quantization/format candidate should remain active after the decision.

## Build And Validation Workflow

The generator currently emits exactly the three active head-64 backward instantiations and removes stale generated backward sources. `setup.py` runs it as part of source discovery, so focused iteration needs no mode flag:

```powershell
$Env:NVCC_APPEND_FLAGS = '-lineinfo'
$Env:MAX_JOBS = '4'
python setup.py build_ext --inplace
```

Use `MAX_JOBS=4` rather than `build_ext -j 4` on this Windows setup. The latter parallelizes both extension modules, launches two Ninja processes in the same object directory, and can fail with object-file `Permission denied` races.

List the active generated objects with:

```powershell
python scripts/generate_cutlass_bwd_instantiations.py --list-sources
```

Broader production generation is not hidden behind an environment toggle in the current tree. Restore the previous multi-configuration generator and matching Python/dispatch coverage from git, regenerate, and then build serially before running the broader matrix.

Focused correctness and Racecheck commands:

```powershell
python -m pytest -q tests/test_sagebwd_cutlass.py

$racecheck = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\compute-sanitizer.bat'
& $racecheck --tool racecheck --error-exitcode 1 python -m pytest -q `
  'tests/test_sagebwd_cutlass.py::test_sagebwd_cutlass_config_matches_flashattention[64-64-HND-config0]'
& $racecheck --tool racecheck --error-exitcode 1 python -m pytest -q `
  'tests/test_sagebwd_cutlass.py::test_sagebwd_cutlass_config_matches_flashattention[65-64-HND-config0]'
& $racecheck --tool racecheck --error-exitcode 1 python -m pytest -q `
  'tests/test_sagebwd_cutlass.py::test_sagebwd_cutlass_config_matches_flashattention[65-64-HND-config2]'
```

Before broader production use, restore the multi-configuration generator/dispatch from git, build the resulting objects with four workers, run the complete correctness matrix and Racecheck, and verify the expected generated source count.

## Latest Results

### Focused Timing

SM86 laptop RTX 3080 Ti, batch 1, 16 heads, head dimension 64, public Q32/K64, kernel-only mode, 30 warmups, 100 repeats:

| Layout | Sequence | Sage median | Flash median | Sage/Flash speedup |
|---|---:|---:|---:|---:|
| NHD | 4096 | 5.3165 ms | 4.5301 ms | 0.852x |
| NHD | 8192 | 20.7345 ms | 17.1755 ms | 0.828x |
| HND | 4096 | 5.3134 ms | 4.5670 ms | 0.860x |
| HND | 8192 | 20.6602 ms | 17.1626 ms | 0.831x |

These are serial aligned-specialization measurements. Relative to the retained shared-dQ kernel without the aligned specialization, they improve Sage by approximately `6.5-6.8%` across NHD/HND and 4096/8192. Sage remains approximately `16-21%` slower than FlashAttention.

A forced reverse rebuild independently confirmed the decision in NHD. Ninja recompiled the backward translation unit and all three focused generated objects for both binaries. The fully predicated kernel measured `5.6320/22.2366 ms`; after rebuilding the aligned source, the same 4096/8192 run measured `5.2127/20.9284 ms`, improvements of `7.44%/5.88%`. The HND reverse run also favored aligned, but its predicated 8192 sample had `1.306 ms` standard deviation, so its larger apparent gain is excluded from the retention claim.

A previous serial thermal diagnostic showed approximately 2.3% timing drift as GPU temperature, power, and clocks varied. Treat sub-2% differences as directional unless supported by profile counters or a controlled A/B rebuild.

The earlier full requested matrix predates the exact eight-warp path. Do not use those absolute times as the current head-64 baseline. Rerun the matrix after focused optimization stops and broader generated coverage is restored.

### Focused NCU Profile

Profile shape: sequence 512, batch 1, 16 heads, NHD, ten warmups.

| Metric | Fully predicated | Aligned scalar/dQ specialization |
|---|---:|---:|
| Duration | 168.448 us | 153.280 us |
| Instructions | 10.909M | 8.769M |
| Registers/thread | 128 | 128 |
| Dynamic shared memory | 33,088 B | 33,088 B |
| Local-load sectors | 308,224 | 131,072 |
| Local-store sectors | 28,672 | 8,192 |
| Global-reduction sectors | 524,288 | 524,288 |
| Active warps/scheduler | about 3.57 | 3.52 |
| Eligible warps/scheduler | 0.56 | 0.47 |

The aligned specialization removes `19.6%` of the profile instructions and reduces profile duration by `9.0%` without changing occupancy resources or dQ reduction traffic. It reports `20.94%` barrier stalls, `5.56%` long-scoreboard stalls, and `12.35%` short-scoreboard stalls. INT8/FP16 tensor activity is `10.54%/5.27%`. Shared load/store bank conflicts are `288,489/67,033` with `2,172,714/585,139` wavefronts. Broadly removing Q/dO/K/V copy, row-state, K-packet, and dKV-store predicates raised local traffic and regressed long-shape timing, so only scalar score/dS and dQ output predicates are specialized.

Preprocessing and dQ conversion contribute roughly 0-2% of end-to-end time. Their CUTLASS-vs-Triton results vary by shape and do not justify separate optimization while the fused kernel is slower than FlashAttention.

### Flash Comparison At Target Shapes

The target-shape profiles show that the intended tensor-core arithmetic reduction is already realized. They also show why it does not become a `1.67x` whole-backward speedup:

| Metric | Sage 4096 | Flash 4096 | Sage 8192 | Flash 8192 |
|---|---:|---:|---:|---:|
| NCU main-kernel duration | 6.206 ms | 5.748 ms | 24.554 ms | 22.859 ms |
| Tensor-pipe activity | 24.88% (`16.59%` IMMA + `8.29%` HMMA) | 44.74% HMMA | 25.14% (`16.76%` IMMA + `8.38%` HMMA) | 45.00% HMMA |
| Total warp instructions | 524.664M | 155.685M | 2.088B | 617.554M |
| Tensor warp instructions | 25.166M | 41.943M | 100.663M | 167.772M |
| Non-tensor warp instructions | 499.498M | 113.742M | 1.988B | 449.782M |
| LSU throughput | 68.05% | 20.98% | 68.72% | 21.03% |
| L1/TEX throughput | 68.05% | 27.93% | 68.72% | 27.43% |
| DRAM throughput | 5.07% | 4.25% | 3.77% | 2.72% |

Sage executes `3.37-3.38x` as many total instructions and `4.39-4.42x` as many non-tensor instructions as Flash. Multiplying duration by tensor activity gives a tensor-active cycle-time proxy of `1.544/2.571 ms` at 4096 and `6.173/10.286 ms` at 8192 for Sage/Flash. Both ratios are `0.600`, exactly the theoretical three-versus-five MMA-work ratio. Tensor-core throughput is delivering the intended arithmetic saving; non-MMA work consumes it.

The 4096 SASS operation census identifies where that non-MMA excess lands:

| Warp operation | Sage | Flash | Sage/Flash |
|---|---:|---:|---:|
| Explicit shared loads, excluding LDSM | 25.199M | 0.033M | 769x |
| Shared stores | 31.883M | 8.520M | 3.74x |
| LDSM loads | 29.360M | 21.004M | 1.40x |
| Async global-to-shared loads | 2.122M | 1.081M | 1.96x |
| Global loads | 3.547M | 2.097M | 1.69x |
| Global reductions | 8.389M | 4.194M | 2.00x |
| Branches | 8.996M | 0.803M | 11.2x |
| Compiler-local loads | 2.097M | 0 | unbounded |

Shared load/store wavefronts are `141.330M/36.535M` for Sage versus `84.148M/9.187M` for Flash, or `1.68x/3.98x`. Sage also transfers `1.29x` the DRAM bytes, but the low DRAM throughput rules out HBM bandwidth as the limiting resource. Source counters point to scalar score/softmax/dS reconstruction, FP32-to-INT8 conversion, dQ shared staging, transposed-loader shuffles and byte permutations, dKV rescaling, and address/control work. Flash's installed binary lacks CUDA line information, so its side of this attribution uses SASS operation classes rather than source lines.

The `1.67x` figure applies only to the five equal-size MMA paths: four paths at twice the FP16 throughput plus one FP16 path reduce tensor work from five units to three. Flash's full backward has duration-weighted HMMA activity of `43.68%` at 4096 and `44.46%` at 8192 after including its preprocess and dQ-conversion kernels. If tensor activity were a disjoint critical-path fraction, reducing it by `40%` would imply only `1.212x/1.216x` whole-backward ceilings. This is an activity-based Amdahl proxy, not a wall-time decomposition: tensor and non-tensor execution overlap. The 512-token weighted activity is only `20.47%`, giving a diagnostic `1.089x` proxy and demonstrating why short-shape activity must not be extrapolated to long shapes.

## Completed Work

- Built the standalone stable-torch backward wrapper and generated static dispatch.
- Migrated all five matmuls to CuTe/CUTLASS contracts with four INT8 paths and one FP16 path.
- Added the CuTe-facing transposed INT8 LDSM atom and a standalone copy-plus-MMA test (`failures=0`).
- Adopted the KV-owned fused schedule with direct dK/dV output and atomic dQ accumulation.
- Made K/V CTA-resident and Q/dO resident in canonical 32-row pairs.
- Recast score and dP fragments in place for P and dS.
- Materialized score operands with warp-contiguous CuTe C-copy contracts.
- Fused Delta, dO quantization, dO scale reduction, and dQ zeroing into one preprocessing CTA per pair.
- Added `bench/bench_sagebwd_cutlass.py` with kernel-only/end-to-end modes, statistics, TFLOPS, CSV output, and ranking.
- Focused the generator and Python dispatch on three head-64 comparison configurations; setup now emits those three objects and removes stale backward generated sources.
- Added two-dimensional eight-warp ownership, direct final-scale quantization, a 128-register launch bound, 4x8 dQ ownership, and the lane-base contiguous dQ atomic address.
- Replaced the dQ epilogue's source shuffles with a scaled-FP32 warp-local shared transpose in dead score-pair storage. The candidate passes all 12 focused tests and both odd-tail HND Racecheck gates, and its matched long-shape A/B is faster by `3.8-4.7%`.
- Added a compile-time aligned scalar/dQ specialization with automatic predicated fallback. It cuts profile instructions by `19.6%`, improves serial long-shape timing by approximately `6.5-6.8%`, and passes aligned plus odd-tail K64/K128 Racecheck.
- Profiled Sage and Flash at 4096 and 8192, confirming the expected `0.600` tensor-work ratio and attributing the remaining gap to non-MMA execution and on-chip movement.
- Audited shared/register lifetimes. Existing score/dP and fp16-dO/scratch aliases are valid. Concurrently live resident and accumulator families are not overlaid.

## Accepted Optimization Ledger

| Change | Evidence |
|---|---|
| Head-64 direct pair-scale dS quantization | `17.176M -> 15.933M` instructions and `290.432 -> 276.352 us` on the generic four-warp path |
| 4x8 dQ ownership | reduction sectors `2,097,152 -> 1,048,576`. Focused correctness clean |
| Eight-warp `(M-half, N-tile)` ownership | removes all pair requantization. Unbounded resources `146` registers and `14.654M` instructions |
| `__launch_bounds__(256,2)` | two resident CTAs. 4096 timing improved from about `11.69` to `9.02-9.22 ms` despite local traffic |
| Lane-base contiguous dQ atomics, exact path only | `15.112M -> 14.428M` instructions, `802,816 -> 675,840` local-load sectors, `8.92-9.01 ms` at 4096. Correctness and Racecheck clean |
| Exact-path row-state cache | Caches the two lane-owned Q-scale/LSE/Delta rows per materialization phase. `14.428M -> 14.169M` instructions, `8.586-8.604 ms` at 4096, correctness and Racecheck clean |
| Exact-path accumulator-coordinate mapping | Replaces identity-view coordinate extraction in the three scalar materialization loops with the verified lane/fragment mapping. `720,896 -> 327,680` local-load sectors, `65,536 -> 40,960` local-store sectors, `8.43-8.54 ms` at 4096, correctness and Racecheck clean |
| Block-scaled Q/K and one logical dS quantization | Uses `QBlock=32`, `KBlock=64`, one 32x32 dS scale, one conversion, and mirrored MMA orientations. Instructions fell `14.127M -> 12.479M`, local loads `327,680 -> 65,536`, local stores `40,960 -> 8,192`, shared memory `25,312 -> 24,896 B`, and profile duration `260.800 -> 245.504 us`. Serial 4096 timing improved to `7.96-8.06 ms`; strict correctness and Racecheck are clean |
| CTA-local dQ fusion with one 32x64 dS scale | Splits output dimensions across four owner warps and accumulates both N32 pairs before one atomic epilogue. Reduction sectors fell `1,048,576 -> 524,288`, instructions to `11.750M`, barrier stalls to `28.92%`, and profile duration to `211.552 us`. Local traffic rose to `458,752/40,960` load/store sectors, but serial 4096 timing improved to `6.78-6.83 ms`; strict correctness and Racecheck are clean |
| Compile-time quantization geometry | Makes Q/K quantization blocks independent kernel template parameters and replaces runtime block division in scale extents and indices. For 64x64/Q128/K64, instructions fell `11.750M -> 11.332M` and NCU duration `211.552 -> 208.448 us` with unchanged `128` registers, `24,896 B` shared memory, and `458,752/40,960` local sectors. For 64x128/Q128/K128, instructions fell `11.039M -> 10.893M`; controlled long-shape timing ultimately favors the 64-column CTA |
| Ampere hardware max reduction | Replaces five shuffle/fmax steps with one `redux.sync.max.u32` for nonnegative P, dS, and dO maxima. Q32/K64 instructions fell `11.332M -> 11.069M`, profile duration `207.872 -> 202.016 us`, and short-scoreboard stalls `13.23% -> 10.30%` with unchanged resources and traffic. Serial timing improved `6.7804 -> 6.6222 ms` at 4096 and `25.7622 -> 25.1678 ms` at 8192; correctness and Racecheck are clean |
| In-loop Q/dO tail prefetch | Odd-N warps prefetch the next Q/int8-dO/fp16-dO pair while even-N warps execute dQ MMA, shuffles, and atomics. Separating fp16 dO from dS/dKV scratch raises shared memory `24,896 -> 28,992 B` but keeps two K64 CTAs. NCU duration falls `202.016 -> 184.448 us`, barrier stalls `30.89% -> 24.01%`, eligible warps `0.45 -> 0.52`, and local sectors `458,752/40,960 -> 0/0`. NHD timing improves `6.6222 -> 6.1240 ms` at 4096 and `25.1678 -> 24.0691 ms` at 8192; all 12 focused tests and K64/K128 odd-tail Racecheck cases are clean |
| Resident-K MMA-B packet cache, direct global startup pack | Producer warps retain the normal async K tile for QK, then form the dQ K fragment from a predicated global MMA-B partition once per resident K pair. dQ consumers load four cached words instead of repeating the transposed LDSM conversion. The candidate uses `33,088 B` shared memory, `128` registers, `10.976M` instructions, `196,608/8,192` local load/store sectors, and `181.248 us` at the 512-token profile shape. The same-build shared-transpose A/B measured the fallback at `172.032 us`, `10.910M` instructions, `28,992 B`, and `131,072/8,192` local load/store sectors, but serial fallback timing regressed to `5.8972/23.1982 ms` NHD and `5.8542/22.8951 ms` HND at 4096/8192. Direct-global timing remains selected because it wins the controlled long-shape comparison by `3.3-4.1%`; correctness and both K64/K128 odd-tail Racecheck cases are clean |
| Shared dQ transpose epilogue | Stages one scaled-FP32 16x16 dQ fragment per owner warp in a dead `score_pair_i8` slot, then reads direct four-row/eight-column groups for the existing eight FP32 atomics. Matched source-level A/B reduced profile duration `180.896 -> 168.448 us`, barrier stalls `24.05% -> 19.63%`, and raised eligible warps `0.51 -> 0.56`, with unchanged `128` registers, `33,088 B` shared memory, and `524,288` reduction sectors. NHD/HND 4096/8192 timing improved by `3.8-4.7%`; all focused correctness and odd-tail Racecheck gates are clean. Extra shared wavefronts and local sectors remain the next tuning target |
| Aligned scalar/dQ specialization | Selects a compile-time aligned kernel only for `seq_len % BlockN == 0` and otherwise launches the predicated kernel. It removes scalar score/dS and dQ row predicates, reducing profile instructions `10.909M -> 8.769M`, local sectors `308,224/28,672 -> 131,072/8,192`, and duration `168.448 -> 153.280 us` with unchanged `128` registers, `33,088 B`, and `524,288` reduction sectors. Serial long-shape timing improves by approximately `6.5-6.8%`; a reverse NHD rebuild confirms `7.44%/5.88%` at 4096/8192. All focused tests and aligned/odd-tail K64/K128 Racecheck gates are clean |

## Rejected Experiment Ledger

| Experiment | Rejection evidence |
|---|---|
| Direct pair-scale specialization for head-128 | reached 255 registers and regressed long-shape timing |
| Shared dQ staging | correctness/Racecheck clean, but 168 registers and about `11.65 ms` at 4096 |
| Standard C retile for dQ | retained `2,097,152` reduction sectors and increased registers |
| Direct row-major dQ fragment matching | failed all 32 correctness cases (`dQ` cosine about 0.762) |
| First packed dQ shuffle | increased reduction sectors to `5,242,880` |
| Full-warp row-major shuffle | clean and `1,572,864` sectors, but 32 shuffles/fragment caused about `12.11 ms` |
| CUDA shared-spill pragma | required static shared memory and did not change local sectors or storage |
| Two dKV fragments in explicit shared memory | local traffic fell, but shared traffic raised 4096 timing to about `9.31 ms` |
| Split retained dKV tensor into fixed 1D fragments | no resource or code-generation change |
| Broad phase-scoped global view reconstruction | instructions/local loads fell, but 4096 was neutral at `9.25 ms` and 512 regressed to `0.350 ms` |
| Four-warp named barriers for scale and operand exchange | profile duration stayed neutral (`273.28 -> 272.51 us`), barrier stalls remained `32.27%`, and local loads regressed to `802,816` |
| Asymmetric loop-tail arrive/wait barrier | barrier stalls fell to `29.98%` and local loads to `458,752`, but short-scoreboard stalls rose to `20.91%`, instructions to `14.735M`, and duration to `280.83 us` |
| Flash-style `exp2f` softmax rewrite | saved `121K` profile instructions but raised local loads to `868,352`. 4096 timing was neutral at `8.583 ms`, so the spill tradeoff is rejected |
| Four-value K-scale cache in both scalar phases | reduced instructions to `13.935M` and profile duration to `261.7 us`, but raised local loads to `933,888`. 4096 timing was neutral at `8.569 ms` |
| Four-value K-scale cache only for dS·K quantization | reduced instructions to `14.032M` and local loads to `671,744`, but two 4096 runs regressed to `8.681-8.710 ms` |
| Raw dQ pointer recomputation after coordinate mapping | local loads fell only `327,680 -> 311,296`, while instructions rose to `16.032M`, profile duration to `283.584 us`, and short-scoreboard stalls to `19.05%` |
| Naive raw dQ pointer per atomic | local loads fell to `393,216`, but address arithmetic raised instructions to `16.676M` and timing to `9.57 ms` |
| Shared-memory padding and several isolated swizzle/copy-width changes | neutral or worse conflict-per-wavefront, resources, or timing |
| Register-fed dQ operand and packed dS stores | reduced some raw conflicts but worsened conflict rates or store traffic |
| Cooperative per-warp dS-scale reduction | Replacing each warp's unrolled shared loads with lane-partitioned loads plus warp shuffles reduced 64x128 instructions `10.893M -> 10.778M`, but duration regressed `263.680 -> 265.536 us`. On 64x64 it raised instructions `11.332M -> 11.430M` and duration `208.448 -> 210.272 us` |
| One-time odd-CTA phase skew | A `256 ns` skew intended to desynchronize the two resident CTAs regressed 4096 timing `6.6222 -> 6.7415 ms` and was neutral at 8192 (`25.1678 -> 25.1944 ms`). Cross-CTA alignment is not sufficient; overlap requires an in-loop schedule change |
| Shared atomic CTA dS maximum | Eight lane-0 `atomicMax` operations plus one shared read replaced each warp's serial scale loop, but contention/reset overhead raised instructions `11.069M -> 11.151M` and duration `202.016 -> 203.072 us` |
| Full packed Q/K/dO MMA-B mirror | Passed the focused correctness suite, but raised shared memory to `37,184 B`, instructions to `12.140M`, and local load/store sectors to `776,704/53,248`; the higher INT8 activity did not improve duration |
| No-inline resident-K pack helper | Isolated the startup conversion in a device helper, but profile duration rose to `183.968 us` and local loads to `477,696` |
| Scalar direct-partition resident-K pack | Replaced the existing LDSM startup conversion with a direct shared partition copy, but duration rose to `184.064 us` and local loads to `393,216` |
| Tail-only resident-K shared fallback in the main body | Preserved tail safety but caused the compiler to retain the fallback state for all shapes: `184.512 us`, `11.058M` instructions, and `393,216/20,480` local load/store sectors |
| Same-build shared-LDSM resident-K fallback | Removing the packet allocation reduced shared memory `33,088 -> 28,992 B` and local loads `308,224 -> 131,072`, but raised barrier stalls `19.63% -> 22.51%`, profile duration `168.448 -> 172.032 us`, and serial NHD/HND timing to `5.8972/23.1982 ms` and `5.8542/22.8951 ms` at 4096/8192. The direct-global packet is retained |
| Retained temporal eight-warp `64x128` kernel | Correct after updating its dormant Q/K scale loads to the block-scale contract, but failed the profile gate at `551.040 us`, `17.283M` instructions, `168` registers, `1,048,576` reduction sectors, and only `0.25` eligible warps/scheduler. The old two-orientation dS and unfused dQ body overwhelm any benefit from halving physical warps, so it does not justify a direct port. |
| Raw-INT32 dQ transpose stage | Reduced the scaled-FP32 stage from `10.909M` to `10.891M` instructions and lowered shared-store conflicts, but regressed profile duration `168.448 -> 178.592 us`, raised short-scoreboard stalls `11.81% -> 15.14%`, and reduced eligible warps `0.56 -> 0.52`. Scaling before the shared store overlaps better with the atomic load/issue chain. |
| Scaled-FP16 dQ transpose stage | Passed all 12 focused accuracy cases, but conversion and compiler-local overhead raised instructions to `11.254M`, local load/store sectors to `393,216/20,480`, and duration to `180.160 us`. NHD 4096/8192 timing regressed to `5.8394/22.7291 ms`, so halving the stage element size does not improve execution. |
| Remove trailing dQ-stage `__syncwarp()` | The barrier is redundant with the following CTA barrier, but the reverse rebuild A/B was neutral at 4096 (`5.6755` versus `5.6745 ms`) and only `0.85%` faster at 8192 (`22.0191` versus `22.2085 ms`), while the profile was slightly slower. Keep the explicit stage-lifetime bracket unless a larger scheduling change removes it. |
| Row-major dQ transpose stage | Cut instructions to `10.215M`, but raised shared load/store conflicts to `408,972/183,080`, local loads to `425,984` sectors, and profile duration to `170.176 us`. Serial NHD timing was `5.7431/22.3099 ms` at 4096/8192 versus `5.6929/22.2536 ms` for the retained swizzle. |
| Simpler affine-XOR dQ stage swizzle | Preserved low shared-conflict counts, but compiler local traffic rose to `454,656/40,960` sectors, instructions to `11.019M`, and duration to `172.576 us`; the deinterleave/XOR mapping remains the best code-generation and scheduler tradeoff. |
| Double-buffer resident Q/dO stages | Separating the aliased fp16 dO and dS/dKV scratch storage restored correctness, but the candidate regressed from `0.3123 -> 0.3369 ms` at 512 and `8.43-8.54 -> 8.90 ms` at 4096. It increased dynamic shared memory to `37,984 B`, local-load sectors to `774,656`, and instructions to `14.662M`. |
| Broad unpredicated aligned path | Removing Q/dO/K/V copy, row-state, resident-K packet, and dKV-store predicates increased local sectors from `131,072/8,192` to `262,144/16,384`, instructions from `8.769M` to `8.823M`, profile duration from `153.280` to `154.016 us`, and NHD timing to `5.4006/20.8451 ms`. Retain only scalar score/dS and dQ-row specialization. |

Do not repeat rejected paths without a new attribution result that changes the tradeoff. The rejected temporal-eight-warp K128 result applies to that old two-orientation, unfused-dQ body; the clean M64xN128 program above is a different schedule and must not inherit the old rejection by association.

## Remaining Bottlenecks

- Long-sequence gap: current Sage is approximately `1.16-1.21x` slower than FlashAttention at 4096/8192, depending on layout and thermal state.
- Non-MMA instruction volume: aligned Sage still executes `499.5M` non-tensor warp instructions at 4096 and `1.988B` at 8192, versus `113.7M` and `449.8M` for Flash. The remaining work is scalar/vector and memory scheduling, not missing tensor-core arithmetic.
- Scalar score/softmax/dS work: `expf`, LSE/Delta selection, scale products, FP32-to-INT8 conversion, bounds/control flow, and dKV rescaling are repeated around each N tile. The aligned specialization removes only the provably unnecessary full-shape predicates; source counters still identify these loops as major instruction sources.
- On-chip movement: at 4096 Sage has `68.05%` LSU/L1 throughput versus Flash's `20.98%` LSU and `27.93%` L1/TEX. Sage issues `25.199M` explicit shared loads, `31.883M` shared stores, and `2.097M` compiler-local loads; Flash issues `0.033M`, `8.520M`, and zero respectively in the corresponding SASS classes. Shared wavefronts are `1.68x/3.98x` higher for load/store.
- Transposed INT8 B preparation: the custom Q/K/int8-dO loader performs integer shuffles and byte permutations around LDSM, and the same operand transformation is consumed by dV, dK, and dQ. It is a structural candidate only if a replacement reduces both instruction and long-shape time without adding a CTA barrier.
- Synchronization and dependencies: aligned barrier/short-scoreboard stalls remain `20.94%/12.35%`; the loop still publishes scales, quantized operands, dKV state, and dQ staging through dependent barriers. Lower eligible warps (`0.47` per scheduler) indicate limited independent work despite the two-CTA occupancy contract.
- dQ reduction: CTA-local fusion and 4x8 ownership reduce the profile reduction contract to `524,288` sectors, but at 4096 Sage still issues `2x` Flash's global-reduction instructions/sectors. FP32 atomics remain a dependency chain in both implementations, so changing ownership requires full workspace and launch accounting.
- N-tile amortization: Sage's K64 main kernel uses twice the K-tile count of Flash's K128 head-64 kernel. This repeats Q/dO, row-state, scalar score/dS, and reduction work more often. The new M64xN128 design deliberately allows one resident CTA and higher register use when they buy register-resident operands, fewer barriers, and lower total instruction volume; useful issue efficiency and no compiler-local traffic matter more than nominal occupancy.
- Kernel-internal geometry coupling: host configuration is now decoupled, but the current exact kernel still requires the K quantization block to cover the complete CTA N tile. That prevents the selected Q32/K64 format from using an N128 backward tile and is the next implementation barrier.
- Helper kernels and HBM: preprocessing/allocation and dQ conversion are only about 0-2% of Sage's long-shape time, and Sage/Flash DRAM throughput is low (`5.07%/4.25%` at 4096). They are not the current optimization targets.
- Head-128: it is absent from the active generated dispatch and remains recoverable from git; restoration and optimization are deferred until focused head-64 work is complete.

## Ordered Next Work

### Priority 0: Structural Backward Redesign

- **Configuration separation complete.** Add exactly one candidate mapping: Q32/K64 quantization to head-64 M64xN128 with eight internal warps. Keep the current K64 mapping only for same-build comparison.
- Compile a resource-complete aligned K128 skeleton before filling in lower-level tuning. Establish that 256 threads, persistent dK+dV state, selected shared buffers, and any register-resident V/K packets fit the SM86 architectural limits without compiler-local memory. Do not impose the K64 128-register launch bound.
- Implement the score/dP ownership and the two-Q32-by-two-K64 scale-domain mapping. Validate exact accumulator coordinates and scale application with a standalone small-shape correctness probe before adding optimized stores or overlap.
- Implement the canonical P/dS packet and all three gradient consumers around it. The first complete version should halve K-tile-level dQ reductions and remove mirrored dS materialization; source counters must confirm that it does not recreate the old transposed-loader and shared-store excess elsewhere.
- Add the two-stage Q/dO/row-state and P/dS software pipeline. Evaluate register-resident V first, then persistent K operand packets only if register headroom remains. Synchronization changes are part of this coordinated schedule, not isolated barrier substitutions.
- Add the predicated arbitrary-length specialization for the selected geometry, then run matched K64/K128 correctness, Racecheck, source/profile, and serial 4096/8192 NHD/HND A/B. If K128 is retained, remove the old generated configurations and geometry-specific code in the same development phase.

### Priority 1: Format And Remaining Structural Pressure

- If scalar quantization remains on the critical path after the one-touch schedule, evaluate one format change at a time: power-of-two P/dS scales, a different P/dS packet representation, or a coarser Q/K block. Use the separate accuracy tier and keep only the selected format.
- If dQ atomics remain material after N128 fusion, compare alternative ownership or split workspaces only with complete workspace, reduction-launch, and end-to-end timing. Sector reduction alone is not evidence.
- If shared movement remains above Flash after canonical packing, use fresh source attribution to redesign the specific copy atom or register handoff. Do not resume the prior broad mirror, padding, or swizzle search without a new structural target.

### Priority 2: Small-Margin Work

Only after Priority 0 and applicable Priority 1 designs are retained or rejected should isolated predicate, address, conversion, bank-conflict, or epilogue changes resume. Keep them only when controlled long-shape timing exceeds thermal noise and the added code does not undermine the single-configuration design.

## Deferred Work

- Forward performance optimization and forward autotuning. A compatibility change, if required, is limited to the CUTLASS forward kernel at one fixed geometry; do not change the CUDA or Triton forward kernels.
- Production exposure of Q/K quantization block sizes and forward/backward matmul autotuning.
- Additional backward block sizes, head-128, and restoration of broad generated dispatch while the selected head-64 M64xN128 path is under investigation.
- Backward trainable/compile wrappers.
- GQA/MQA, causal attention, variable length, and additional dtypes.

## Retention Gate

Retain an optimization only when all applicable checks pass:
- FlashAttention-referenced correctness, including HND/NHD and tail cases, under the applicable schedule-preserving or format-changing accuracy policy above.
- Compute Sanitizer Racecheck.
- serial warmed 4096- and 8192-token timing, with thermal drift considered. The 512-token shape is profile context only and is too noisy for retention decisions.

Also check the following, while using the overall speed as the optimization target:
- registers, shared memory, stack/local memory, and occupancy. One resident K128 CTA and register use above 128/thread are acceptable; compiler-local traffic requires explicit performance evidence.
- global atomic/reduction sectors.
- shared conflicts per wavefront and synchronization behavior.
- target-shape non-tensor instructions and explicit shared load/store classes relative to both the K64 baseline and Flash.

A prerequisite refactor may be retained without an immediate timing gain when it is required for the selected K128 schedule and does not regress the K64 baseline. A standalone micro-optimization below approximately 3% is not sufficient while higher-margin structural work remains. The complete K128 candidate must beat the K64 baseline by a stable long-shape margin and materially improve at least one governing structural counter; nominal occupancy or a 512-token duration alone cannot decide it.

## Key Artifacts

- Benchmark harness: `bench/bench_sagebwd_cutlass.py`
- Focused profile harness: `build/profile_sagebwd_once.py`
- Latest benchmark CSVs: `build/bench_sagebwd_q32_k64_aligned_reverse_candidate_nhd_8192_4096.csv`, `build/bench_sagebwd_q32_k64_aligned_reverse_candidate_hnd_8192_4096.csv`, `build/bench_sagebwd_q32_k64_predicated_reverse_baseline_nhd_8192_4096.csv`, `build/bench_sagebwd_q32_k64_predicated_reverse_baseline_hnd_8192_4096.csv`
- Latest Sage profile CSVs: `build/ncu_sage_hd64_q32_k64_dq_shared_aligned_scalar.csv`, `build/ncu_sage_hd64_q32_k64_aligned_seq4096_sass_ops.csv`, `build/ncu_sage_hd64_q32_k64_aligned_seq8192_core_compare.csv`
- Latest Flash comparison CSVs: `build/ncu_flash_hd64_seq4096_sass_ops.csv`, `build/ncu_flash_hd64_seq8192_core_compare.csv`, `build/ncu_flash_hd64_seq8192_full_weight.csv`
- Earlier focused benchmark CSVs: `build/bench_sagebwd_coord_mapping.csv`, `build/bench_sagebwd_coord_mapping_repeat.csv`, `build/bench_sagebwd_coord_mapping_512.csv`
- Earlier focused profile CSVs: `build/ncu_sage_hd64_coord_mapping.csv`, `build/ncu_sage_hd64_coord_mapping_final.csv`
- Previous launch-bound profile: `build/ncu_sage_hd64_2dwarp8_lb2.csv`
- Runtime tests: `tests/test_sagebwd_cutlass.py`
- Transposed-copy test: `tests/cuda/test_qattn_cutlass_bwd_int8_transposed_copy.cu`

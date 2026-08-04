# QAttn CUTLASS Backward Kernel Plan

## Status Snapshot

The correctness-first CuTe migration is complete. Optimization is intentionally focused on one active public head-64 `64x64x32x64` configuration. That public four-warp instantiation selects an internal eight-warp `2 M-halves x 4 N-microtiles` kernel. Other block-size and head-dimension paths are temporarily removed from the active generated dispatch while this kernel is optimized. The prior implementations remain recoverable from git.

The latest focused kernel combines block-scaled Q/K, one logical dS quantization, CTA-local dQ fusion, compile-time quantization geometry, Ampere hardware max reductions, in-loop Q/dO tail prefetch, and a provisional resident-K MMA-B packet cache. On the SM86 test GPU the retained Q32/K64 64x64 path:
- passes all 12 focused head-64 NHD/HND and tail cases across the three generated candidates.
- passes Compute Sanitizer Racecheck on odd-tail HND cases for both the default K64 and exploratory K128 CTAs with `0 hazards` (`0 errors`, `0 warnings`).
- uses `128` registers/thread and `33,088` bytes of dynamic shared memory, preserving two resident K64 CTAs on SM86.
- executes `10.976M` instructions at the 512-token profile shape.
- reports `196,608` local-load sectors, `8,192` local-store sectors, and `524,288` global-reduction sectors.
- measures `5.9443/23.4593 ms` at 4096/8192 tokens in a serial 30-warmup/100-repeat NHD run. HND measures `5.8935/23.2125 ms`.

The direct-global packet cache is effectively tied with the earlier shared-LDSM resident-K cache in long-shape timing, while reducing startup conversion instructions and local traffic in the profile. It remains provisional pending the final documentation, pre-commit, and broader validation pass. The fused main kernel, not preprocessing or allocation, remains the performance blocker.

The coordinated backward quantization rewrite is accepted. Quantization geometry is intentionally independent from matmul geometry. The focused build contains Q32/K64 and Q128/K64 on the two-resident-CTA 64x64 path plus exploratory Q128/K128 on the one-resident-CTA 64x128 path. Controlled long-shape timing and accuracy favor Q32/K64 as the default: it is at least as fast as Q128/K64, is slightly more accurate, and the 64-column CTA is more stable than the wider candidate after compile-time scale indexing.

The active focused source generation contains three head-64 backward instantiations for direct comparison. A production rebuild and the full validation matrix are deferred until the focused kernel is complete; the previous multi-configuration generator and dispatch can be restored from git when broader coverage resumes.

## Scope And Contracts

- SM80/SM86, matching the CUTLASS forward extension.
- fp16 `q`, `k`, `v`, `out`, and `dout`.
- Non-causal, fixed-length, dense attention.
- HND and NHD layouts.
- Head dimensions 64 and 128.
- No GQA/MQA: Q, K, and V must have matching head counts.
- Standalone backward API only. Trainable wrapper and autotuning remain deferred.
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
    quantize P, dS*q_scale, and dS*k_scale
    accumulate dV, dK, and dQ with int8 MMA
  write owned dK/dV directly, and atomically reduce dQ

postprocess:
  convert the float dQ workspace to fp16
```

### Storage And Lifetimes

- K and V are CTA-resident across the complete M loop.
- Producer warps also build one packed MMA-B K fragment per resident 32-row pair from the global K tensor. Complete pairs use a predicated global CuTe partition; incomplete tail pairs are zero-filled by predication. dQ consumers load the four packed words directly, while QK, dV, and dK retain the existing transposed shared-memory path.
- Q, fp16 dO, and preprocess-produced int8 dO use one canonical 32-row resident pair each.
- The dead int32 score fragment is recast in place for P. The dP fragment is overwritten by dS.
- `score_pair_i8` has four slots, one P/`dS*q_scale` pair for each N microtile.
- `WarpScratchStorage` has four slots for the mirrored dS orientation and the half dKV epilogue. The accepted tail prefetch keeps the 32-row fp16 dO pair in a separate allocation because it remains live while dQ consumes the scratch slots.
- Each exact-schedule warp owns one persistent 32-float dV or dK fragment. Source-correlated SASS keeps it in `R5` through `R36`. It is not the source of the launch-bound local traffic.
- Q/int8-dO, K/V, P/`dS*q_scale`, and persistent dKV state overlap in the schedule and cannot be blindly aliased.

The score, dV, dK, and dQ paths use CuTe MMA/copy contracts. Resident Q/K/int8-dO transposed-B reads use the custom CuTe-facing LDSM atom rather than explicit shared transposes. The dKV epilogue uses `SmemCopyAtomdKVC` and `GmemTiledCopydKV` for row-contiguous vector stores.

### Exact Head-64 Schedule

For public `64x64x32x64`, eight warps cover `(M-half, N-tile)` ownership:
- the two M-half owners of an N tile exchange maxima and directly quantize P and `dS*q_scale` at final pair scales.
- adjacent N-tile owners exchange maxima and directly quantize paired `dS*k_scale`.
- all old pair store/reload/re-round rescale passes are eliminated.
- one M-half owner accumulates dV and the other dK.
- only the even N-tile owner executes dQ for its adjacent N pair.

The kernel has `__launch_bounds__(256, 2)`. The 128-register limit permits two 256-thread CTAs on SM86. Unbounded compilation uses 146 registers and permits only one CTA, which is slower despite having no local traffic. The resident-K packet cache adds 4,096 bytes of shared storage; its direct-global startup pack avoids the repeated transposed LDSM conversion for dQ and remains within the two-CTA K64 resource budget.

The dQ epilogue uses a 4x8 destination permutation. Two scalar shuffles per output iteration give each atomic instruction four rows by eight contiguous columns. CTA-local fusion accumulates both N pairs before this epilogue, reducing global-reduction sectors from the original `2,097,152` to `524,288`. The exact specialization computes one contiguous lane-specific workspace base per dQ MMA fragment. Its eight atomics then use compile-time offsets. Generic kernels retain the CuTe workspace view.

After dV and dK finish for a pair, odd-N warps 2 and 3 prefetch the next Q/int8-dO/fp16-dO pair while even-N warps execute dQ MMA, permutation shuffles, and atomics; warp 6 loads the next LSE/Delta pair. The Q/dO reuse barrier is at the dKV/dQ boundary, so the prefetch adds no CTA barrier. Separating fp16 dO from the dS/dKV scratch allocation costs 4 KiB but preserves the K64 two-CTA occupancy threshold.

### Higher-Leverage Quantization Rewrite

The current per-thread Q/K scale contract makes the mainloop redo scale-dependent work for every score tile. For each 32x64 pair it currently:
- loads multiple Q and K scales while reconstructing scores.
- computes independent maxima for `P`, `dS*q_scale`, and `dS*k_scale`.
- materializes two distinct INT8 dS operands.
- exchanges and synchronizes those operands before dK and dQ.

The proposed backward-specific contract uses block-scaled Q/K with blocks chosen so each 32-row Q mainloop pair and resident K tile see invariant scale factors. The current 64x128 exploratory CTA uses `QBlock=128` and `KBlock=128`: one Q scale spans four aligned mainloop pairs and one K scale spans the resident CTA tile. The score path remains INT8 QK, and dV continues to use a separately quantized P and INT8 dO. The dK and dQ paths share one quantized dS tile:

```text
score = int8(Q) @ int8(K) * (q_scale * k_scale)
dS_i8, ds_scale = quantize(dS)
dK += int8(dS_i8).T @ int8(Q) * (ds_scale * q_scale * sm_scale)
dQ += int8(dS_i8) @ int8(K).T * (ds_scale * k_scale * sm_scale)
```

This preserves all four INT8 MMA paths and moves the Q/K dequantization factors to the FP32 accumulation epilogues. It should remove one dS max reduction, one dS shared materialization, and one set of scalar scale multiplications per pair, while also reducing the scale exchange surface. The first implementation will leave the forward per-thread path unchanged and have the backward wrapper produce its own block-scaled Q/K workspace. A later API/state-contract change can let the user choose `QBlock`/`KBlock` as an accuracy control, while backward matmul tiles, compute-dtype casts, ownership, and other implementation details are tuned independently with much less effect on accuracy.

The main risk is numerical rather than architectural: block scaling has less range adaptation than the existing per-thread representation, and the backward recomputed scores must remain compatible with the forward LSE. The rewrite is therefore experimental until it passes the existing cosine, relative-error, tail, HND/NHD, and Racecheck gates and improves serial 4096/8192 timing. The Q128/K64 and Q128/K128 candidates both pass the long 4096/8192 NHD/HND matrix without tolerance changes. Q128/K64 has dQ/dK relative error around `4.10-4.20%`, compared with `4.85-4.99%` for Q128/K128. QBlock=32 remains the finer-grained accuracy reference; the next same-matmul comparison adds Q32/K64 beside Q128/K64 to measure its end-to-end cost independently of CTA geometry.

## Active Configuration

The active generated dispatch contains three focused public configurations:

| Public config | CTA M | CTA N | Generated warps | Current implementation |
|---|---:|---:|---:|---|
| `32x64x32x64` | 64 | 64 | 4 | default block-scaled exact internal eight-warp head-64 kernel |
| `128x64x128x64` | 64 | 64 | 4 | coarser-Q comparison on the same internal eight-warp kernel |
| `128x128x128x128` | 64 | 128 | 8 | exploratory internal sixteen-warp head-64 kernel |

The prior `{64,128}` head-dimension cross product and three other public block configurations are intentionally out of the active build during focused optimization. Retrieve them from git before resuming broader coverage.

Current explicit shared-memory/resource figures are:

| Head dim | `32x128x4` | `64x128x8` | `32x64x4` | public `64x64x4` |
|---:|---:|---:|---:|---:|
| 64 shared memory | 37,328 B | 45,792 B | 25,040 B | 25,312 B for exact internal kernel |
| 64 registers | 196 | 164 | 164 | 128 launch-bounded |
| 128 shared memory | 66,000 B | 74,464 B | 41,424 B | 41,424 B |
| 128 registers | 255 | 250 | 252 | 250 |

The retained focused exact K64 kernel now uses `128` registers and `33,088 B` dynamic shared memory after tail prefetch and the provisional resident-K packet cache. The exploratory K128 kernel uses `128` registers, `49,536 B`, and one resident CTA.

The seven generic variants had zero stack/local memory in the last normal resource build. Refresh those figures only when normal generation is rebuilt. Do not infer them from a focused link containing stale generic objects.

## Build And Validation Workflow

Focused iteration builds only the active public configuration and leaves stale generated source files in place:

```powershell
$Env:SAGEATTN_CUTLASS_BWD_FOCUSED_BUILD = '1'
$Env:MAX_JOBS = '4'
python setup.py build_ext --inplace
```

Use `MAX_JOBS=4` rather than `build_ext -j 4` on this Windows setup. The latter parallelizes both extension modules, launches two Ninja processes in the same object directory, and can fail with object-file `Permission denied` races.

To restore normal generation and build all eight backward objects:

```powershell
Remove-Item Env:SAGEATTN_CUTLASS_BWD_FOCUSED_BUILD -ErrorAction SilentlyContinue
$Env:MAX_JOBS = '4'
python setup.py build_ext --inplace
```

Focused correctness and Racecheck commands:

```powershell
python -m pytest -q tests/test_sagebwd_cutlass.py -k '64-64 and config3 or 65-64 and config3'

& 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\compute-sanitizer.bat' `
  --tool racecheck --error-exitcode 1 `
  python -m pytest -q `
  'tests/test_sagebwd_cutlass.py::test_sagebwd_cutlass_config_matches_flashattention[65-64-HND-config3]'
```

Before broader production use, restore the multi-configuration generator/dispatch from git, build the resulting objects with four workers, run the complete correctness matrix and Racecheck, and verify the expected generated source count.

## Latest Results

### Focused Timing

SM86 laptop RTX 3080 Ti, batch 1, 16 heads, NHD, head dimension 64, public Q32/K64, kernel-only mode, 30 warmups, 100 repeats:

| Layout | Sequence | Sage median | Flash median | Sage/Flash speedup |
|---|---:|---:|---:|---:|
| NHD | 4096 | 5.9443 ms | 4.5763 ms | 0.770x |
| NHD | 8192 | 23.4593 ms | 17.5538 ms | 0.748x |
| HND | 4096 | 5.8935 ms | 4.4969 ms | 0.763x |
| HND | 8192 | 23.2125 ms | 17.3123 ms | 0.746x |

These are serial 30-warmup/100-repeat kernel-only measurements for the predicated global resident-K packet candidate. End-to-end timing and a same-build K64/K128 comparison remain pending. The earlier post-prefetch NHD comparison kept K64 ahead of K128 by a small margin: `6.1378` versus `6.1394 ms` at 4096 and `24.0205` versus `24.0850 ms` at 8192.

A previous serial thermal diagnostic showed approximately 2.3% timing drift as GPU temperature, power, and clocks varied. Treat sub-2% differences as directional unless supported by profile counters or a controlled A/B rebuild.

The earlier full requested matrix predates the exact eight-warp path. It showed `64x64x32x64` generally best for head-64 and `128x32x32x32` generally best for head-128, with Sage/Flash speedups around `0.41-0.57x` and `0.44-0.48x`. Do not use those absolute times as the current head-64 baseline. Rerun the matrix only after the focused head-64 target beats FlashAttention or focused optimization is stopped.

### Focused NCU Profile

Profile shape: sequence 512, batch 1, 16 heads, NHD, ten warmups.

| Metric | Hardware-reduction baseline | Predicated global K packet |
|---|---:|---:|
| Duration | 202.016 us | 181.248 us |
| Instructions | 11.069M | 10.976M |
| Registers/thread | 128 | 128 |
| Dynamic shared memory | 24,896 B | 33,088 B |
| Local-load sectors | 458,752 | 196,608 |
| Local-store sectors | 40,960 | 8,192 |
| Global-reduction sectors | 524,288 | 524,288 |
| Active warps/scheduler | about 3.6 | 3.57 |
| Eligible warps/scheduler | 0.45 | 0.51 |

The packet candidate reports `24.05%` barrier stalls, `5.07%` long-scoreboard stalls, and `9.40%` short-scoreboard stalls. INT8/FP16 tensor activity is `8.87%/4.44%`. It reduces startup conversion work relative to the shared-LDSM cache, but the 4 KiB packet allocation reintroduces limited compiler local traffic; long-shape timing, not the 512-token profile alone, determines retention.

Preprocessing and dQ conversion contribute roughly 0-2% of end-to-end time. Their CUTLASS-vs-Triton results vary by shape and do not justify separate optimization while the fused kernel is nearly 2x behind FlashAttention.

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
- Added focused source generation through `SAGEATTN_CUTLASS_BWD_FOCUSED_BUILD=1` while preserving normal eight-source generation.
- Added two-dimensional eight-warp ownership, direct final-scale quantization, a 128-register launch bound, 4x8 dQ ownership, and the lane-base contiguous dQ atomic address.
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
| Resident-K MMA-B packet cache, direct global startup pack | Producer warps retain the normal async K tile for QK, then form the dQ K fragment from a predicated global MMA-B partition once per resident K pair. dQ consumers load four cached words instead of repeating the transposed LDSM conversion. The candidate uses `33,088 B` shared memory, `128` registers, `10.976M` instructions, `196,608/8,192` local load/store sectors, and `181.248 us` at the 512-token profile shape. Serial long-shape timing is `5.9443/23.4593 ms` NHD and `5.8935/23.2125 ms` HND at 4096/8192. Focused correctness and both K64/K128 odd-tail Racecheck cases are clean; retention is provisional because the shared-LDSM cache is within long-shape timing noise |
| Double-buffer resident Q/dO stages | Separating the aliased fp16 dO and dS/dKV scratch storage restored all four focused NHD/HND and tail correctness cases, but the candidate regressed from `0.3123 -> 0.3369 ms` at 512 and `8.43-8.54 -> 8.90 ms` at 4096. It also increased dynamic shared memory to `37,984 B`, local-load sectors to `774,656`, and instructions to `14.662M` |

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

Do not repeat rejected paths without a new attribution result that changes the tradeoff.

## Remaining Bottlenecks

- Long-sequence gap: current Sage is approximately `1.35-1.40x` slower than FlashAttention at 4096/8192.
- Synchronization: tail prefetch lowers barrier stalls to `24.01%`, but the exact loop still synchronizes after scale exchange, quantized-operand materialization, and gradient consumption.
- Transposed INT8 B loads: imported-source counters attribute about `884,736` warp instructions at the integer shuffle intrinsic. The custom Q/K/int8-dO B loader performs eight integer shuffles plus four byte permutations per LDSM copy and is repeated across dV, dK, and dQ consumers.
- Scalar/vector overhead: score/dS reconstruction, conversion, bounds predicates, dQ permutation shuffles, and atomic address/issue work remain material around the five MMA paths.
- dQ reduction traffic: CTA-local fusion and 4x8 ownership reduce sectors to `524,288`, but atomics remain a serialized dependency chain.
- The resident-K packet cache reduces repeated dQ operand preparation, but its additional 4 KiB allocation reintroduces `196,608/8,192` local load/store sectors in the profile. The shared-LDSM cache and direct-global pack are within long-shape timing noise; source attribution and a same-build A/B remain the deciding evidence.
- FlashAttention dQ comparison: its normal sequence-parallel backward also atomically accumulates each KV CTA's FP32 dQ contribution into one workspace, then converts dQ in a separate kernel. The relevant transferable ideas are wider head-64 K/V tiles and Q/dO double buffering, not eliminating atomics by assumption. Deterministic FlashAttention uses split dQ planes plus a reduction, but that adds workspace and a reduction launch.
- Double-buffer status: the resident head-64 Q, int8 dO, fp16 dO, and row state pass the focused correctness gate after separating fp16 dO from the dS/dKV scratch allocation, but the resource and timing regression keeps the candidate rejected.
- Head-128: it remains on the generic high-register path and is intentionally not being optimized until head-64 wins.

## Ordered Next Work

- Use the predicated-global packet profile (`23.98%` barrier stalls, `9.37%` short-scoreboard stalls, `8.84%` INT8 and `4.42%` FP16 tensor activity) as the current scheduler baseline, while tracking its limited local traffic.
- Use FlashAttention's head-64 SM80 schedule as the structural reference: its normal dQ path still uses FP32 atomics, while its wider K/V tile and Q/dO double buffering reduce the mainloop's synchronization cost.
- Compare the 16-warp `64x128`, `QBlock=32`, `KBlock=128` candidate directly with the retained 8-warp `64x64`, `QBlock=32`, `KBlock=64` kernel in one focused build. The wider candidate preserves one persistent dK or dV fragment per warp, uses one 32x128 dS scale, splits the four head-dimension blocks across eight dQ owner warps, and accumulates all four N32 pairs before each atomic epilogue. Retain both with a sequence-length dispatch only if the wider tile's long-shape gain is stable enough to justify its short-shape regression.
- Complete the resident-K packet decision with a same-build A/B of direct-global packing versus the shared-LDSM fallback, including K64/K128 resource attribution and serial NHD/HND 4096/8192 timing.
- Use fresh imported-source attribution on the retained packet path. The remaining candidates are repeated shuffle/conversion work, scalar score/dS reconstruction, dQ atomic address/issue work, and synchronization; another broad layout rewrite needs a measured source-level target.
- Reduce synchronization without changing arithmetic. Four-warp named exchange barriers are rejected. Retain double buffering only if it lowers barrier/short-scoreboard stalls without increasing local traffic or losing occupancy.
- Reduce scalar quantization and conversion instructions only after the operand-loader probe. Ampere `redux.sync.max.u32` is retained; cooperative shared-load and shared-atomic CTA dS reductions are rejected.
- Extend overlap from asynchronous copies to operand preparation only when producer warps can wait on their own copies and publish fully packed fragments before the existing end-of-tail barrier. Do not add a second per-pair CTA barrier.
- Revisit dQ accumulation only with an end-to-end workspace/launch accounting: split planes, a separate reduction, or different ownership must beat the retained 4x8 atomic path in total time.
- Continue serial 512/4096 timing after resource/profile gates. Do not run concurrent benchmarks or profiles.
- Once head-64 beats FlashAttention, restore the removed generator/dispatch and other kernel paths from git, rebuild with four Ninja workers, run the complete correctness/Racecheck gates, and refresh the full `{heads 16/32} x {D 64/128} x {S 4096/8192}` matrix before broader optimization.

## Deferred Work

- Optimizing block sizes other than head-64 `64x64x32x64` before the focused target beats FlashAttention.
- Decoupling quantization tile sizes from matmul tile sizes.
- Decoupling forward and backward tile sizes.
- Backward autotuning and trainable/compile wrappers.
- GQA/MQA, causal attention, variable length, and additional dtypes.

## Retention Gate

Retain an optimization only when all applicable checks pass:
- FlashAttention-referenced correctness, including HND/NHD and tail cases.
- Compute Sanitizer Racecheck.
- serial warmed 4096- and 8192-token timing, with thermal drift considered. The 512-token shape is profile context only and is too noisy for retention decisions.

Also check the following, while using the overall speed as the optimization target:
- registers, shared memory, stack/local memory, and occupancy.
- global atomic/reduction sectors.
- shared conflicts per wavefront and synchronization behavior.

## Key Artifacts

- Benchmark harness: `bench/bench_sagebwd_cutlass.py`
- Focused profile harness: `build/profile_sagebwd_once.py`
- Latest benchmark CSVs: `build/bench_sagebwd_q32_k64_packed_k_predicated_global_serial_8192_4096.csv`, `build/bench_sagebwd_q32_k64_packed_k_predicated_global_serial_hnd_8192_4096.csv`
- Latest profile CSV: `build/ncu_sage_hd64_q32_k64_packed_k_predicated_global.csv`
- Earlier focused benchmark CSVs: `build/bench_sagebwd_coord_mapping.csv`, `build/bench_sagebwd_coord_mapping_repeat.csv`, `build/bench_sagebwd_coord_mapping_512.csv`
- Earlier focused profile CSVs: `build/ncu_sage_hd64_coord_mapping.csv`, `build/ncu_sage_hd64_coord_mapping_final.csv`
- Previous launch-bound profile: `build/ncu_sage_hd64_2dwarp8_lb2.csv`
- Runtime tests: `tests/test_sagebwd_cutlass.py`
- Transposed-copy test: `tests/cuda/test_qattn_cutlass_bwd_int8_transposed_copy.cu`

# QAttn CUTLASS Backward Kernel Plan

## Status Snapshot

The correctness-first CuTe migration is complete. Optimization is currently limited to the public head-64 `64x64x32x64` configuration. That public four-warp instantiation selects an internal eight-warp `2 M-halves x 4 N-microtiles` kernel. All other generated configurations keep the validated generic schedule.

The latest accepted focused kernel adds lane-base contiguous dQ atomics, a lane-local row-state cache, and exact accumulator-coordinate mapping to the launch-bounded eight-warp schedule. On the SM86 test GPU it:
- passes the four focused head-64 NHD/HND and tail cases.
- passes Compute Sanitizer Racecheck with `0 hazards` (`0 errors`, `0 warnings`).
- uses `128` registers/thread and `25,312` bytes of dynamic shared memory.
- executes `14.127M` instructions at the 512-token profile shape.
- reports `327,680` local-load sectors, `40,960` local-store sectors, and `1,048,576` global-reduction sectors.
- measures `0.3123 ms` at 512 tokens and `8.43-8.54 ms` at 4096 tokens in serial 30-warmup/100-repeat runs.

FlashAttention measured `0.4690 ms` at 512 and `4.58-4.61 ms` at 4096 in those runs. Sage remains approximately `1.83-1.86x` slower at 4096 tokens. The fused main kernel, not preprocessing or allocation, remains the performance blocker.

Normal source generation is restored to exactly eight backward instantiations in the worktree. A normal eight-object rebuild and the full 32-case suite have not yet been rerun after the latest focused optimizations. They remain a retention gate once the focused configuration beats FlashAttention or focused work is otherwise concluded.

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
- Q, fp16 dO, and preprocess-produced int8 dO use one canonical 32-row resident pair each.
- The dead int32 score fragment is recast in place for P. The dP fragment is overwritten by dS.
- `score_pair_i8` has four slots, one P/`dS*q_scale` pair for each N microtile.
- `WarpScratchStorage` has four slots for paired `dS*k_scale`. The same allocation aliases fp16 dO before dS materialization and the half dKV epilogue after the M loop.
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

The kernel has `__launch_bounds__(256, 2)`. The 128-register limit permits two 256-thread CTAs on SM86. Unbounded compilation uses 146 registers and permits only one CTA, which is slower despite having no local traffic.

The dQ epilogue uses a 4x8 destination permutation. Two scalar shuffles per output iteration give each atomic instruction four rows by eight contiguous columns, reducing global-reduction sectors from `2,097,152` to `1,048,576`. The exact specialization computes one contiguous lane-specific workspace base per dQ MMA fragment. Its eight atomics then use compile-time offsets. Generic kernels retain the CuTe workspace view.

## Generated Configurations

Normal generation produces the cross product of head dimensions `{64,128}` and these four public configurations:

| Public config | CTA M | CTA N | Generated warps | Current implementation |
|---|---:|---:|---:|---|
| `128x64x32x64` | 32 | 128 | 4 | generic |
| `128x64x16x64` | 64 | 128 | 8 | generic |
| `128x32x32x32` | 32 | 64 | 4 | generic |
| `64x64x32x64` | 64 | 64 | 4 | exact internal 8-warp kernel for head-64. Generic for head-128 |

Current explicit shared-memory/resource figures are:

| Head dim | `32x128x4` | `64x128x8` | `32x64x4` | public `64x64x4` |
|---:|---:|---:|---:|---:|
| 64 shared memory | 37,328 B | 45,792 B | 25,040 B | 25,312 B for exact internal kernel |
| 64 registers | 196 | 164 | 164 | 128 launch-bounded |
| 128 shared memory | 66,000 B | 74,464 B | 41,424 B | 41,424 B |
| 128 registers | 255 | 250 | 252 | 250 |

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

Before production retention, unset the focused flag, regenerate/build all eight objects with four workers, run all 32 cases, run Racecheck, and verify exactly eight generated backward sources.

## Latest Results

### Focused Timing

SM86 laptop RTX 3080 Ti, batch 1, 16 heads, NHD, head dimension 64, public `64x64x32x64`, kernel-only mode, 30 warmups, 100 repeats:

| Sequence | Sage median | Flash median | Sage/Flash |
|---:|---:|---:|---:|
| 512 | 0.3123 ms | 0.4690 ms | 1.502x |
| 4096, run 1 | 8.4315 ms | 4.6075 ms | 0.546x |
| 4096, run 2 | 8.5417 ms | 4.5798 ms | 0.536x |

A previous serial thermal diagnostic showed approximately 2.3% timing drift as GPU temperature, power, and clocks varied. Treat sub-2% differences as directional unless supported by profile counters or a controlled A/B rebuild.

The earlier full requested matrix predates the exact eight-warp path. It showed `64x64x32x64` generally best for head-64 and `128x32x32x32` generally best for head-128, with Sage/Flash speedups around `0.41-0.57x` and `0.44-0.48x`. Do not use those absolute times as the current head-64 baseline. Rerun the matrix only after the focused head-64 target beats FlashAttention or focused optimization is stopped.

### Focused NCU Profile

Profile shape: sequence 512, batch 1, 16 heads, NHD, ten warmups.

| Metric | Launch-bounded checkpoint | Current coordinate-mapped row state |
|---|---:|---:|
| Duration | about 280 us | 260.800 us |
| Instructions | 15.112M | 14.127M |
| Registers/thread | 128 | 128 |
| Dynamic shared memory | 25,312 B | 25,312 B |
| Local-load sectors | 802,816 | 327,680 |
| Local-store sectors | 61,440 | 40,960 |
| Global-reduction sectors | 1,048,576 | 1,048,576 |
| Active warps/scheduler | 3.58 | 3.62 |
| Eligible warps/scheduler | 0.44 | 0.45 |

The current coordinate-mapped profile reports `32.83%` barrier stalls, `15.75%` short-scoreboard stalls, `3.89%` long-scoreboard stalls, `6.16%` INT8 tensor activity, and `3.08%` FP16 tensor activity. Refresh bank-conflict and source-attribution metrics after each further retained change.

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
| Exact-path row-state cache | Caches the two lane-owned Q-scale/LSE/Delta rows per materialization phase; `14.428M -> 14.169M` instructions, `8.586-8.604 ms` at 4096, correctness and Racecheck clean |
| Exact-path accumulator-coordinate mapping | Replaces identity-view coordinate extraction in the three scalar materialization loops with the verified lane/fragment mapping; `720,896 -> 327,680` local-load sectors, `65,536 -> 40,960` local-store sectors, `8.43-8.54 ms` at 4096, correctness and Racecheck clean |

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
| Flash-style `exp2f` softmax rewrite | saved `121K` profile instructions but raised local loads to `868,352`; 4096 timing was neutral at `8.583 ms`, so the spill tradeoff is rejected |
| Four-value K-scale cache in both scalar phases | reduced instructions to `13.935M` and profile duration to `261.7 us`, but raised local loads to `933,888`; 4096 timing was neutral at `8.569 ms` |
| Four-value K-scale cache only for dS·K quantization | reduced instructions to `14.032M` and local loads to `671,744`, but two 4096 runs regressed to `8.681-8.710 ms` |
| Naive raw dQ pointer per atomic | local loads fell to `393,216`, but address arithmetic raised instructions to `16.676M` and timing to `9.57 ms` |
| Shared-memory padding and several isolated swizzle/copy-width changes | neutral or worse conflict-per-wavefront, resources, or timing |
| Register-fed dQ operand and packed dS stores | reduced some raw conflicts but worsened conflict rates or store traffic |

Do not repeat rejected paths without a new attribution result that changes the tradeoff.

## Remaining Bottlenecks

- Long-sequence gap: current Sage is still approximately 1.83-1.86x slower than FlashAttention at 4096.
- Synchronization: the current full profile attributes `32.79%` of stalls to barriers. The exact loop still has CTA synchronization after staging, scale exchange, quantized-operand materialization, and gradient consumption.
- Launch-bound local traffic: exact coordinate mapping reduces the profile to `327,680` local-load and `40,960` local-store sectors. Remaining traffic is concentrated in address/view state and dQ atomic address state.
- Instruction overhead: current Sage executes `14.127M` instructions at the 512 profile shape. Scalar quantization, max reductions, address/view setup, and atomics dominate around the five MMA paths.
- dQ reduction traffic: 4x8 ownership halves sectors, but `1,048,576` reduction sectors remain and still serialize global accumulation.
- Mainloop pipeline depth: Q/dO is single-buffered. FlashAttention overlaps more staging and has mature head-specific phase layouts.
- Head-128: it remains on the generic high-register path and is intentionally not being optimized until head-64 wins.

## Ordered Next Work

- Use the refreshed coordinate-mapped profile (`32.83%` barrier stalls, `15.75%` short-scoreboard stalls, `6.16%` INT8 and `3.08%` FP16 tensor activity) as the current scheduler baseline.
- Compare the current mainloop and dQ ownership with FlashAttention's head-64 SM80 schedule, then select a schedule-level change with enough upside to justify focused implementation.
- Use fresh line-info to isolate the remaining repeated local values. Change one address/view lifetime at a time. Broad view reconstruction is already rejected.
- Reduce synchronization without changing arithmetic. Four-warp named exchange barriers are rejected; evaluate Q/fp16-dO/int8-dO double buffering only if two-CTA occupancy remains possible.
- Reduce scalar quantization and reduction instructions. A custom packed conversion/copy atom is justified only if SASS attribution shows existing CuTe operations cannot express the required packed flow.
- Revisit dQ accumulation only with an end-to-end workspace/launch accounting: split planes, a separate reduction, or different ownership must beat the retained 4x8 atomic path in total time.
- Continue serial 512/4096 timing after resource/profile gates. Do not run concurrent benchmarks or profiles.
- Once head-64 beats FlashAttention, restore normal generation, build all eight objects with four Ninja workers, run all 32 correctness cases and Racecheck, refresh the full `{heads 16/32} x {D 64/128} x {S 4096/8192}` matrix, and only then optimize other configurations.

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
- serial warmed 512- and 4096-token timing, with thermal drift considered.

Also check the following, while using the overall speed as the optimization target:
- registers, shared memory, stack/local memory, and occupancy.
- global atomic/reduction sectors.
- shared conflicts per wavefront and synchronization behavior.

## Key Artifacts

- Benchmark harness: `bench/bench_sagebwd_cutlass.py`
- Focused profile harness: `build/profile_sagebwd_once.py`
- Latest benchmark CSVs: `build/bench_sagebwd_coord_mapping.csv`, `build/bench_sagebwd_coord_mapping_repeat.csv`, `build/bench_sagebwd_coord_mapping_512.csv`
- Latest profile CSVs: `build/ncu_sage_hd64_coord_mapping.csv`, `build/ncu_sage_hd64_coord_mapping_final.csv`
- Previous launch-bound profile: `build/ncu_sage_hd64_2dwarp8_lb2.csv`
- Runtime tests: `tests/test_sagebwd_cutlass.py`
- Transposed-copy test: `tests/cuda/test_qattn_cutlass_bwd_int8_transposed_copy.cu`

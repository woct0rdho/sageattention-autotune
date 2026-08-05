# QAttn CUTLASS Backward Kernel Plan

## Current Design And Results

### Status

The focused implementation targets head-dimension 64, fp16, non-causal fixed-length attention on SM86. Q/K quantization uses QBlock=32 and KBlock=64. Backward quantization geometry and CTA geometry are independent compile-time choices.

The worktree currently generates exactly two backward configurations:

| Config `(QBlock,KBlock,CtaM,CtaN)` | Generated template | Role |
|---|---|---|
| `(32,64,64,64)` | `<64,64,64,4,32,64>` | K64 controlled reference |
| `(32,64,64,128)` | `<64,64,128,8,32,64>` | K128 selected implementation and public default |

`sageattention/cutlass_bwd.py` selects K128 through `_BWD_CONFIG = (32, 64, 64, 128)`. K64 remains in `_BWD_CONFIGS` for same-build controls. There is no active source candidate beyond the retained K128 schedule. The format and geometry candidates below are planning-only until they have a separate generated instantiation, correctness evidence, and a full long-shape comparison.

`setup.py` emits only `arch=compute_86,code=sm_86`. Existing filenames and module names containing `sm80` are legacy identifiers; they do not indicate a built SM80 image. Sage's native-SM86 image materially improved its earlier SM80-image profile on the NVIDIA GeForce RTX 3080 Ti Laptop GPU.

The installed FlashAttention 2.9.1 extension has no `sm_86` cubin, so its active backward reference uses the wheel's `sm_80` image. A targeted compile of the exact current-source specialization found no material SM80/SM86 code-generation delta, so the installed wheel remains the Flash reference; a whole-package FlashAttention rebuild is not justified.

### Supported Contract

- Inputs and outputs: fp16 Q, K, V, output, and dOutput; float32 LSE.
- Layouts: NHD and HND.
- Attention: dense, non-causal, and fixed length.
- Focused padded head dimension: 64 only.
- Sequence lengths: arbitrary positive lengths under the current int32 parameter and stride contract.
- Q, K, V, output, and dOutput must have matching shapes and head counts.
- Last-dimension storage is contiguous; wrappers materialize contiguous tensors when needed.
- GQA/MQA, causal attention, variable length, additional dtypes, and head dimension 128 are outside the active generated dispatch.
- The standalone kernel header remains torch-free. Stable-torch validation, allocation, launch, and dispatch remain in their existing layers.

The arithmetic contract is fixed:

| Path | MMA contract |
|---|---|
| QK recompute | INT8 `m16n8k32` |
| dV | INT8 `m16n8k32` |
| dK | INT8 `m16n8k32` |
| dQ | INT8 `m16n8k32` |
| `dP = dO @ V.T` | FP16 `m16n8k16`, FP32 accumulator |

dV, dK, and dQ remain INT8 MMA paths in the retained and default schedule. A lower-bit tensor path is an isolated research format only: it must first prove an SM86 copy-plus-MMA mapping, pass the exploratory accuracy tier, and beat the full pipeline before it can affect the default. The custom transposed INT8 LDSM atom is retained because existing CuTe atoms do not express the required shared-memory operand transformation. New hardware atoms are acceptable only when CuTe cannot express the operation and end-to-end evidence justifies them.

### Quantization Domains

The current operation-specific domains are:

| Operand | Dynamic scale domain |
|---|---|
| Q | `M32 x D64` |
| K | `N64 x D64` |
| int8 dO | `M32 x D16` |
| P | `M32 x N16` |
| dS | `M32 x N64` |

The K128 CTA therefore contains two independent K64/dS domains. CTA-N=128 controls reuse and dQ ownership; it does not change KBlock or merge the two numerical scale domains.

The mathematical reconstruction is:

```text
score = int8(Q) @ int8(K) * (q_scale * k_scale)
dS = P * (dP - Delta) * sm_scale
dS_i8, ds_scale = quantize(dS)
dV += int8(P).T @ int8(dO) * (p_scale * do_scale)
dK += int8(dS_i8).T @ int8(Q) * (ds_scale * q_scale)
dQ += int8(dS_i8) @ int8(K).T * (ds_scale * k_scale)
```

One logical dS quantization is reused by dK and dQ. Dequantization and cross-domain dQ combination occur in FP32.

### Quantization And Geometry Search

`Q32/K64` with the eight-warp `M64xN128` CTA is the controlled performance baseline, not a claim that it is the global optimum. The current specialized body statically requires `Q32/K64`, `M64xN128`, and eight warps; adding a tuple to the generator alone cannot test a new format. Each candidate below needs a separate specialized body so the retained baseline and its correctness controls remain intact.

| Candidate | Proposed domains and CTA | Measured work it can remove | Constraint and status |
|---|---|---|---|
| Baseline | `Q=M32xD64`, `K=M64xD64`, `dS=M32xN64`; `M64xN128`, 8 warps | None; reference for every result | Retained default |
| Primary format probe | `Q=M32xD64`, `K=M128xD64`, `dS=M32xN128`; keep `P=M32xN16`; `M64xN128`, 8 warps | Collapses two K/dS domains into one. Tensor MMA count is unchanged, but one dQ scaled partial/accumulator path and one K-scale path disappear. | Measured and rejected with exact dS scaling: the eight-warp dS maximum dependency outweighed the saved dQ scale work. |
| Paired M-domain probe | `Q=M64xD64`, `K=M128xD64`, and an intentionally staged `dS=M64xN128`; `M64xN128`, 8 warps | Can share Q scale and dS scale across two adjacent M32 pairs, but only if the schedule directly quantizes or retains both pairs. | Do not implement as a Q64-only tuple. It needs a two-M-pair lifetime redesign. |
| Larger CTA feasibility only | `M128xN128` or `M64xN256`, likely 16 warps | May amortize a scale/handoff only when paired with the M64 or N128 format change. | Not a blind CTA search. It requires a new storage contract, likely raises the one-CTA resource cost, and dQ reduction volume already matches Flash. |

`Q64/K64` by itself is deliberately not a primary candidate. It reduces external Q-scale generation, but it neither removes an in-main P/dS pass nor collapses the K128 CTA's two K64/dS domains. The fresh exact-scale `Q32/K128` single-domain body is now rejected; a paired M64/K128 design remains eligible only if a direct or predicted dS format removes the exact eight-warp maximum dependency.

The geometry comparison is intentionally asymmetric. For every shape and layout, FlashAttention runs its installed dispatch-selected backward specialization, while Sage reports the best stable median across the predeclared Sage candidates. The report records the selected Flash kernel name, each Sage configuration, and the winning Sage configuration. Matching nominal tile dimensions across implementations is neither required nor desirable.

### On-The-Fly Quantization Program

The native source counters show that the problem is the exact max-based P/dS quantization protocol, not simply how four converted bytes are packed. A scalar float-to-int conversion remains necessary for an arbitrary per-tile scale on SM86; the prior explicit F2IP packing experiment did not replace that work. The first candidates must remove a pass, a reduction, a reciprocal, or a shared handoff.

- Fixed-scale P, dynamic dS is closed for a global linear INT8 scale. Full-domain scans and dV emulation showed that a fixed multiplier zeros nearly all long-shape P values and collapses dV cosine, even though conversion-level saturation is cheap. Retain local P scaling unless a new multi-level or residual representation preserves the tail with fewer total instructions.
- Exact power-of-two dS is retained as an optional K128 policy, not the default. It keeps the exact maximum and second conversion pass, derives a symmetric block exponent, and replaces one reciprocal with bit-level inverse construction. Paired long timing wins by about `0.6-1.0%`, but executed instructions rise and dQ/dK relative error approaches the `0.06` schedule-preserving limit.
- Direct clipped or predicted dS is next. Use a fixed, running, or cheaply predicted dS scale so P and dS can both be quantized during reconstruction. This is the only route that removes the dS maximum/reduction/barrier/second pass together. It is deliberately an approximate format experiment: record saturation, zero rate, and scale histograms, and keep it behind a separate dispatch path.
- Do not retry packed conversion in isolation. An inline `cvt.pack` or vector conversion is acceptable only if a standalone SASS comparison proves it replaces scalar F2IP/conversion instructions rather than adding packing around them. It is a complement to a one-pass format, not the primary optimization.

Q/K block changes are separately timed in the full backward pipeline because `per_block_int8` runs outside the fused main kernel. A coarser Q or K block is useful only if its external quantization saving, fewer scale domains, and any in-main handoff saving outweigh its accuracy loss. dO quantization is likewise measured separately from the main-kernel attribution before it becomes a target.

### Kernel Pipeline

The backward operation has three kernels:

```text
preprocess, one CTA per adjacent 32-row pair:
  Delta = sum(output * dOutput)
  quantize dOutput and emit its D16 scales
  clear the float dQ accumulation workspace

fused KV-owned main kernel:
  keep one K/V CTA tile resident
  for each 32-row Q/dOutput pair:
    recompute QK with INT8 MMA
    compute dP with FP16 MMA
    reconstruct P and dS
    quantize P and one logical dS
    accumulate dV, dK, and dQ with INT8 MMA
  write owned dK/dV directly
  atomically accumulate dQ

postprocess:
  convert the float dQ workspace to fp16
```

Each main-kernel CTA owns one KV tile, writes dK/dV without a global reduction, and contributes to dQ through atomics. The K128 geometry halves dQ reduction sectors relative to K64 at the diagnostic shape.

### Selected K128 Schedule

The selected kernel is a separate M64xN128 eight-warp implementation, not a parameterized copy of the older temporal-eight-warp body.

- Each physical warp owns one N16 tile and processes both temporal M16 halves.
- dK and dV accumulators remain persistent in FP32 across the M loop.
- Four FP16 V MMA-B fragments per warp remain register-resident.
- A predicated direct-global resident-K packet is built once per dQ ownership tile.
- Q and int8-dO are converted once per M pair into scalar word-major MMA-B packets reused by dK/dV.
- The Q/dO packet cache overlays the dead V shared-memory staging region.
- P and dS MMA-A fragments are loaded once per M pair and reused across all four D16 dV/dK tiles.
- Canonical P/dS stores feed dV/dK; a mirrored dS view feeds the retained dQ layout.
- Warps 0, 2, 4, and 6 own dQ. Each combines both K64 domains in FP32 before one N128 atomic epilogue.
- dQ uses the scaled-FP32 shared transpose epilogue.
- Dynamic shared memory above 48 KiB is enabled with `cudaFuncSetAttribute`.
- Launch bounds are `__launch_bounds__(256,1)`.

K128 intentionally accepts one resident CTA. The retained aligned SM86 kernel uses 235 registers per thread, 57,664 bytes of dynamic shared memory, and no compiler-local traffic. Reducing the register count is useful only if it removes work or enables a viable pipeline; it does not improve occupancy by itself.

### K64 Reference

K64 uses a 64x64 CTA with four logical N ownership warps and eight physical launch warps. It remains unchanged as the controlled reference:
- 128 registers per thread and 33,088 bytes of dynamic shared memory.
- Two resident CTAs on SM86.
- Direct-global resident-K packets.
- CTA-local dQ fusion.
- Scaled-FP32 dQ transpose epilogue.
- Aligned and predicated specializations.

K64 is no longer the public default. It remains generated until the next attributed structural decision or an explicit decision to remove the reference.

### Alignment And Tail Rules

- `seq_len % BlockN == 0` selects the aligned specialization.
- Every other sequence length selects the predicated specialization.
- The aligned specialization removes only scalar score/dS and dQ output-row predicates.
- Q/dO/K/V copies, row-state loads, resident-K packet construction, and dK/dV stores remain tail-safe.
- Direct-global K packets are cleared before predicated copies so partial 32-row pairs cannot expose stale bytes.
- Broad predicate removal remains rejected because it increased local traffic and regressed timing.

### Correctness And Accuracy

The retained source passes the expanded focused FlashAttention-referenced matrix (`16 passed`): dynamic K64/K128, optional power-of-two and periodic K128 policies, NHD/HND, and aligned/tail lengths 64/65.

```text
python -m pytest -q tests/test_sagebwd_cutlass.py
16 passed
```

Retained K64 and K128 schedules have passed Compute Sanitizer Racecheck at sequence lengths 512 and 513:

```text
0 hazards displayed (0 errors, 0 warnings)
```

K128 and K64 agree to the printed precision in long-shape comparisons. Against FlashAttention at 4096/8192 in both layouts, the observed K128 ranges are:

| Gradient | Cosine similarity | Frobenius relative error | Maximum absolute error |
|---|---:|---:|---:|
| dQ | `0.999129-0.999166` | `0.040855-0.041763` | `0.004089-0.005981` |
| dK | `0.999144-0.999172` | `0.040711-0.041398` | `0.004242-0.005737` |
| dV | `0.999640-0.999665` | `0.025880-0.026832` | `0.002808-0.004486` |

The schedule-preserving gate remains `1 - cosine < 2e-3`, relative error `< 0.06`, maximum absolute error `< 0.20`, and no NaN/Inf.

### Diagnostic Resources

These values are from native-SM86 aligned sequence-512 captures. The latest retained K128 recapture is used below. This shape is diagnostic; long serial timing governs retention.

| Metric | K64 reference | K128 selected |
|---|---:|---:|
| Physical launch | 256 threads / 8 warps | 256 threads / 8 warps |
| Registers/thread | 128 | 235 |
| Dynamic shared memory | 33,088 B | 57,664 B |
| Resident CTAs on SM86 | 2 | 1 |
| Instructions | 8.797M | 6.609M |
| Local load/store sectors | 188,928 / 28,672 | 0 / 0 |
| Global-reduction sectors | 524,288 | 262,144 |
| Shared-load wavefronts | 2,180,382 | 1,318,912 |
| Shared-store wavefronts | 579,791 | 480,154 |
| Shared load/store conflicts | 296,205 / 60,995 | 32,768 / 23,159 |
| Profile duration | 150.304 us | 161.312 us |
| Barrier stalls | 20.68% | 11.40% |
| Long/short scoreboard stalls | 6.58% / 15.24% | 7.67% / 5.15% |
| Eligible warps/scheduler | 0.51 | 0.38 |

Changing only the image from SM80-generated code to native SM86 changed the original K128 profile from `199.456 -> 165.024 us` and registers from `255 -> 238`. Persistent P/dS MMA-A fragments then changed the accepted profile to `160.896 us`, 235 registers, `6.609M` instructions, and `1.319M` shared-load wavefronts. The final `161.312 us` recapture is normal run-to-run variation of that retained source.

### Long-Shape Performance

These are warmed, serial, kernel-only 100-repeat medians for batch=1, heads=16, and head dimension 64. "Forward" and "reverse" describe configuration/sequence invocation order controls, not attention direction.

Effective dense backward TFLOPS use:

```text
FLOPs = 8 * batch * heads * head_dim * seq_len^2
SageAttention/FlashAttention speed ratio = Sage TFLOPS / Flash TFLOPS = Flash time / Sage time
```

A SageAttention/FlashAttention ratio greater than `1.0` means SageAttention is faster.

| Order | Layout | Seq | K64 ms | K128 ms | K128 TFLOPS | Flash ms | Flash TFLOPS | SageAttention/FlashAttention speed ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| forward | NHD | 4096 | 5.3857 | 4.2957 | 31.995 | 4.7913 | 28.685 | 1.115 |
| forward | NHD | 8192 | 20.2977 | 15.9565 | 34.453 | 17.0926 | 32.163 | 1.071 |
| forward | HND | 4096 | 5.6852 | 4.3648 | 31.488 | 4.6484 | 29.567 | 1.065 |
| forward | HND | 8192 | 20.1165 | 15.8868 | 34.604 | 17.3884 | 31.616 | 1.095 |
| reverse | NHD | 4096 | 5.1809 | 4.1998 | 32.725 | 4.5240 | 30.380 | 1.077 |
| reverse | NHD | 8192 | 20.0714 | 15.8444 | 34.697 | 17.1433 | 32.068 | 1.082 |
| reverse | HND | 4096 | 5.1692 | 4.1088 | 33.450 | 4.5701 | 30.073 | 1.112 |
| reverse | HND | 8192 | 20.2609 | 15.8796 | 34.620 | 17.3644 | 31.660 | 1.094 |

K128 beats K64 in all eight controls by `18.9-23.2%`. It delivers `31.488-34.697` effective TFLOPS and beats paired FlashAttention in all eight controls by a SageAttention/FlashAttention speed ratio of `1.065-1.115`, equivalent to `6.1-10.3%` lower elapsed time.

### Native-SM86 Attribution (Completed)

Matched raw-NCU captures now use the same seeded NHD input (`batch=1`, `heads=16`, `head_dim=64`, `seq_len=8192`), five warmups, one profiled main-kernel launch, and seven replay passes. Flash ran first, then Sage. The captured kernels are Flash's non-causal even-dimension sequence-K-parallel `64x64x128`, eight-warp specialization and Sage's selected `<64,64,128,8,32,64,1>` K128 specialization.

| Main-kernel metric | FlashAttention wheel (`sm_80`) | Sage native `sm_86` | Sage / Flash |
|---|---:|---:|---:|
| Duration | 22.870 ms | 20.284 ms | 0.887x |
| Registers / dynamic shared memory | 255 / 73,728 B | 235 / 57,664 B | - |
| Total instructions | 617.554M | 1,599.230M | 2.589x |
| Tensor instructions | 167.772M HMMA | 33.554M HMMA + 67.109M IMMA | 0.600x |
| Non-tensor instructions | 449.782M | 1,498.567M | 3.332x |
| Tensor-pipe active cycle-time proxy | 44.99% HMMA | 10.15% HMMA + 20.29% IMMA | 0.600x |
| DRAM bytes | 277.191 MB | 286.043 MB | 1.032x |
| Shared load / store wavefronts | 336.069M / 36.333M | 331.743M / 121.020M | 0.987x / 3.331x |
| Global reduction sectors | 67.109M | 67.109M | 1.000x |

The duration sample is a matched profiler observation, not a replacement for the serial 4096/8192 timing controls. It is nevertheless sufficient for attribution: Sage retains the intended three-versus-five tensor-work reduction exactly, and its `0.600x` tensor-active cycle-time proxy is unchanged on the native image. The remaining cost is not HBM-bound: DRAM throughput is only `2.71%` of peak for Flash and `3.15%` for Sage. Sage instead executes `3.332x` the non-tensor instructions, has `3.331x` shared-store wavefronts, and raises L1TEX throughput from `27.43%` to `53.71%` of peak. Its main stalls are MIO (`16.33%`), wait (`16.49%`), math-pipe (`23.69%`), and barrier (`11.76%`); Flash is predominantly math-pipe stalled (`63.06%`). Sage's dQ reduction sectors already exactly match Flash, so a dQ workspace or ownership redesign is not the first response.

Fresh SourceCounters with CUDA/SASS correlation identify the live native image. Source-line counts are attribution weights, not additive totals, because compiler inlining maps one SASS instruction to both helper and caller locations. The direct hotspots are nevertheless clear:
- P/dS reconstruction at lines 1745-1778 carries `268.435M` dynamic instruction weights for score conversion, `expf`, and dS formation; its four quantization call sites at lines 1821-1858 carry `136.315M` more.
- The packed MMA-B loader at line 694 carries `102.760M` instruction weights and `100.663M` shared-load wavefronts, with the largest sampled MIO stall contribution (`50,937` samples).
- FP32 dequantized accumulation costs `134.218M` instruction weights for dV at line 1906, `136.315M` for dK at line 1922, and `76.546M` for dQ at line 2001.
- The retained dQ shared stage and atomic path remain visible at lines 841, 854, and 859, but their global reduction-sector volume is already optimal relative to the Flash main kernel.

The next source-level work is therefore ranked as: (1) remove P/dS reconstruction or quantization work through a representation change that preserves the numerical contract; (2) remove or amortize packed MMA-B load/handoff work without recreating the rejected transposed-load cost; and (3) remove a proven portion of dV/dK/dQ scale application only if it does not increase register pressure or shared stores. Canonical-plus-mirrored dS movement remains part of the first two targets because the raw native kernel still has the `3.331x` shared-store gap. Occupancy, broad dQ ownership changes, generic double buffering, and an HBM-focused change are lower priority.

#### Flash Reference Image Decision

The wheel's active Flash kernel reports 255 registers at runtime. To isolate architecture rather than package/compiler differences, the exact current-source device implementation was wrapped in a single targeted probe and compiled serially for both architectures. The SM80 and SM86 builds took approximately `34.6 s` and `34.7 s`, respectively, well below the five-minute build limit. The probes have no stack frame or spills; SM80 uses 242 registers and SM86 uses 240. Each disassembly has 1,248 printed instruction slots; excluding padding NOPs leaves 1,235 SM80 and 1,237 SM86 instructions. Major counts are unchanged: 160 HMMA, 88 LDSM, 64 STS, and 16 LDGSTS instructions in each image.

The probe's 242/240 register counts are not a replacement for the wheel's 255-register runtime image: they reflect current source and CUDA 13.3 rather than the packaged extension's source/compiler combination. They do establish that a native SM86 Flash cubin would not create a notable architectural advantage for this specialization. Keep the installed Flash wheel for upcoming experiments, use the probes only as architecture evidence, and do not rebuild the whole FlashAttention package.

The activity-based planning proxy remains useful only as a rough overlap model:

```text
speedup = 1 / (0.60 * f + b * (1 - f)), with f ~= 0.45
```

The matched native capture implies `b ~= 1.12`; it is not a wall-time decomposition because tensor and non-tensor work overlap.

#### Route Toward The 1.67x Tensor Limit

With the observed Flash tensor-active fraction `f ~= 0.45` and Sage tensor factor `0.60`, the activity proxy makes the remaining requirement explicit:

| Desired Sage / Flash speed ratio | Required effective non-tensor factor `b` | Reduction from current `b ~= 1.12` |
|---:|---:|---:|
| 1.13x (current matched capture) | 1.12 | - |
| 1.30x | 0.91 | 19% |
| 1.40x | 0.81 | 28% |
| 1.50x | 0.72 | 36% |
| 1.67x tensor-limit proxy | 0.60 | 47% |

Instruction counts do not map linearly to elapsed time, but this rules out minor tuning as a path to the tensor limit. A candidate needs to eliminate a substantial part of P/dS reconstruction/quantization, packet handoff, or scale application; reducing register count, reduction sectors, or one isolated stall category is insufficient.

## Clamp And Fixed-P Reconnaissance

- The original `round_to_int8` helper was an unchecked `__float2int_rn` followed by `int8_t` narrowing. Dynamic P/dS scales normally bounded the input, but there was no source-level `min`, `max`, or clamp in the on-the-fly conversion.
- A standalone SM86 PTX/SASS probe compiled `cvt.rni.sat.s8.f32` successfully. It lowers to one `F2I.S8` instruction, so the retained helper now enforces saturation at the conversion without adding a separate floating-point clamp sequence.
- Corrected full reconstructed-P scans with the actual Sage forward LSE and Q32/K64 backward quantization reached maxima `0.324/0.155/0.172` at `seq=512` and `0.060/0.051/0.043` at `seq=8192` for seeds 0/1/2, with no values above 1. A fixed multiplier of 127 nevertheless zeroed about `99.94%` at `seq=4096` and effectively all values at `seq=8192`.
- A literal fixed multiplier of 127 rounded about `99.9%` of long-shape P values to zero in the range scan. A guarded multiplier near 64 avoids the observed overshoot but loses still more tail precision. dV-only emulation at `seq=2048/4096` measured dynamic local P quantization at about `0.9996` cosine and `2.6-2.7%` relative error, versus fixed multipliers 127/96/64 at `0.52/0.44/0.33` cosine for `seq=2048` and `0.35/0.28/0.19` for `seq=4096`.
- The naive global fixed-P representation is therefore rejected: it removes the local scale reduction by destroying the long-shape dV signal. The useful clamp optimization is conversion-level: keep local dynamic scales, use a single saturating `F2I.S8` conversion, and verify that the bounded baseline produces identical results and no instruction increase.
- The native baseline rebuild used four jobs and passed the focused K64/K128, NHD/HND, aligned/tail matrix (`8 passed in 6.50 s`). INT4 is explicitly deferred and is not part of the active experiment queue.
- The conversion-level saturation experiment replaced the 32 aligned K128 `F2I.FTZ.NTZ` instructions with `F2I.S8`; instruction slots stayed at `2088` with `33` padding NOPs, and resource counts did not change. Focused correctness remained `8 passed`.
- Fresh serial long controls for the saturating helper measured K128 at `4.198/16.155 ms` NHD and `4.180/15.982 ms` HND for 4096/8192. This is neutral within the existing run-to-run spread, so the helper is retained as overflow protection with no claimed speedup. It is a conversion-level clamp, not an added floating-point clamp sequence.
- Exact-max power-of-two dS simulation is viable for a kernel probe. Relative dQ/dK error increased from `3.4-3.8%` for dynamic scaling to `5.0-5.5%`; cosine remained about `0.9985`, zero rate increased from `18-19%` to `24-26%`, and conversion saturation stayed between `2e-6` and `8e-6`. The real candidate must prove that exponent construction removes enough reciprocal/scale work to offset the format loss.
- The real exact-max policy passes the expanded focused matrix (`12 passed`) and the strict schedule-preserving limits through sequence 8192. At 8192, dQ/dK cosine is `0.998340/0.998372`, relative error is `5.768%/5.711%`, maximum absolute error is `0.00587/0.00536`, and dV is identical to the dynamic-dS kernel.
- Aligned SASS remains 235 registers, zero local memory, and `2088` slots. Power-of-two dS removes one `MUFU.RCP` and one `FFMA`, adds one `FMNMX`, two `IADD3`, and two `LOP3`, and raises non-NOP slots `2055 -> 2058`; it is a latency trade rather than an instruction-count reduction.
- Alternating same-process controls measured power-of-two/dynamic paired ratios of `0.994/0.990` for NHD and `0.994/0.990` for HND at 4096/8192, reproduced at `0.994/0.991` and `0.992/0.992` with 200 pairs and another seed. Fixed-clock NCU at 8192 measured `20.294 -> 20.125 ms`, `5.297B -> 5.253B` elapsed SM cycles, and `1.599B -> 1.606B` executed instructions.
- Aligned sequence 512 and odd-tail 513 Racecheck both report `0 hazards, 0 errors, 0 warnings` for the power-of-two specialization.
- Keep this policy as an explicit same-build format reference and composition candidate, with dynamic dS remaining the public default. Its modest speedup does not justify silently spending the accuracy margin, but it proves that reciprocal latency is measurable and provides the control for a direct one-pass dS experiment.
- Periodic predicted-dS simulation is bounded but approximate. A `2x` guarded scale recalibrated every 16 M32 blocks produced sequence-8192 dQ/dK cosine `0.99180-0.99196`, relative error `12.66-12.79%`, zero rate `31.99-32.32%`, and saturation `5.0e-5-5.2e-5` for seeds 0/1. It passes the first format tier but not the schedule-preserving tier; the next kernel probe must show a material removal of dS max/reduction work before this trade is retained.
- The periodic K128 probe passes the expanded focused matrix (`16 passed`) and Racecheck at 512/513 after adding a barrier around the cached domain-scale update. Full-pipeline sequence-8192 dQ/dK relative error is `12.74-13.17%` with cosine `0.99169-0.99186` and maximum absolute error `0.082-0.108` for seeds 0/1; dV remains identical to dynamic dS.
- Corrected alternating controls show periodic/dynamic paired ratios of `0.957/0.949` NHD and `0.955/0.951` HND at 4096/8192. Fixed-clock NCU at 8192 measures `20.294 -> 19.340 ms`, `5.297B -> 5.048B` elapsed SM cycles, and `1.599B -> 1.597B` executed instructions. The branch removes most exact dS max/reduction work without increasing the aligned register or shared-memory footprint.
- Retain periodic dS as an optional approximate K128 reference, with dynamic dS still the public default. Its `4.3-5.1%` main-kernel win is large enough to justify the format tier, and it unlocks direct P/dS-to-dKV handoff experiments; it is not a default promotion because its error exceeds the schedule-preserving gate.
- The mirror-only dS-to-dK mapping probe established an exact synthesized `N16 x M32` view across two adjacent existing dS mirror slots: its loaded MMA-A fragment matched the canonical `sdSPair` reload with zero mismatches for both N-half positions. An isolated `periodic_ds_mirror_dkv` specialization retained the dQ mirror, omitted canonical dS stores, and loaded dK through `AutoVectorizingCopyWithAssumedAlignment<8>` after the existing CTA barrier. It passed the expanded focused matrix (`20 passed` while present), but lost paired kernel-only timing by `1.9%/2.2%` at NHD 4096/8192 and `0.8%/4.5%` at HND 4096/8192. The aligned candidate added 8 SASS slots; the HND variant rose from 235 to 237 registers. Reusing the canonical LDSM atom is not a valid fallback because the synthesized source layout fails CuTe's register-vectorization contract. Remove the specialization and retain only the mapping probe for a future producer/register-direct consumer with a different cost model.

## What We Have Done

### Foundation And Accepted Work

- Built the standalone stable-torch backward wrapper, generated instantiations, and runtime dispatch.
- Migrated QK, dV, dK, dQ, and dP to CuTe/CUTLASS contracts with four INT8 paths and one FP16 path.
- Added and validated `int8_transposed_ldsm_cute.cuh`; its copy-plus-MMA test reports `failures=0`.
- Kept the device header torch-free and separated launch/allocation ownership.
- Decoupled Q/K quantization blocks from backward CTA M/N geometry across the API, generator, dispatch, wrapper, tests, and benchmark labels.
- Corrected generic scale indexing to use actual Q/K quantization blocks.
- Adopted Q32/K64, one logical dS quantization, the KV-owned fused schedule, direct dK/dV stores, and CTA-local dQ fusion.
- Fused Delta, dO quantization/scaling, and dQ clearing into one preprocessing CTA per 32-row pair.
- Added Ampere max reductions, tail-safe Q/dO prefetch, direct-global resident-K packets, aligned specialization, and the scaled-FP32 dQ transpose epilogue.
- Preserved K64 at 128 registers and two resident CTAs while reducing its instructions, local traffic, and dQ reduction volume.
- Implemented the fresh K128 eight-warp schedule with persistent dK/dV, two K64 scale domains, and one N128 dQ epilogue.
- Made V register-resident and overlaid Q/dO packets on dead V staging.
- Retained scalar word-major Q/dO packets after they reduced instructions/shared loads and won all governing long controls.
- Switched to native SM86-only code generation after proving a material compiler and register-allocation difference.
- Cached P/dS MMA-A fragments once per M pair, reducing shared loads without spills or new synchronization and improving all four 8192 controls by `2.5-5.5%`.
- Promoted K128 to the public default after full NHD/HND, 4096/8192, forward/reverse controls.
- Added an optional exact-max power-of-two dS policy for K128. It preserves the strict accuracy gate and improves paired long main-kernel timing by `0.6-1.0%`; dynamic dS remains the default.
- Added an optional periodic predicted-dS policy for K128. It recalibrates every 16 M32 blocks with a `2x` guard, passes the first approximate accuracy tier and Racecheck, and improves paired long main-kernel timing by `4.3-5.1%`; dynamic dS remains the default.

### Completed Decisions

The following paths are closed unless new source/SASS evidence or a new hardware representation changes their cost model:

| Experiment family | Decision |
|---|---|
| Native explicit F2IP packing | Rejected. Scalar source already contracts on SM86; forced grouping measured `167.776 us`, `6.953M` instructions, and 236 registers versus `165.024 us`, `6.658M`, and 238 for the native original. |
| Hoisted dQ scales | Rejected. Instructions fell `6.609M -> 6.568M`, but duration changed only `160.896 -> 160.320 us` and registers rose `235 -> 238`. |
| Early FP16 dO prefetch | Rejected. It passed correctness, long accuracy, and Racecheck, but raised registers/instructions/shared-store conflicts; three of four fresh NHD/HND long controls regressed. |
| Warp-local P/dS-to-dKV handoff | Rejected. It was race-free, but profile duration became `163.104 us`; NHD regressed `1.1%` at 4096 and `2.9%` at 8192. |
| Lane-major 128-bit Q/dO packets | Rejected. Better short counters did not produce stable long timing; shared-store conflicts increased. |
| Canonical dS direct reads | Rejected. Removing the mirror added transposed reconstruction, shuffles, shared-load conflicts, and worse long timing. |
| Transposed `dQ.T = K.T @ dS.T` | Rejected. It passed correctness and Racecheck and removed mirror stores, but canonical transposed reads produced `163,840` load conflicts and six of eight long controls regressed. |
| All-eight-warp dQ ownership | Rejected. The diagnostic profile improved, but three of four normalized 8192 controls regressed and the fourth tied. |
| Interleaved `{P_i8,dS_i8}` packet | Rejected analytically for the current loader. It saves the mirror store but doubles each transposed dQ read, recreating the rejected canonical-loader cost. |
| FP16x2 P/dS compaction | Rejected. Raw packing underflowed; normalized variants passed focused correctness but stayed at 255 registers and worsened instructions, stores, conflicts, and duration. |
| Serialized/split dKV schedules | Rejected. They introduced a stack frame or extra synchronization without a long-shape win. |
| Temporal eight-warp K128 body | Rejected. Its old dS orientations and unfused dQ reached `17.283M` instructions and `551.040 us`; the selected K128 body is a new implementation. |
| dQ staging alternatives | Rejected. Raw-INT32, FP16, row-major, affine-XOR, and no-post-fence variants lost on timing, traffic, conflicts, or correctness. |
| Broad aligned predicate removal | Rejected. It increased local traffic and regressed long timing. |
| Full Q/dO double buffering | Rejected for the tested schedule. It introduced local traffic and regressed long timing; low native scoreboard stalls do not support retrying it without new evidence. |
| Coarser Q/K blocks | Exact-scale Q32/K128 is rejected. It passed the 12-case focused matrix and long accuracy, but lost every kernel-only and end-to-end NHD/HND 4096/8192 forward/reverse control. Q64/K128 remains open only as a direct/predicted-dS two-M-pair redesign. |
| Exact-max power-of-two dS | Retained as an optional K128 policy. It removes one reciprocal and wins paired long timing by `0.6-1.0%`, while increasing executed instructions `0.39%` and raising dQ/dK relative error to about `5.7%`. It is not the public default. |
| Periodic predicted dS max-skip | Retained as an optional K128 policy. It skips exact dS max/reduction on 15 of every 16 M32 blocks, wins paired long timing by `4.3-5.1%`, and stays spill-free, but dQ/dK relative error is `12.7-13.2%`; it remains outside the default path. |
| Mirror-only dS-to-dK handoff | Rejected as an isolated periodic policy. The exact two-slot mirror mapping passed focused correctness, but ordinary vectorized mirror loads added SASS/register overhead and lost all four paired long kernel-only controls. The mapping remains a probe artifact; no generated policy is retained. |

The fresh Q32/K128 aligned image retained 235 registers and zero local traffic while reducing SASS slots `2088 -> 2056` and FFMA instructions `114 -> 98`; `FMNMX` increased `47 -> 53`. Kernel-only regressions ranged from `0.2%` to `3.6%`, and end-to-end regressions from `0.8%` to `4.0%`. Halving external K-scale generation did not recover the longer exact dS reduction dependency, so the specialization and generated dispatch were removed.

### Coupled Revisit Candidates

The decisions above apply to their isolated source implementations. The following are worth revisiting only when the named representation change removes the condition that made the original experiment lose. Each is a new candidate, not a restoration of the prior source.

| Coupled candidate | Why the isolated result is no longer decisive | Required proof before long timing |
|---|---|---|
| Direct-register P/dS to dV/dK plus mirror-only dS for dQ | The rejected warp-local handoff still paid for dynamic P/dS scaling and canonical-plus-mirrored storage. Fixed P or a direct dS format can let dV/dK consume producer fragments while only dQ materializes the known-good mirrored dS layout. | CuTe copy-plus-MMA mapping, no spills, source/SASS evidence that canonical P/dS stores and reloads disappear, and preserved dQ race safety. |
| Q32/K128 plus direct or predicted dS | The isolated exact-scale single-domain body removed dQ FFMA work but lost because all eight warps remained on the dS maximum dependency. A one-pass dS format can remove that newly identified failure condition. | Remove the exact eight-warp maximum/barrier path, retain one-domain dQ scaling, show no spills, and beat Q32/K64 in the full serial matrix. |
| Q64/K128 two-M-pair staging plus Q/dO double buffering | The rejected buffer raised live state without amortizing quantization. A true M64 domain can amortize Q scale and allow a direct-quantized pair to overlap the next pair's staging. | Storage/liveness prototype first; no local traffic, bounded registers, and an explicit removal of an M-pair quantization/reduction pass. |
| Producer-specialized Q/dO MMA-B handoff plus direct P/dS consumers | The rejected lane-major packet still served the same broad consumer pattern and increased conflicts. A direct dV/dK path can reduce consumers and may justify a new packet/copy mapping. | Standalone producer-to-consumer copy/MMA test and source evidence that line-694 loader instructions or MIO stalls fall rather than move elsewhere. |
| FP16 dO prefetch after a direct-quantization register reduction | The rejected prefetch was evaluated with the current high-live-range P/dS path. It can be reconsidered only if the new format releases registers and wait stalls remain material. | Register count no higher than the direct-quant baseline, zero local sectors, and a same-build timing win. |

Transposed dQ, all-eight-warp dQ ownership, FP16x2 P/dS compaction, broad predicate removal, and the old temporal K128 body remain closed. The direct-register candidate preserves the mirrored dS direction instead of retrying the rejected canonical/transposed dQ reads. Explicit F2IP packing also remains closed as an isolated change.

Failed transient source implementations have been removed. The retained source contains the persistent P/dS MMA-A reuse, saturating `F2I.S8` conversion, optional power-of-two dS policy, and optional periodic dS max-skip policy relative to the packet-cache source.

### Validation And Tooling Completed

- Added `bench/bench_sagebwd_cutlass.py` with kernel-only/end-to-end modes, TFLOPS, medians, variance, ranking, and CSV output.
- Added the focused profile helper `build/profile_sagebwd_once.py`.
- Added source/SASS, resource, shared-memory, long-accuracy, and Racecheck workflows.
- Validated aligned and tail NHD/HND correctness, long-shape accuracy, local traffic, shared-memory allocation, and serial order controls.
- Kept 512-token NCU captures diagnostic and 4096/8192 serial timing governing.
- The current native build succeeds under CUDA 13.3; PyTorch was built with CUDA 13.2, so the existing minor-version mismatch warning remains non-blocking.

## What Remains To Do

### Priority 0: Native-SM86 Attribution Closed

- Captured matched sequence-8192 Flash-wheel and native-SM86 Sage main kernels with the same seeded inputs, warmup count, launch order, and raw-NCU metric set.
- Added native Sage CUDA/SASS SourceCounters correlation and re-ranked P/dS reconstruction/quantization, packed MMA-B handoff, scale application, dQ staging, and control work.
- Built only the exact FlashAttention kernel probe for SM80 and SM86; both builds remained below five minutes. No full FlashAttention package rebuild occurred.
- Keep the installed FlashAttention wheel as the runtime reference for upcoming experiments because its targeted SM80/SM86 comparison showed no notable architectural difference.
- Keep K64 generated as the same-build geometry reference until one attributed candidate is tested or K64 removal is explicitly approved.

### Priority 1: Direct Quantization And Single-Domain Format

The next candidate sequence is deliberately narrow. Each step must retain the current `Q32/K64, M64xN128` implementation as a same-build control and must remove measured work, rather than only alter register allocation or occupancy.

- Full-pipeline baseline first: separate Q/K `per_block_int8`, dO preprocessing, fused main, and dQ postprocess timing for the retained implementation. Record full-pipeline time as well as the main-kernel NCU counters, so a coarser Q/K block cannot win only outside the measured path.
- Fixed-P result: the naive global linear scale is rejected before kernel integration because dV emulation failed badly at long sequence lengths. Conversion-level `F2I.S8` saturation is retained as a neutral safety improvement.
- Direct P/dS handoff variant: only after fixed P is measured, combine producer fragments with a mirror-only dS dQ store. This is the highest-value revisit of the rejected warp-local handoff because it can remove canonical P/dS movement rather than merely retile it.
- Q32/K128 single-domain result: the exact-scale specialization is rejected and removed after losing all kernel-only and end-to-end controls. Revisit K128 domains only with a direct/predicted dS format that removes the eight-warp maximum dependency.
- Power-of-two and clipped-dS variants: exact power-of-two dS and the periodic max-skip policy are retained as measured optional references. The mirror-only periodic dS-to-dK handoff is rejected because its vectorized shared load loses to canonical LDSM; direct producer-to-dKV work remains open only if it eliminates that load and the canonical P/dS handoff with a new producer mapping. Proceed to a more aggressive direct dS scale only when it removes the exact maximum, barrier, and second dS pass. Record saturation and zero rates with every accuracy result.
- Q64/K128 paired-M variant: attempt only if a fixed/predicted dS path makes two M32 pairs live at once worthwhile. It must use a new M64 lifetime design; a Q64-only source change is not sufficient.
- Packet-handoff variant: revisit a Q/dO packet mapping only when the direct P/dS consumer topology removes line-694 work. Do not retry lane-major packets or generic double buffering as isolated changes.
- dV/dK/dQ scale application: treat it as the next target after a representation path has removed P/dS work. It is dynamically large, but no candidate is retained if it converts multiply work into spills, higher shared traffic, or a weaker format without a timing win.
- Flash comparison: compare the best predeclared Sage configuration with the installed FlashAttention dispatch-selected kernel for each shape. Do not constrain either implementation to the other's quantization or CTA tile size.

### Priority 2: Lower-Bit Atom Feasibility (Deferred)

Signed-INT4 feasibility and integration are deferred outside the current SM86 INT8 optimization scope. Do not spend build or profiling time on an S4 assembler, copy-plus-MMA, or low-bit accuracy probe during this plan; reopen it only after the INT8 representation and dataflow work is complete and explicitly approved.

Every candidate must remain spill-free, preserve the 57,664-byte shared-memory lifetime graph unless a measured redesign justifies changing it, and beat the retained source in the full serial matrix.

### Priority 3: Production Completion

- Decide whether to remove the K64 reference after the next attributed experiment.
- Restore broader generated dispatch and head dimension 128 only after the head-64 source is finalized.

### Validation Gates

Retain a source change only when all applicable gates pass:
- Focused FlashAttention-referenced correctness for K64/K128, NHD/HND, and aligned/tail lengths.
- Schedule-preserving limits: `1 - cosine < 2e-3`, relative error `< 0.06`, maximum absolute error `< 0.20`, and no NaN/Inf.
- First format/P/dS approximation limits: `1 - cosine < 1e-2`, relative error `< 0.15`, maximum absolute error `< 0.50`, and no NaN/Inf. Record each gradient separately plus P/dS saturation, clipping, and zero rates. These limits permit a deliberate block-format tradeoff but do not automatically promote a candidate to the default.
- Low-bit feasibility limits: `1 - cosine < 2e-2`, relative error `< 0.25`, maximum absolute error `< 0.75`, and no NaN/Inf. Passing this tier establishes only whether the format merits further work; default promotion still requires the first format tier or application-level validation.
- Racecheck at aligned 512 and odd-tail 513 reports zero hazards, errors, and warnings.
- K128 has zero local load/store sectors. K64 retains 128 registers and two resident CTAs.
- Shared memory, register count, instructions, conflicts per wavefront, reductions, synchronization, saturation, and scale/zero distributions are recorded.
- Full-pipeline Q/K quantization, preprocessing, main-kernel, and postprocess timing are recorded separately before judging a quantization-block change.
- Warmed serial NHD/HND 4096/8192 timing is run in both configuration/sequence orders.
- Report effective TFLOPS and the SageAttention/FlashAttention speed ratio alongside elapsed time. Compare the fastest predeclared Sage candidate with Flash's runtime-selected kernel, while retaining per-configuration results to prevent selection bias.
- A short-profile gain alone is insufficient. Long-shape timing decides retention.
- `git diff --check`, focused tests, and pre-commit all pass.

### Standard Commands

Build with four jobs by default. Use a sequential build only for an isolated compiler or SASS-debugging probe:

```powershell
$Env:NVCC_APPEND_FLAGS = '-lineinfo'
$Env:MAX_JOBS = '4'
python setup.py build_ext --inplace
```

Focused correctness:

```powershell
python -m pytest -q tests/test_sagebwd_cutlass.py
```

Aligned and odd-tail K128 Racecheck:

```powershell
$racecheck = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\compute-sanitizer.bat'
& $racecheck --tool racecheck --error-exitcode 1 python build/profile_sagebwd_once.py sage `
  --head-dim 64 --seq-len 512 --num-heads 16 --warmup 1 --block-config 32,64,64,128
& $racecheck --tool racecheck --error-exitcode 1 python build/profile_sagebwd_once.py sage `
  --head-dim 64 --seq-len 513 --num-heads 16 --warmup 1 --block-config 32,64,64,128
```

Matched long-shape timing:

```powershell
python bench/bench_sagebwd_cutlass.py `
  --batch-size 1 --num-heads 16 --head-dims 64 `
  --seq-lens 4096 8192 --layout NHD `
  --warmup 25 --repeats 100 `
  --block-configs 32,64,64,64 32,64,64,128 `
  --mode kernel-only --csv build/bench_sagebwd_current_nhd.csv
```

Repeat for HND, then reverse sequence/configuration order for the paired control.

### Key Artifacts

- Kernel: `csrc/qattn_cutlass/qk_int8_sv_f16_bwd_kernel_cutlass_sm80.cuh`
- Launch: `csrc/qattn_cutlass/qk_int8_sv_f16_bwd_launch_cutlass_sm80.cuh`
- Generated dispatch: `csrc/qattn_cutlass/generated/qk_int8_sv_f16_accum_f32_attn_bwd_cutlass_dispatch.cuh`
- Generator: `scripts/generate_cutlass_bwd_instantiations.py`
- Python wrapper: `sageattention/cutlass_bwd.py`
- Benchmark: `bench/bench_sagebwd_cutlass.py`
- Profile helper: `build/profile_sagebwd_once.py`
- Focused tests: `tests/test_sagebwd_cutlass.py`
- Transposed-copy test: `tests/cuda/test_qattn_cutlass_bwd_int8_transposed_copy.cu`
- Final retained K128 profile: `build/ncu_sage_hd64_q32_k64_k128_cache_pds_retained_sm86_final.csv`
- K64 native reference profile: `build/ncu_sage_hd64_q32_k64_k64_sm86_baseline.csv`
- Retained timing: `build/bench_sagebwd_sm86_cache_pds_nhd_forward.csv`, `build/bench_sagebwd_sm86_cache_pds_hnd_forward.csv`, `build/bench_sagebwd_sm86_cache_pds_nhd_reverse.csv`, `build/bench_sagebwd_sm86_cache_pds_hnd_reverse.csv`
- Rejected prefetch profile: `build/ncu_sage_hd64_q32_k64_k128_cache_pds_early_do_fp16_sm86.csv`
- Rejected warp-sync profile: `build/ncu_sage_hd64_q32_k64_k128_cache_dkv_warp_sync_sm86.csv`
- Historical SM80 packet profile: `build/ncu_sage_hd64_q32_k64_k128_qdo_packet_v_alias.csv`
- Matched Flash-wheel raw NCU attribution: `build/ncu_flash_hd64_seq8192_attribution_wheel_sm80.csv`
- Matched native-SM86 Sage raw NCU attribution: `build/ncu_sage_hd64_q32_k64_k128_cache_pds_seq8192_attribution_sm86.csv`
- Native-SM86 Sage CUDA/SASS SourceCounters: `build/ncu_sage_hd64_q32_k64_k128_cache_pds_seq8192_source_sm86.csv`
- Power-of-two dS fixed-clock NCU controls: `build/ncu_sage_hd64_power2_ds_dynamic_seq8192.csv`, `build/ncu_sage_hd64_power2_ds_candidate_seq8192.csv`
- Periodic dS fixed-clock NCU control: `build/ncu_sage_hd64_periodic_ds_candidate_seq8192_corrected.csv`
- Rejected mirror-only dS-to-dK paired timing: `build/bench_sagebwd_periodic_ds_mirror_dkv_nhd_kernel.csv`, `build/bench_sagebwd_periodic_ds_mirror_dkv_hnd_kernel.csv`
- Targeted Flash SM80/SM86 cubins, build logs, and SASS: `build/native_sm86_attribution/`

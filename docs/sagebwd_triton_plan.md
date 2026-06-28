# SageBwd Triton Kernel Plan

## Goal

Add and tune a trainable Triton SageAttention path that follows the SageBwd design: use int8 tensor core matmuls for the backward matmuls that tolerate quantization, keep the numerically sensitive `dP = dO @ V^T` matmul in fp16/bf16 precision, and validate accuracy against FlashAttention backward.

## Current Implementation Status

- Non-causal attention only.
- Fixed-length dense tensors only.
- Canonical internal layout is `tensor_layout="NHD"`. The trainable wrapper accepts `NHD` and `HND` by converting to/from NHD.
- Trainable paths require `q`, `k`, and `v` to have the same shape. GQA/MQA is not implemented.
- Trainable paths support `torch.float16` only.
- Forward quantization products `q_int8`, `k_int8`, `q_scale`, `k_scale` are saved for backward.
- The default trainable backward is split into fused preprocess/`dO` quantization, `dQ`, and `dK/dV` kernels.
- The fused preprocess kernel computes `Delta` and pre-quantizes `dO` into `DOInt8`/`DOScale` once per Q block.
- Both default `dK/dV` and reuse backward consume pre-quantized `DOInt8`/`DOScale` instead of quantizing `dO` inside KV-owned loops.
- The reuse trainable backward uses a FlashAttention-style KV-block-owned kernel that computes `dQ`, `dK`, and `dV` from the same `P/dS` tile and accumulates `dQ` through fp32 workspace.
- Public APIs are `sageattn_qk_int8_pv_fp16_triton_trainable` and `sageattn_qk_int8_pv_fp16_triton_trainable_reuse`. `SAGEATTN_BACKEND=triton_trainable_reuse` selects the reuse backend.
- Trainable eager and `torch.compile(fullgraph=True, mode="max-autotune")` paths autotune shared outer `(BLOCK_M, BLOCK_N)` by forward + backward total time.
- The Triton forward, backward preprocess, default `dQ`, default `dK/dV`, and reuse kernels autotune their inner launch configs independently. Backward preprocess and reuse `DQAccum` utility kernels tune `num_warps`. Default `dQ`, default `dK/dV`, and reuse tune `num_warps` and `num_stages` with shared-memory pruning.

## Paper Facts

SageBwd backward matmuls are:

```text
S  = Q @ K^T
dV = P^T @ dO
dP = dO @ V^T
dQ = dS @ K
dK = dS^T @ Q
```

SageBwd keeps `dP = dO @ V^T` in fp16 precision and quantizes the other four backward matmuls with per-block int8.

The paper evaluates numerical accuracy with:

```text
CosSim = cosine_similarity(sage_tensor, reference_tensor)
Rel-L2 = ||sage_tensor - reference_tensor||_2 / ||reference_tensor||_2
```

Main reported intermediate accuracy versus full-precision attention:

| Tensor | CosSim | Rel-L2 |
|---|---:|---:|
| delta | 0.9973 | 0.0736 |
| P | 0.9917 | 0.1293 |
| dP | 1.0000 | 0.0000 |
| dS | 0.9789 | 0.2045 |
| O | 0.9969 | 0.0793 |
| dQ | 0.9664 | 0.2579 |
| dK | 0.9537 | 0.3074 |
| dV | 0.9985 | 0.0540 |

Reported speed goals:
- Backward-only: roughly `1.2x` to `1.6x` over FlashAttention in the earlier SageAttention3 paper appendix.
- Forward + backward kernel throughput: up to `1.67x` over FlashAttention2 on RTX4090.
- End-to-end training latency: about `1.15x` in their Llama runs.

## `pv_accum_dtype` Note

The paper does not map SageBwd to this repo's `pv_accum_dtype` option. In the paper, SageBwd forward uses int8 quantization for `P @ V`. This repo currently has `qk_int8_pv_fp16` forward with `pv_accum_dtype="fp32"` or `"fp16"`, and does not yet implement the fp8/int8 `P @ V` path known from SageAttention2++ or SageBwd.

For the current backward implementation:
- Use `pv_accum_dtype="fp32"` as the default forward setting for correctness tests, because it minimizes unrelated forward accumulation error and isolates the backward kernel.
- Keep test coverage for `pv_accum_dtype="fp16"` as a performance-relevant mode after the trainable autotune path is stable.
- Do not block the backward work on `pv_accum_dtype="fp8"`. It is not implemented in this repo and is outside the current scope.
- Accuracy numbers may differ from the paper until the forward path also matches SageBwd's int8 `P @ V` design.

## Kernel Architecture

### Forward

The trainable wrapper currently reuses the Triton forward path:

```text
q, k -> per_block_int8 -> q_int8, q_scale, k_int8, k_scale
q_int8, k_int8, v, scales -> Triton forward -> out, lse
```

The forward pass saves the quantized tensors and scales for backward. Shared quantization currently uses `max(abs(x)) / 127.5 + 1e-7` with PTX round-to-nearest conversion and no extra clamp, because local accuracy checks showed this gives lower practical quantization error than clamped `127.0` variants.

### Backward Preprocess

The fused preprocess kernel computes the row-wise softmax-gradient scalar and per-Q-block `dO` quantization:

```text
delta = sum(out * dO, dim=-1)
DOInt8, DOScale = quantize_per_block(dO)
```

`Delta` has shape `[batch, heads, seqlen]` in canonical NHD-derived storage. `DOInt8` matches `dO`'s NHD shape, and `DOScale` has shape `[batch, heads, ceil(seqlen / BLOCK_M)]`.

### `dQ` Kernel

Grid over `(q_block, head, batch)`.

For each Q block, loop over all KV blocks:

```text
S_ij = int8(q_i) @ int8(k_j)^T * q_scale_i * k_scale_j * sm_scale
P_ij = exp(S_ij - lse_i)
dP_ij = dO_i @ V_j^T  # fp16/bf16 matmul, not int8
dS_ij = P_ij * (dP_ij - delta_i)
quantize dS_ij per block
dQ_i += int8(dS_ij) @ int8(k_j) * dS_scale * k_scale_j
```

If `smooth_k=True`, include the correction term:

```text
dQ_i += rowsum(dS_ij) * k_mean
```

### `dK/dV` Kernel

Grid over `(kv_block, head, batch)`.

Before launching this kernel, the fused preprocess quantizes `dO` once per Q block into `DOInt8` and `DOScale`.

For each K/V block, loop over all Q blocks:

```text
S_ij = int8(q_i) @ int8(k_j)^T * q_scale_i * k_scale_j * sm_scale
P_ij = exp(S_ij - lse_i)
quantize P_ij per block
load pre-quantized int8(dO_i), dO_scale_i
dV_j += int8(P_ij)^T @ int8(dO_i) * p_scale * dO_scale

dP_ij = dO_i @ V_j^T  # fp16/bf16 matmul, not int8
dS_ij = P_ij * (dP_ij - delta_i)
quantize dS_ij per block
dK_j += int8(dS_ij)^T @ int8(q_i) * dS_scale * q_scale_i
```

Pre-quantizing `dO` avoids quantizing the same `dO_i` tile once per KV block. For `seq_len=4096` and `BLOCK_N=64`, this removes roughly 64 repeated `dO` quantizations per Q tile. Fusing that quantization into preprocess removes the separate quantization launch and extra standalone `dO` read from the default backend.

This split keeps write ownership simple:
- `dQ` kernel owns one Q block, so no atomics for `dQ`.
- `dK/dV` kernel owns one KV block, so no atomics for `dK` or `dV`.
- `S`, `P`, `dP`, and `dS` are recomputed in both kernels. This remains the conservative default trainable backend.

### Reuse Backward Kernel

The reuse backend is a FlashAttention-style path with grid over `(kv_block, head, batch)`. Each KV-owned program loads one K/V tile and loops over Q blocks:

```text
S_ij = int8(q_i) @ int8(k_j)^T * q_scale_i * k_scale_j * sm_scale
P_ij = exp(S_ij - lse_i)
quantize P_ij per block
load pre-quantized int8(dO_i), dO_scale_i
dV_j += int8(P_ij)^T @ int8(dO_i) * p_scale * dO_scale

dP_ij = dO_i @ V_j^T  # fp16 matmul, not int8
dS_ij = P_ij * (dP_ij - delta_i)
quantize dS_ij per block
dK_j += int8(dS_ij)^T @ int8(q_i) * dS_scale * q_scale_i
dQ_i partial = int8(dS_ij) @ int8(k_j) * dS_scale * k_scale_j
```

`dK` and `dV` are owned by the KV program and stored directly. `dQ` is accumulated into fp32 `DQAccum` workspace and converted to fp16 by a final utility kernel.

`SAGEATTN_REUSE_DQ_SPLITS` controls optional split-plane `DQAccum` workspace:
- Default is `1`, matching the original single fp32 accumulation plane with `tl.atomic_add`. Nsight Compute reports these updates as global reduction operations.
- Values greater than `1` shard KV blocks across split planes, reducing per-plane reduction contention and adding a final split reduction.
- If the requested split count covers all KV blocks, the reuse kernel uses stores instead of reductions for `dQ` partials.
- On the local sm86 validation machine, split counts greater than `1` were correct but slower for the benchmarked 2048/4096/8192 sequence lengths, so the default remains `1`.

## Local Profiling Findings

Latest profiling used Nsight Compute 2026.2 on the local sm86 machine with `batch=1`, `heads=16`, `head_dim=64`, `seq_len=4096`, non-causal fp16.

Representative public benchmark after standalone `dO` pre-quantization, before fused preprocess was added:

| Method | Forward ms | Backward ms | Total ms |
|---|---:|---:|---:|
| FlashAttention | 1.984 | 4.667 | 6.651 |
| Sage default | 1.303 | 6.035 | 7.338 |

Default-backend kernel replay NCU summary after standalone `dO` pre-quantization, before fused preprocess was added:

| Kernel | Time ms | Tensor core activity | Registers/thread | Notes |
|---|---:|---:|---:|---|
| FlashAttention main backward | 5.749 | HMMA ~45.7% | 255 | seq-k-parallel fused `dQ/dK/dV` kernel |
| Sage `dQ` | 2.401 | IMMA/HMMA ~21.9% | 255 | owns Q block, no atomics |
| Sage `dK/dV` | 4.907 | IMMA ~16.0%, HMMA ~10.7% | 235 | current dominant kernel |
| Sage fused preprocess + `dO` quant | not re-profiled | none | not re-profiled | computes `delta`, `DOInt8`, and `DOScale` in one launch |

Matched FlashAttention-vs-reuse reduction profiling used the same shape and matched Sage reuse to FlashAttention's observed main backward tile `(BLOCK_M=64, BLOCK_N=128)`. Nsight Compute reports the `dQ_accum` updates as global reduction operations, not global atom operations.

| Kernel | Main Kernel Time ms | Global Reduction Bytes | Global Reduction Requests | LTS Reduction-Active Cycles | Notes |
|---|---:|---:|---:|---:|---|
| FlashAttention main backward | 5.745 | 536.9 MB | 4.19M | 16.78M | `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel`, grid 512, block 256 |
| Sage reuse `(64,128)`, split 1 | 16.720 | 2.147 GB | 4.19M | 67.11M | Uses global reductions for `DQAccum` |
| Sage reuse `(64,128)`, split 32 | 15.743 | 0 | 0 | 0 | Store-only main-kernel path. Final split reduction still needed outside this kernel |

Measured conclusions:
- Default split-backend profiling found `_bwd_dkdv_kernel` was the main default-backend bottleneck at `seq_len=4096`, and pre-quantizing `dO` moved it from roughly `5.65 ms` to `4.91 ms` in NCU kernel replay.
- The reuse backend is now the primary optimization target because its fused KV-owned structure most closely matches FlashAttention backward: it computes `dQ`, `dK`, and `dV` from the same score/probability tiles instead of recomputing them in separate split kernels.
- Matched `(64,128)` profiling shows Sage reuse has roughly `4x` FlashAttention's global reduction bytes/sectors and LTS reduction-active cycles for `DQAccum`, but a Sage split-1 vs split-32 A/B changes main-kernel time by only about `5.8%`, so reductions are not the only bottleneck.
- A high-level physical `DQAccum` layout-only experiment did not reduce global reduction bytes, requests, sectors, or LTS reduction-active cycles. Further `DQAccum` optimization is deferred until the main reuse kernel is closer on scheduling and tensor-core utilization.
- The next performance gap is structural: raise tensor-core utilization and reduce register/scheduling pressure in the fused reuse kernel, not split `dK` and `dV` into separate kernels.

## Optimization Roadmap

- Re-profile after recent structural changes.
  - Measure total backward time and main-kernel NCU counters for FlashAttention, default Sage, and reuse Sage after even-shape masks and fused preprocess/`dO` quantization.
  - Update benchmark tables for `seq_len=2048/4096/8192` after cache/autotune warmup.
- Prioritize the reuse kernel.
  - Focus on the fused KV-owned reuse path because it is the closest structural match to FlashAttention backward.
  - Reduce reuse-kernel register pressure and improve scheduling/tensor-core utilization before returning to `DQAccum` layout or split-workspace work.
  - Retune `num_warps`, `num_stages`, and selected `(BLOCK_M, BLOCK_N)` candidates using NCU metrics for IMMA/HMMA active cycles, registers/thread, achieved occupancy, and kernel time.
- Keep the split default backend conservative.
  - The default `dQ` plus `dK/dV` split backend remains useful for correctness and as a no-`DQAccum` baseline.
  - Do not prioritize separate `dV-only` and `dK-only` kernels now. Splitting them further is unlikely to close the FlashAttention gap because FlashAttention itself follows a fused reuse structure.
- Defer `DQAccum` optimization.
  - Current split-plane workspace and high-level layout experiments did not improve total behavior enough on sm86.
  - If revisited later, include total backward time, not just `_bwd_reuse_kernel`, because store-only/split paths add zero/convert reduction work outside the main kernel.
  - Matching FlashAttention's reduction traffic likely requires different per-lane/write mapping or a CUDA/CUTE-style `dQaccum` copy path rather than only a different high-level tensor layout.

## Benchmark and Profiling Workflow

- Use `bench/bench_fwd_bwd.py` for public latency comparisons across `sdpa`, `flash`, `sage`, and `sage_reuse`.
  - Primary target shape: `--batch_size 1 --num_heads 16 --head_dim 64 --seq_lens 2048 4096 8192`.
  - Use `--reuse-dq-splits` to benchmark split-plane `DQAccum` choices.
- Use Nsight Compute for kernel-level comparisons.
  - Profile after warmup and isolate backward with `cudaProfilerStart`/`cudaProfilerStop`.
  - Prefer kernel replay for Triton/autotuned workloads when application replay sees inconsistent kernels.
  - Track at least kernel time, registers/thread, shared memory, achieved occupancy, DRAM/L2 throughput, HMMA activity, IMMA activity, and global reduction counters for reuse/FlashAttention comparisons.
- Expand support only after the non-causal fp16 path is stable.
  - `head_dim=128` and `head_dim=256` correctness/performance sweeps beyond the current autotune validity coverage.
  - bf16.
  - GQA/MQA.
  - Causal mode.
  - Varlen.

## Risks And Open Questions

- int8 quantization granularity for `P` in backward currently follows the paper's Algorithm 3 per-block description. Accuracy may still require experimenting with per-row/per-token scaling.
- `dS` is expected to be the fragile tensor. Debug tooling should expose `dS` metrics if accuracy falls below target.
- Long sequence performance may need additional scheduling work beyond the current split-kernel and reuse implementations.
- Local profiling shows Sage backward still trails FlashAttention at `seq_len=4096`. The primary near-term gap is reuse-kernel tensor-core utilization, scheduling, and register pressure.
- A single shared trainable block size is the right current design, but it means the chosen config is a compromise across quantization, forward attention, and backward kernels.
- Reuse `dQ` accumulation can be sharded to reduce main-kernel reduction traffic/contention, but current sm86 benchmarks show the extra zero/reduce workspace cost outweighs the benefit for the tested shapes, so larger `DQAccum` work is deferred.

## Validation Snapshot

- Correctness uses FlashAttention backward as the reference and flattened cosine similarity plus Frobenius relative L2 metrics.
- Configured reuse and forced split-plane `DQAccum` tests pass.
- Latest full validation: `python -m pytest -n 8 tests/ -q` reported `1069 passed`. `pre-commit run --all-files` passed.
- After fused default/reuse `dO` pre-quantization and even-shape specialization, focused validation passed: `python -m pytest tests/test_sagebwd_triton.py tests/test_sagebwd_triton_compile.py tests/test_sagebwd_triton_reuse.py tests/test_sagebwd_triton_reuse_compile.py -q` reported `17 passed`. `pre-commit run --files sageattention/triton/attn_bwd_qk_int8.py sageattention/triton/attn_bwd_qk_int8_reuse.py` passed.

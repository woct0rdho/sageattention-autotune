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
- The default trainable backward is split into preprocess `delta`, `dQ`, and `dK/dV` kernels.
- The reuse trainable backward uses a FlashAttention-style KV-block-owned kernel that computes `dQ`, `dK`, and `dV` from the same `P/dS` tile and accumulates `dQ` through fp32 workspace.
- Public APIs are `sageattn_qk_int8_pv_fp16_triton_trainable` and `sageattn_qk_int8_pv_fp16_triton_trainable_reuse`; `SAGEATTN_BACKEND=triton_trainable_reuse` selects the reuse backend.
- Trainable eager and `torch.compile(fullgraph=True, mode="max-autotune")` paths autotune shared outer `(BLOCK_M, BLOCK_N)` by forward + backward total time.
- The Triton forward, backward preprocess, default `dQ`, default `dK/dV`, and reuse kernels autotune their inner launch configs independently. Backward preprocess and reuse `DQAccum` utility kernels tune `num_warps`; default `dQ`, default `dK/dV`, and reuse tune `num_warps` and `num_stages` with shared-memory pruning.

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

The forward pass saves the quantized tensors and scales for backward. Shared quantization uses symmetric int8 rounding with output clamped to `[-127, 127]`, so `-128` is not produced.

### Backward Preprocess

The preprocess kernel computes the row-wise softmax-gradient scalar:

```text
delta = sum(out * dO, dim=-1)
```

Shape: `[batch, heads, seqlen]` in canonical NHD-derived storage.

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

For each K/V block, loop over all Q blocks:

```text
S_ij = int8(q_i) @ int8(k_j)^T * q_scale_i * k_scale_j * sm_scale
P_ij = exp(S_ij - lse_i)
quantize P_ij per block
quantize dO_i per block
dV_j += int8(P_ij)^T @ int8(dO_i) * p_scale * dO_scale

dP_ij = dO_i @ V_j^T  # fp16/bf16 matmul, not int8
dS_ij = P_ij * (dP_ij - delta_i)
quantize dS_ij per block
dK_j += int8(dS_ij)^T @ int8(q_i) * dS_scale * q_scale_i
```

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
quantize dO_i per block
dV_j += int8(P_ij)^T @ int8(dO_i) * p_scale * dO_scale

dP_ij = dO_i @ V_j^T  # fp16 matmul, not int8
dS_ij = P_ij * (dP_ij - delta_i)
quantize dS_ij per block
dK_j += int8(dS_ij)^T @ int8(q_i) * dS_scale * q_scale_i
dQ_i partial = int8(dS_ij) @ int8(k_j) * dS_scale * k_scale_j
```

`dK` and `dV` are owned by the KV program and stored directly. `dQ` is accumulated into fp32 `DQAccum` workspace and converted to fp16 by a final utility kernel.

`SAGEATTN_REUSE_DQ_SPLITS` controls optional split-plane `DQAccum` workspace:
- Default is `1`, matching the original single fp32 accumulation plane with `tl.atomic_add`.
- Values greater than `1` shard KV blocks across split planes, reducing atomic contention and adding a final split reduction.
- If the requested split count covers all KV blocks, the reuse kernel uses stores instead of atomics for `dQ` partials.
- On the local sm86 validation machine, split counts greater than `1` were correct but slower for the benchmarked 2048/4096/8192 sequence lengths, so the default remains `1`.

## Optimization Roadmap

- Save/reuse as much forward state as practical.
  - Already saves Q/K int8 and scales.
  - Reuse backend avoids recomputing `P/dS` separately for `dQ` and `dK/dV`.
  - Consider dedicated `dO` quantization if repeated quantization dominates.
- Benchmark and tune forward+backward.
  - `bench/bench_fwd_bwd.py` compares `sdpa`, `flash`, `sage`, and `sage_reuse` for non-causal forward+backward.
  - It reports separate forward, backward, and total latency/TFLOP/s.
  - Use `--reuse-dq-splits` to benchmark split-plane `DQAccum` choices.
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
- A single shared trainable block size is the right current design, but it means the chosen config is a compromise across quantization, forward attention, and backward kernels.
- Reuse `dQ` accumulation can be sharded to reduce atomic contention, but current sm86 benchmarks show the extra zero/reduce workspace cost outweighs the benefit for the tested shapes.

## Validation Snapshot

- Correctness uses FlashAttention backward as the reference and flattened cosine similarity plus Frobenius relative L2 metrics.
- Configured reuse and forced split-plane `DQAccum` tests pass.
- Latest full validation: `python -m pytest -n 8 tests/ -q` reported `1069 passed`; `pre-commit run --all-files` passed.

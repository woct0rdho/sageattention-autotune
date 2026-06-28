# SageBwd Triton Kernel Plan

## Goal

Add and tune a trainable Triton SageAttention path that follows the SageBwd design: use int8 tensor core matmuls for the backward matmuls that tolerate quantization, keep the numerically sensitive `dP = dO @ V^T` matmul in fp16/bf16 precision, and validate accuracy against FlashAttention backward.

## Current Implementation Status

- Non-causal attention only.
- Fixed-length dense tensors only.
- Canonical internal layout is `tensor_layout="NHD"`. The trainable wrapper accepts `NHD` and `HND` by converting to/from NHD.
- Initial trainable path requires `q`, `k`, and `v` to have the same shape. GQA/MQA is not implemented.
- Initial trainable path supports `torch.float16` only.
- Forward quantization products `q_int8`, `k_int8`, `q_scale`, `k_scale` are saved for backward.
- Backward is split into preprocess `delta`, `dQ`, and `dK/dV` kernels.
- Trainable eager and `torch.compile(fullgraph=True, mode="max-autotune")` paths autotune shared outer `(BLOCK_M, BLOCK_N)` by forward + backward total time.
- The Triton forward, backward preprocess, `dQ`, and `dK/dV` kernels autotune their inner launch configs independently. Backward preprocess tunes `num_warps`; `dQ` and `dK/dV` tune `num_warps` and `num_stages` with separate shared-memory pruning.

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

The forward pass saves the quantized tensors and scales for backward.

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
- `S`, `P`, `dP`, and `dS` are recomputed in both kernels. That is acceptable for the first optimized path and can be revisited after profiling.

## Optimization Roadmap

- Save/reuse as much forward state as practical.
  - Already saves Q/K int8 and scales.
  - Consider dedicated `dO` quantization if repeated quantization dominates.
- Add benchmark script.
  - Backward-only.
  - Forward+backward.
  - FlashAttention comparison.
- Expand support only after the non-causal fp16 path is stable.
  - `head_dim=128` and `head_dim=256` correctness/performance sweeps beyond the current autotune validity coverage.
  - bf16.
  - GQA/MQA.
  - Causal mode.
  - Varlen.

## Risks And Open Questions

- int8 quantization granularity for `P` in backward currently follows the paper's Algorithm 3 per-block description. Accuracy may still require experimenting with per-row/per-token scaling.
- `dS` is expected to be the fragile tensor. Debug tooling should expose `dS` metrics if accuracy falls below target.
- Long sequence performance may need additional scheduling work beyond the current split-kernel implementation.
- A single shared trainable block size is the right current design, but it means the chosen config is a compromise across quantization, forward attention, `dQ`, and `dK/dV` kernels.

# sageattention-autotune

Port of [SageAttention](https://github.com/thu-ml/SageAttention) with autotuned block sizes and other quality-of-life improvements.

This fork ports:

- The SageAttention 2 CUDA **sm80** kernels (int8 QK / fp16 PV) and the Triton kernels, with autotune configs originally tuned for RTX 30xx.
- The SageAttention2++ **sm89** kernels (int8 QK / **fp8 PV**, the `sv_f8` fast path) for **RTX 40xx (sm_89)** and **RTX 50xx (sm_120)**, with their own autotune config set. The official SageAttention ships a single, undertuned block size for these kernels; here the block size is selected by autotuning, which especially helps RTX 50xx.

On Ada / Blackwell the CUDA backend automatically routes through the fp8 kernels (see `SAGEATTN_DISABLE_FP8` below to opt out). On a 5090 the fp8 path runs at roughly 0.999 cosine similarity vs. an fp16 reference and ~2.8x faster than PyTorch's flash-attention SDPA.

This repo also serves as an example of how to do autotune when multiple kernels (like quant kernel and attn kernel) need consistent parameters (like block sizes).

## Installation

This is experimental. You should know what you're doing. If not, maybe this is not for you and you can install my [SageAttention Windows wheel](https://github.com/woct0rdho/SageAttention) instead.

This repo builds a package with pip package name `sageattention` and import name `sageattention`. It's a drop-in replacement of the official one.

PyTorch >= 2.12 is required for the latest fixes with `torch.compile`.

### GPU architectures (fp8 / RTX 40xx & 50xx)

The fp8 (`sv_f8`) extension is built for the GPU it detects at build time. The sm_89 fp8 kernels need CUDA >= 12.4; sm_120 (Blackwell) needs CUDA >= 12.8. To build a single wheel that runs on both Ada and Blackwell, or to cross-build, set:

```
SAGEATTN_CUDA_ARCH="89;120"   # also accepts "89", "120", commas, or "8.9"
```

If unset, the installed GPU's compute capability is used. The build skips the fp8 extension (with a clear error) if the CUDA toolkit is too old for the requested arch; the fp16 sm80 extension is always built (for sm_80 plus the detected arch, so the fp16 path also runs on Ada/Blackwell).

## Usage

The APIs `sageattn_qk_int8_pv_fp16_cuda` and `sageattn_qk_int8_pv_fp16_triton` are provided. `sageattn` is aliased to CUDA by default, and Triton if the env var `SAGEATTN_TRITON_BACKEND=1`.

The official SageAttention's Triton kernel behave like `pv_accum_dtype="fp16"`. I've added `pv_accum_dtype="fp32"` to the Triton kernel. `pv_accum_dtype="fp16+fp32"` is still not supported in the Triton kernel.

You can set the default `pv_accum_dtype` using the env var like `SAGEATTN_DEFAULT_PV_ACCUM_DTYPE=fp16`. As always, `pv_accum_dtype="fp16+fp32"` is faster than `pv_accum_dtype="fp32"`, and `pv_accum_dtype="fp16"` is even faster, but more likely to cause black/noise/degraded output. Maybe the time has come that we have to tune it for each model.

### fp8 backend (RTX 40xx / 50xx)

On Ada (sm_89) and Blackwell (sm_120) the CUDA backend uses the int8-QK / fp8-PV kernels. `pv_accum_dtype` maps as:

- `"fp32"` -> fp8 kernel with f32 PV accumulation (most accurate).
- `"fp16+fp32"` / `"fp16"` -> SageAttention2++ instruction-buffer path with fp16 PV accumulation (the `sv_f8` fast path).

The default on these GPUs is the fast 2++ path (`fp16+fp32`); set `SAGEATTN_DEFAULT_PV_ACCUM_DTYPE` to override. `smooth_v` is currently ignored on the fp8 backend. Set `SAGEATTN_DISABLE_FP8=1` to force the fp16 (sm80) path instead.

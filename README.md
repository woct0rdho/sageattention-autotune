# SageAttention Ampere

This is a port of SageAttention 2 that only keeps code paths for Ampere GPUs. It does not include SageAttention 1 Triton attention kernels, SageAttention 3 Blackwell kernels, sm89/sm90 FP8 kernels, CUDA architecture dispatch, or varlen wrappers.

## Build

```bash
python setup.py install --verbose
```

Runtime still requires Triton because the fast per-thread Q/K quantizer is a Triton kernel.

## API

```python3
from sageattention import sageattn

out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
```

Inputs must be CUDA tensors with dtype `torch.float16` or `torch.bfloat16`. Head dimensions are padded to 64, 128, or 256.

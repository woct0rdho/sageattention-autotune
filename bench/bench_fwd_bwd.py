import argparse
import logging

import torch
from flash_attn import flash_attn_func
from flash_attn.utils.benchmark import benchmark_fwd_bwd
from torch.nn.functional import scaled_dot_product_attention as sdpa

from sageattention import sageattn_qk_int8_pv_fp16_triton_trainable, sageattn_qk_int8_pv_fp16_triton_trainable_fused

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="sage", choices=["sdpa", "flash", "sage", "sage_fused"])
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_heads", type=int, default=16)
parser.add_argument("--head_dim", type=int, default=64)
parser.add_argument("--seq_lens", nargs="+", type=int, default=[1024, 2048, 4096, 8192])
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--repeats", type=int, default=10)
args = parser.parse_args()

num_heads = args.num_heads
batch_size = args.batch_size
head_dim = args.head_dim

print(f"method: {args.method}")
print(f"batch_size: {batch_size}, num_heads: {num_heads}, head_dim: {head_dim}")
print("is_causal: False")


def _make_inputs(seq_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if args.method == "sdpa":
        shape = (batch_size, num_heads, seq_len, head_dim)
    else:
        shape = (batch_size, seq_len, num_heads, head_dim)

    q = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    dout = torch.randn_like(q)
    return q, k, v, dout


def _fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if args.method == "flash":
        return flash_attn_func(q, k, v, causal=False)
    if args.method == "sage":
        return sageattn_qk_int8_pv_fp16_triton_trainable(q, k, v, tensor_layout="NHD", is_causal=False)
    if args.method == "sage_fused":
        return sageattn_qk_int8_pv_fp16_triton_trainable_fused(q, k, v, tensor_layout="NHD", is_causal=False)

    return sdpa(q, k, v, is_causal=False)


for seq_len in args.seq_lens:
    # Non-causal attention FLOPs: forward has QK^T and PV; backward has dV, dP, dQ, and dK.
    fwd_flops = 4 * num_heads * batch_size * head_dim * seq_len * seq_len
    bwd_flops = 8 * num_heads * batch_size * head_dim * seq_len * seq_len
    total_flops = fwd_flops + bwd_flops
    q, k, v, dout = _make_inputs(seq_len)

    for _ in range(args.warmup):
        out = _fn(q, k, v)
        out.backward(dout, retain_graph=False)
        q.grad = None
        k.grad = None
        v.grad = None
    torch.cuda.synchronize()

    (_, fwd_time), (_, bwd_time) = benchmark_fwd_bwd(
        _fn, q, k, v, grad=dout, repeats=args.repeats, verbose=False, desc=args.method
    )
    total_time = fwd_time.mean + bwd_time.mean
    print(
        f"{seq_len} "
        f"fwd_ms:{fwd_time.mean * 1e3:.3f} "
        f"bwd_ms:{bwd_time.mean * 1e3:.3f} "
        f"total_ms:{total_time * 1e3:.3f} "
        f"fwd_tflops:{fwd_flops / fwd_time.mean * 1e-12:.3f} "
        f"bwd_tflops:{bwd_flops / bwd_time.mean * 1e-12:.3f} "
        f"total_tflops:{total_flops / total_time * 1e-12:.3f}"
    )

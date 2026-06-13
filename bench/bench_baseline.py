import argparse

import torch
from flash_attn import flash_attn_func
from flash_attn.utils.benchmark import benchmark_forward
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention as sdpa

from sageattention import sageattn

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="sdpa", choices=["sdpa", "flash_attn", "sage_attn"])
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_heads", type=int, default=32)
parser.add_argument("--head_dim", type=int, default=128)
args = parser.parse_args()

num_heads = args.num_heads
batch = args.batch_size
head_dim = args.head_dim

print(f"Baseline: {args.method}")
print(f"batch: {batch}, num_heads: {num_heads}, head_dim: {head_dim}")


def run_benchmark(is_causal: bool) -> None:
    print(f"is_causal: {is_causal}")
    for seq_len in [1024, 2048, 4096, 8192]:
        flops = 4 * num_heads * batch * head_dim * seq_len * seq_len // (2 if is_causal else 1)

        # Generate inputs in BHSD format (Batch, Head, Seq, Dim)
        q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        if args.method in ("flash_attn", "sage_attn"):
            # flash_attn_func and SageAttention NHD expect (Batch, Seq, Head, Dim)
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()

        if args.method == "flash_attn":

            def fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                return flash_attn_func(q, k, v, causal=is_causal)
        elif args.method == "sage_attn":

            def fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                return sageattn(q, k, v, tensor_layout="NHD", is_causal=is_causal)
        else:

            def fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    return sdpa(q, k, v, is_causal=is_causal)

        for _ in range(5):
            fn(q, k, v)
        torch.cuda.synchronize()

        _, time = benchmark_forward(fn, q, k, v, repeats=10, verbose=False, desc="Triton")
        print(f"{seq_len} flops:{flops / time.mean * 1e-12}")


run_benchmark(is_causal=False)
run_benchmark(is_causal=True)

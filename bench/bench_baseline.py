import torch
from flash_attn.utils.benchmark import benchmark_forward
from torch.nn.functional import scaled_dot_product_attention as sdpa

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

try:
    from sageattention import sageattn
except ImportError:
    sageattn = None

import argparse

parser = argparse.ArgumentParser(description="Benchmark Baseline")
parser.add_argument(
    "--method", type=str, default="fa2", choices=["fa2", "torch", "xformers", "flash_attn", "sage_attn"]
)
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
parser.add_argument("--num_heads", type=int, default=32, help="Number of heads")
parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
args = parser.parse_args()

head = args.num_heads
batch = args.batch_size
headdim = args.head_dim

assert args.method in ["fa2", "torch", "xformers", "flash_attn", "sage_attn"]

if args.method == "flash_attn":
    if flash_attn_func is None:
        raise ImportError("flash_attn is not installed or cannot be imported.")
elif args.method == "sage_attn":
    if sageattn is None:
        raise ImportError("sageattention is not installed or cannot be imported.")
else:
    # only one of the following is True
    torch.backends.cuda.enable_flash_sdp(args.method == "fa2")  # use FA2
    torch.backends.cuda.enable_math_sdp(args.method == "torch")  # use Torch
    torch.backends.cuda.enable_mem_efficient_sdp(args.method == "xformers")  # use xformers

print(f"Baseline: {args.method}")
print(f"batch: {batch}, head: {head}, headdim: {headdim}")


def run_benchmark(is_causal):
    print(f"is_causal: {is_causal}")
    for seq_len in [1024, 2048, 4096, 8192]:
        flops = 4 * head * batch * headdim * seq_len * seq_len // (2 if is_causal else 1)
        # Generate inputs in BHSD format (Batch, Head, Seq, Dim)
        q = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")

        if args.method in ["flash_attn", "sage_attn"]:
            # flash_attn_func and SageAttention NHD expect (Batch, Seq, Head, Dim)
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()

        if args.method == "flash_attn":
            fn = flash_attn_func
            kwargs = {"causal": is_causal}
        elif args.method == "sage_attn":
            fn = sageattn
            kwargs = {"tensor_layout": "NHD", "is_causal": is_causal}
        else:
            fn = sdpa
            kwargs = {"is_causal": is_causal}

        for i in range(5):
            fn(q, k, v, **kwargs)
        torch.cuda.synchronize()

        _, time = benchmark_forward(fn, q, k, v, repeats=10, verbose=False, desc="Triton", **kwargs)
        print(f"{seq_len} flops:{flops / time.mean * 1e-12}")


run_benchmark(is_causal=False)
run_benchmark(is_causal=True)

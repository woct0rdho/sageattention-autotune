import argparse
import csv
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path

import torch
from flash_attn import flash_attn_func

from sageattention.cuda_attn import _sageattn_configured
from sageattention.cuda_compile import _qattn_sm80
from sageattention.cutlass_attn import _CUTLASS_QK_QUANT_CONFIG, _sageattn_cutlass_configured
from sageattention.cutlass_compile import _qattn_cutlass_sm80
from sageattention.triton.quant_per_block import per_block_int8
from sageattention.triton.quant_per_thread import per_thread_int8

BlockConfig = tuple[int, int, int, int]
PreparedKernelInputs = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    float,
]

_DEFAULT_BLOCK_CONFIGS: tuple[BlockConfig, ...] = (
    (128, 64, 32, 64),
    (64, 64, 32, 64),
    (128, 32, 32, 32),
    (128, 64, 16, 64),
)


def _parse_block_config(value: str) -> BlockConfig:
    parts = tuple(int(part) for part in value.split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("block config must be 'blk_q,blk_k,warp_q,warp_k'")
    return parts


def _parse_block_configs(values: list[str] | None) -> tuple[BlockConfig, ...]:
    if not values:
        return _DEFAULT_BLOCK_CONFIGS
    return tuple(_parse_block_config(value) for value in values)


def _bench(fn, *args, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms = []
    for _ in range(repeats):
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    return {
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "stdev_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
    }


def _make_qkv(
    batch_size: int, num_heads: int, seq_len: int, head_dim: int, layout: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if layout == "HND":
        shape = (batch_size, num_heads, seq_len, head_dim)
    else:
        shape = (batch_size, seq_len, num_heads, head_dim)

    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v


def _to_nhd(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    return tensor.transpose(1, 2).contiguous() if layout == "HND" else tensor


def _flash_end_to_end(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return flash_attn_func(q, k, v, dropout_p=0.0, causal=False)


def _cuda_end_to_end(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: str,
    smooth_k: bool,
    block_config: BlockConfig,
) -> torch.Tensor:
    return _sageattn_configured(
        q,
        k,
        v,
        layout,
        False,
        "fp32",
        smooth_k,
        False,
        False,
        block_config,
    )


def _cutlass_end_to_end(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: str,
    smooth_k: bool,
    block_config: BlockConfig,
) -> torch.Tensor:
    return _sageattn_cutlass_configured(
        q,
        k,
        v,
        layout,
        smooth_k,
        False,
        block_config,
    )


def _prequantized_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: str,
    block_config: BlockConfig,
    per_block_quantization: bool,
    smooth_k: bool,
) -> PreparedKernelInputs:
    blk_q, blk_k, warp_q, warp_k = block_config
    seq_dim = 2 if layout == "HND" else 1
    k_mean = k.mean(dim=seq_dim, keepdim=True) if smooth_k else None
    if per_block_quantization:
        quant_blk_q, quant_blk_k = _CUTLASS_QK_QUANT_CONFIG
        q_int8, q_scale, k_int8, k_scale = per_block_int8(
            q,
            k,
            km=k_mean,
            BLKQ=quant_blk_q,
            BLKK=quant_blk_k,
            tensor_layout=layout,
        )
    else:
        q_int8, q_scale, k_int8, k_scale = per_thread_int8(
            q,
            k,
            km=k_mean,
            BLKQ=blk_q,
            WARPQ=warp_q,
            BLKK=blk_k,
            WARPK=warp_k,
            tensor_layout=layout,
        )
    return q_int8, k_int8, v, torch.empty_like(q), q_scale, k_scale, 1 if layout == "HND" else 0, q.size(-1) ** -0.5


def _cuda_kernel_only(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    layout_i: int,
    sm_scale: float,
    block_config: BlockConfig,
) -> torch.Tensor:
    blk_q, blk_k, warp_q, warp_k = block_config
    _qattn_sm80.qk_int8_sv_f16_accum_f32_attn(
        q_int8,
        k_int8,
        v,
        out,
        q_scale,
        k_scale,
        layout_i,
        False,
        sm_scale,
        blk_q,
        blk_k,
        warp_q,
        warp_k,
        False,
    )
    return out


def _cutlass_kernel_only(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    layout_i: int,
    sm_scale: float,
    block_config: BlockConfig,
) -> torch.Tensor:
    blk_q, blk_k, warp_q, warp_k = block_config
    _qattn_cutlass_sm80.qk_int8_sv_f16_accum_f32_attn_cutlass(
        q_int8,
        k_int8,
        v,
        out,
        q_scale,
        k_scale,
        layout_i,
        sm_scale,
        blk_q,
        blk_k,
        warp_q,
        warp_k,
        False,
    )
    return out


def _rel_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = actual.float() - expected.float()
    rel = (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected.float()).clamp(min=1e-6)).item()
    max_abs = diff.abs().max().item()
    return rel, max_abs


def _forward_flops(batch_size: int, num_heads: int, seq_len: int, head_dim: int) -> int:
    # QK^T and PV each perform 2 * B * H * N^2 * D effective FLOPs.
    return 4 * batch_size * num_heads * head_dim * seq_len * seq_len


def _tflops(flops: int, time_ms: float) -> float:
    return flops / time_ms * 1.0e-9


def _format_block_config(config: BlockConfig) -> str:
    return f"{config[0]}x{config[1]}x{config[2]}x{config[3]}"


def _write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_row(row: dict[str, object]) -> None:
    print(
        "{kind},{head_dim},{seq_len},{num_heads},{smooth_k},{block_config},{forward_flops},"
        "{flash_ms:.4f},{flash_mean_ms:.4f},{flash_stdev_ms:.4f},{flash_tflops:.3f},"
        "{cuda_ms:.4f},{cuda_mean_ms:.4f},{cuda_stdev_ms:.4f},{cuda_tflops:.3f},"
        "{cutlass_ms:.4f},{cutlass_mean_ms:.4f},{cutlass_stdev_ms:.4f},{cutlass_tflops:.3f},"
        "{cutlass_vs_cuda:.3f},"
        "{cutlass_vs_flash:.3f},{cutlass_speedup_vs_flash:.3f},{rel_err:.4g},{max_abs:.4g}".format(**row)
    )


def _benchmark_case(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: str,
    smooth_k: bool,
    block_config: BlockConfig,
    flash_stats: dict[str, float],
    warmup: int,
    repeats: int,
    include_end_to_end: bool,
    include_kernel_only: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    head_dim = q.size(-1)
    seq_len = q.size(2) if layout == "HND" else q.size(1)
    num_heads = q.size(1) if layout == "HND" else q.size(2)
    flops = _forward_flops(q.size(0), num_heads, seq_len, head_dim)
    block_config_text = _format_block_config(block_config)

    if include_end_to_end:
        out_cuda = _cuda_end_to_end(q, k, v, layout, smooth_k, block_config)
        out_cutlass = _cutlass_end_to_end(q, k, v, layout, smooth_k, block_config)
        rel_err, max_abs = _rel_error(out_cutlass, out_cuda)
        cuda_stats = _bench(
            _cuda_end_to_end,
            q,
            k,
            v,
            layout,
            smooth_k,
            block_config,
            warmup=warmup,
            repeats=repeats,
        )
        cutlass_stats = _bench(
            _cutlass_end_to_end,
            q,
            k,
            v,
            layout,
            smooth_k,
            block_config,
            warmup=warmup,
            repeats=repeats,
        )
        rows.append(
            {
                "kind": "end_to_end",
                "head_dim": head_dim,
                "seq_len": seq_len,
                "num_heads": num_heads,
                "smooth_k": smooth_k,
                "block_config": block_config_text,
                "forward_flops": flops,
                "flash_ms": flash_stats["median_ms"],
                "flash_mean_ms": flash_stats["mean_ms"],
                "flash_stdev_ms": flash_stats["stdev_ms"],
                "flash_tflops": _tflops(flops, flash_stats["median_ms"]),
                "cuda_ms": cuda_stats["median_ms"],
                "cuda_mean_ms": cuda_stats["mean_ms"],
                "cuda_stdev_ms": cuda_stats["stdev_ms"],
                "cuda_tflops": _tflops(flops, cuda_stats["median_ms"]),
                "cutlass_ms": cutlass_stats["median_ms"],
                "cutlass_mean_ms": cutlass_stats["mean_ms"],
                "cutlass_stdev_ms": cutlass_stats["stdev_ms"],
                "cutlass_tflops": _tflops(flops, cutlass_stats["median_ms"]),
                "cutlass_vs_cuda": cutlass_stats["median_ms"] / cuda_stats["median_ms"],
                "cutlass_vs_flash": cutlass_stats["median_ms"] / flash_stats["median_ms"],
                "cutlass_speedup_vs_flash": flash_stats["median_ms"] / cutlass_stats["median_ms"],
                "rel_err": rel_err,
                "max_abs": max_abs,
            }
        )

    if include_kernel_only:
        cuda_inputs = _prequantized_inputs(q, k, v, layout, block_config, False, smooth_k)
        cutlass_inputs = _prequantized_inputs(q, k, v, layout, block_config, True, smooth_k)
        _cuda_kernel_only(*cuda_inputs, block_config)
        _cutlass_kernel_only(*cutlass_inputs, block_config)
        torch.cuda.synchronize()
        rel_err, max_abs = _rel_error(cutlass_inputs[3], cuda_inputs[3])
        cuda_stats = _bench(
            _cuda_kernel_only,
            *cuda_inputs,
            block_config,
            warmup=warmup,
            repeats=repeats,
        )
        cutlass_stats = _bench(
            _cutlass_kernel_only,
            *cutlass_inputs,
            block_config,
            warmup=warmup,
            repeats=repeats,
        )
        rows.append(
            {
                "kind": "kernel_only",
                "head_dim": head_dim,
                "seq_len": seq_len,
                "num_heads": num_heads,
                "smooth_k": smooth_k,
                "block_config": block_config_text,
                "forward_flops": flops,
                "flash_ms": flash_stats["median_ms"],
                "flash_mean_ms": flash_stats["mean_ms"],
                "flash_stdev_ms": flash_stats["stdev_ms"],
                "flash_tflops": _tflops(flops, flash_stats["median_ms"]),
                "cuda_ms": cuda_stats["median_ms"],
                "cuda_mean_ms": cuda_stats["mean_ms"],
                "cuda_stdev_ms": cuda_stats["stdev_ms"],
                "cuda_tflops": _tflops(flops, cuda_stats["median_ms"]),
                "cutlass_ms": cutlass_stats["median_ms"],
                "cutlass_mean_ms": cutlass_stats["mean_ms"],
                "cutlass_stdev_ms": cutlass_stats["stdev_ms"],
                "cutlass_tflops": _tflops(flops, cutlass_stats["median_ms"]),
                "cutlass_vs_cuda": cutlass_stats["median_ms"] / cuda_stats["median_ms"],
                "cutlass_vs_flash": cutlass_stats["median_ms"] / flash_stats["median_ms"],
                "cutlass_speedup_vs_flash": flash_stats["median_ms"] / cutlass_stats["median_ms"],
                "rel_err": rel_err,
                "max_abs": max_abs,
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FlashAttention and SageAttention forward paths.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[2048, 4096, 8192])
    parser.add_argument("--head-dims", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--layout", default="NHD", choices=["HND", "NHD"])
    parser.add_argument(
        "--block-configs",
        nargs="*",
        help="Configs as blk_q,blk_k,warp_q,warp_k. Defaults to CUTLASS-supported configs.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--mode", choices=["all", "end-to-end", "kernel-only"], default="all")
    parser.add_argument("--smooth-k", nargs="+", choices=["false", "true"], default=["false"])
    parser.add_argument("--csv", type=Path, help="Optional path to write benchmark rows as CSV.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    block_configs = _parse_block_configs(args.block_configs)
    include_end_to_end = args.mode in ("all", "end-to-end")
    include_kernel_only = args.mode in ("all", "kernel-only")
    smooth_k_values = tuple(value == "true" for value in args.smooth_k)

    props = torch.cuda.get_device_properties(0)
    print(f"device={props.name} capability=sm{props.major}{props.minor}")
    print(
        f"batch={args.batch_size} heads={args.num_heads} layout={args.layout} "
        f"smooth_k={list(smooth_k_values)} warmup={args.warmup} repeats={args.repeats}"
    )
    print("primary timing columns flash_ms/cutlass_ms report medians; CUDA columns are reference diagnostics")
    print("CUTLASS uses Q32/K64 metadata for head dimensions 64 and 128")
    print("FlashAttention baseline=library-selected best available kernel; it is not forced to the Sage block shape")
    print("TFLOPS use 4*batch*heads*head_dim*seq_len^2 effective dense forward FLOPs and median CUDA-event time")
    print(
        "kind,head_dim,seq_len,num_heads,smooth_k,block_config,forward_flops,"
        "flash_ms,flash_mean_ms,flash_stdev_ms,flash_tflops,"
        "cuda_ms,cuda_mean_ms,cuda_stdev_ms,cuda_tflops,"
        "cutlass_ms,cutlass_mean_ms,cutlass_stdev_ms,cutlass_tflops,"
        "cutlass_vs_cuda,cutlass_vs_flash,cutlass_speedup_vs_flash,rel_err,max_abs"
    )

    rows: list[dict[str, object]] = []
    for num_heads in args.num_heads:
        for head_dim in args.head_dims:
            for seq_len in args.seq_lens:
                q, k, v = _make_qkv(args.batch_size, num_heads, seq_len, head_dim, args.layout)
                flash_stats = _bench(
                    _flash_end_to_end,
                    _to_nhd(q, args.layout),
                    _to_nhd(k, args.layout),
                    _to_nhd(v, args.layout),
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
                for smooth_k in smooth_k_values:
                    for block_config in block_configs:
                        case_rows = _benchmark_case(
                            q=q,
                            k=k,
                            v=v,
                            layout=args.layout,
                            smooth_k=smooth_k,
                            block_config=block_config,
                            flash_stats=flash_stats,
                            warmup=args.warmup,
                            repeats=args.repeats,
                            include_end_to_end=include_end_to_end,
                            include_kernel_only=include_kernel_only,
                        )
                        rows.extend(case_rows)
                        for row in case_rows:
                            _print_row(row)

    if args.csv is not None:
        _write_rows(args.csv, rows)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

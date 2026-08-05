import argparse
import csv
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import torch
from flash_attn import flash_attn_func

from sageattention.cutlass_attn import _sageattn_cutlass_configured
from sageattention.cutlass_bwd import (
    _BWD_CONFIG,
    _BWD_CONFIGS,
    _BWD_QUANTIZATION_POLICY_IDS,
    _sageattn_cutlass_bwd_configured,
)
from sageattention.cutlass_compile import _qattn_cutlass_sm80
from sageattention.triton.quant_per_block import per_block_int8

BlockConfig = tuple[int, int, int, int]
BackwardCandidate = tuple[BlockConfig, str]
_FORWARD_QK_CONFIG: BlockConfig = (64, 64, 32, 64)


def _parse_block_config(value: str) -> BlockConfig:
    try:
        parts = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("block config must contain integers") from exc
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("block config must be 'blk_q,blk_k,bwd_block_m,bwd_block_n'")
    return parts


def _parse_block_configs(values: list[str] | None) -> tuple[BlockConfig, ...]:
    configs = tuple(_BWD_CONFIGS) if not values else tuple(_parse_block_config(value) for value in values)
    unsupported = tuple(config for config in configs if config not in _BWD_CONFIGS)
    if unsupported:
        supported = ", ".join(_format_block_config(config) for config in _BWD_CONFIGS)
        invalid = ", ".join(_format_block_config(config) for config in unsupported)
        raise ValueError(f"unsupported backward block config(s): {invalid}; supported configs: {supported}")
    return configs


def _make_candidates(block_configs: tuple[BlockConfig, ...], policies: list[str]) -> tuple[BackwardCandidate, ...]:
    return tuple(
        (config, policy)
        for config in block_configs
        for policy in policies
        if policy == "dynamic" or config == _BWD_CONFIG
    )


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


def _make_inputs(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if layout == "HND":
        shape = (batch_size, num_heads, seq_len, head_dim)
    else:
        shape = (batch_size, seq_len, num_heads, head_dim)

    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q)
    return q, k, v, dout


def _to_nhd(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    return tensor.transpose(1, 2).contiguous() if layout == "HND" else tensor


def _prepare_flash_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_flash = _to_nhd(q, layout).detach().requires_grad_(True)
    k_flash = _to_nhd(k, layout).detach().requires_grad_(True)
    v_flash = _to_nhd(v, layout).detach().requires_grad_(True)
    dout_flash = _to_nhd(dout, layout)
    output = flash_attn_func(
        q_flash,
        k_flash,
        v_flash,
        dropout_p=0.0,
        softmax_scale=q.size(-1) ** -0.5,
        causal=False,
    )
    return q_flash, k_flash, v_flash, output, dout_flash


def _flash_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    dout: torch.Tensor,
) -> None:
    q.grad = None
    k.grad = None
    v.grad = None
    output.backward(dout, retain_graph=True)


def _prepare_sage_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        result = _sageattn_cutlass_configured(
            q,
            k,
            v,
            layout,
            False,
            True,
            _FORWARD_QK_CONFIG,
        )
    output, lse = result
    return output, lse


def _sage_end_to_end_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    layout: str,
    block_config: BlockConfig,
    quantization_policy: str,
) -> None:
    with torch.inference_mode():
        _sageattn_cutlass_bwd_configured(q, k, v, output, dout, lse, layout, block_config, quantization_policy)


def _prepare_prequantized_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: str,
    block_config: BlockConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    float,
]:
    blk_q, blk_k, _, _ = block_config
    with torch.inference_mode():
        q_int8, q_scale, k_int8, k_scale = per_block_int8(
            q,
            k,
            km=None,
            BLKQ=blk_q,
            BLKK=blk_k,
            tensor_layout=layout,
        )
    return (
        q_int8,
        k_int8,
        q_scale,
        k_scale,
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        1 if layout == "HND" else 0,
        q.size(-1) ** -0.5,
    )


def _sage_kernel_only_backward(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    layout_i: int,
    sm_scale: float,
    block_config: BlockConfig,
    quantization_policy: str,
) -> None:
    blk_q, blk_k, bwd_block_m, bwd_block_n = block_config
    with torch.inference_mode():
        _qattn_cutlass_sm80.qk_int8_sv_f16_accum_f32_attn_bwd_cutlass(
            q_int8,
            k_int8,
            q_scale,
            k_scale,
            v,
            output,
            dout,
            lse,
            dq,
            dk,
            dv,
            layout_i,
            sm_scale,
            blk_q,
            blk_k,
            bwd_block_m,
            bwd_block_n,
            _BWD_QUANTIZATION_POLICY_IDS[quantization_policy],
        )


def _backward_flops(batch_size: int, num_heads: int, seq_len: int, head_dim: int) -> int:
    # Effective dense backward FLOPs: dV, dP, dQ, and dK each contribute one matmul.
    return 8 * batch_size * num_heads * head_dim * seq_len * seq_len


def _tflops(flops: int, time_ms: float) -> float:
    return flops / time_ms * 1.0e-9


def _format_block_config(config: BlockConfig) -> str:
    return f"{config[0]}x{config[1]}x{config[2]}x{config[3]}"


def _make_row(
    *,
    kind: str,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    layout: str,
    block_config: BlockConfig,
    quantization_policy: str,
    flops: int,
    flash_stats: dict[str, float],
    sage_stats: dict[str, float],
) -> dict[str, object]:
    flash_ms = flash_stats["median_ms"]
    sage_ms = sage_stats["median_ms"]
    return {
        "kind": kind,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "seq_len": seq_len,
        "layout": layout,
        "block_config": _format_block_config(block_config),
        "quantization_policy": quantization_policy,
        "backward_flops": flops,
        "flash_ms": flash_ms,
        "flash_mean_ms": flash_stats["mean_ms"],
        "flash_stdev_ms": flash_stats["stdev_ms"],
        "flash_tflops": _tflops(flops, flash_ms),
        "sage_ms": sage_ms,
        "sage_mean_ms": sage_stats["mean_ms"],
        "sage_stdev_ms": sage_stats["stdev_ms"],
        "sage_tflops": _tflops(flops, sage_ms),
        "sage_speedup": flash_ms / sage_ms,
        "sage_rank": 0,
    }


def _assign_sage_ranks(rows: list[dict[str, object]]) -> None:
    kinds = {str(row["kind"]) for row in rows}
    for kind in kinds:
        kind_rows = [row for row in rows if row["kind"] == kind]
        for rank, row in enumerate(sorted(kind_rows, key=lambda item: cast(float, item["sage_ms"])), start=1):
            row["sage_rank"] = rank


def _benchmark_case(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    layout: str,
    candidates: tuple[BackwardCandidate, ...],
    warmup: int,
    repeats: int,
    include_end_to_end: bool,
    include_kernel_only: bool,
) -> list[dict[str, object]]:
    batch_size = q.size(0)
    num_heads = q.size(1) if layout == "HND" else q.size(2)
    seq_len = q.size(2) if layout == "HND" else q.size(1)
    head_dim = q.size(-1)
    flops = _backward_flops(batch_size, num_heads, seq_len, head_dim)

    flash_args = _prepare_flash_backward(q, k, v, dout, layout)
    flash_stats = _bench(_flash_backward, *flash_args, warmup=warmup, repeats=repeats)

    rows: list[dict[str, object]] = []
    for block_config, quantization_policy in candidates:
        output, lse = _prepare_sage_forward(q, k, v, layout)

        if include_end_to_end:
            sage_stats = _bench(
                _sage_end_to_end_backward,
                q,
                k,
                v,
                output,
                dout,
                lse,
                layout,
                block_config,
                quantization_policy,
                warmup=warmup,
                repeats=repeats,
            )
            rows.append(
                _make_row(
                    kind="end_to_end",
                    batch_size=batch_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    seq_len=seq_len,
                    layout=layout,
                    block_config=block_config,
                    quantization_policy=quantization_policy,
                    flops=flops,
                    flash_stats=flash_stats,
                    sage_stats=sage_stats,
                )
            )

        if include_kernel_only:
            q_int8, k_int8, q_scale, k_scale, dq, dk, dv, layout_i, sm_scale = _prepare_prequantized_inputs(
                q, k, v, layout, block_config
            )
            sage_stats = _bench(
                _sage_kernel_only_backward,
                q_int8,
                k_int8,
                q_scale,
                k_scale,
                v,
                output,
                dout,
                lse,
                dq,
                dk,
                dv,
                layout_i,
                sm_scale,
                block_config,
                quantization_policy,
                warmup=warmup,
                repeats=repeats,
            )
            rows.append(
                _make_row(
                    kind="kernel_only",
                    batch_size=batch_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    seq_len=seq_len,
                    layout=layout,
                    block_config=block_config,
                    quantization_policy=quantization_policy,
                    flops=flops,
                    flash_stats=flash_stats,
                    sage_stats=sage_stats,
                )
            )

    _assign_sage_ranks(rows)
    return rows


def _write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_row(row: dict[str, object]) -> None:
    print(
        "{kind},{batch_size},{num_heads},{head_dim},{seq_len},{layout},{block_config},{quantization_policy},{backward_flops},"
        "{flash_ms:.4f},{flash_mean_ms:.4f},{flash_stdev_ms:.4f},{flash_tflops:.3f},"
        "{sage_ms:.4f},{sage_mean_ms:.4f},{sage_stdev_ms:.4f},{sage_tflops:.3f},"
        "{sage_speedup:.3f},{sage_rank}".format(**row)
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if any(value <= 0 for value in (*args.num_heads, *args.seq_lens)):
        raise ValueError("number of heads and sequence lengths must be positive")
    if any(head_dim not in (64, 128) for head_dim in args.head_dims):
        raise ValueError("head dimensions must be 64 or 128")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark SageAttention CUTLASS backward against FlashAttention backward."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[4096, 8192])
    parser.add_argument("--head-dims", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--layout", choices=["HND", "NHD"], default="NHD")
    parser.add_argument(
        "--block-configs",
        nargs="*",
        help="Configs as blk_q,blk_k,bwd_block_m,bwd_block_n. Defaults to all generated backward configs.",
    )
    parser.add_argument(
        "--quantization-policies",
        nargs="+",
        choices=tuple(_BWD_QUANTIZATION_POLICY_IDS),
        default=["dynamic"],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--mode", choices=["all", "end-to-end", "kernel-only"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, help="Optional path to write benchmark rows as CSV.")
    args = parser.parse_args()

    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    block_configs = _parse_block_configs(args.block_configs)
    candidates = _make_candidates(block_configs, args.quantization_policies)
    if not candidates:
        raise ValueError("No supported block-config and quantization-policy combinations were selected.")
    include_end_to_end = args.mode in ("all", "end-to-end")
    include_kernel_only = args.mode in ("all", "kernel-only")

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(f"device={props.name} capability=sm{props.major}{props.minor}")
    print(
        f"batch={args.batch_size} heads={args.num_heads} head_dims={args.head_dims} "
        f"seq_lens={args.seq_lens} layout={args.layout} warmup={args.warmup} repeats={args.repeats}"
    )
    print(f"block_configs={[_format_block_config(config) for config in block_configs]}")
    print(f"quantization_policies={args.quantization_policies}")
    print("TFLOPS use 8*batch*heads*head_dim*seq_len^2 effective dense backward FLOPs and median CUDA-event time")
    print(
        "end_to_end includes Sage Q/K quantization and output allocation; kernel_only uses prequantized Q/K and reused dQ/dK/dV"
    )
    print(
        "Sage forward setup is not timed and uses the same block config as Sage backward; sage_speedup > 1 means Sage is faster"
    )
    print(
        "kind,batch_size,num_heads,head_dim,seq_len,layout,block_config,quantization_policy,backward_flops,"
        "flash_ms,flash_mean_ms,flash_stdev_ms,flash_tflops,"
        "sage_ms,sage_mean_ms,sage_stdev_ms,sage_tflops,sage_speedup,sage_rank"
    )

    rows: list[dict[str, object]] = []
    for num_heads in args.num_heads:
        for head_dim in args.head_dims:
            for seq_len in args.seq_lens:
                q, k, v, dout = _make_inputs(args.batch_size, num_heads, seq_len, head_dim, args.layout)
                case_rows = _benchmark_case(
                    q=q,
                    k=k,
                    v=v,
                    dout=dout,
                    layout=args.layout,
                    candidates=candidates,
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

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
from sageattention.cutlass_bwd import _BWD_CONFIGS, _BWD_CONFIGS_BY_HEAD_DIM
from sageattention.cutlass_compile import _qattn_cutlass_sm80
from sageattention.triton.cutlass_bwd import convert_dq, preprocess_delta_zero_dq
from sageattention.triton.quant_per_block import per_block_int8
from sageattention.utils import _lse_correction

BlockConfig = tuple[int, int, int, int]
BackwardCandidate = BlockConfig
PreparedQuantizedQK = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]
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
    supported_configs = _BWD_CONFIGS
    configs = supported_configs if not values else tuple(_parse_block_config(value) for value in values)
    unsupported = tuple(config for config in configs if config not in supported_configs)
    if unsupported:
        supported = ", ".join(_format_block_config(config) for config in supported_configs)
        invalid = ", ".join(_format_block_config(config) for config in unsupported)
        raise ValueError(f"unsupported backward block config(s): {invalid}; supported configs: {supported}")
    return configs


def _make_head_dim_candidates(
    head_dim: int,
    block_configs: tuple[BlockConfig, ...],
) -> tuple[BackwardCandidate, ...]:
    supported = _BWD_CONFIGS_BY_HEAD_DIM[head_dim]
    return tuple(config for config in block_configs if config in supported)


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
    smooth_k: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        result = _sageattn_cutlass_configured(
            q,
            k,
            v,
            layout,
            smooth_k,
            True,
            _FORWARD_QK_CONFIG,
        )
    output, lse = result
    return output, lse


def _prepare_quantized_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    layout: str,
    block_config: BlockConfig,
    smooth_k: bool,
) -> PreparedQuantizedQK:
    blk_q, blk_k, _, _ = block_config
    # Keep the forward-compatible Q32/K64 Q/K representation outside the timed
    # backward region until the public forward saves and routes these tensors.
    with torch.inference_mode():
        seq_dim = 2 if layout == "HND" else 1
        k_mean = k.mean(dim=seq_dim, keepdim=True) if smooth_k else None
        q_int8, q_scale, k_int8, k_scale = per_block_int8(
            q,
            k,
            km=k_mean,
            BLKQ=blk_q,
            BLKK=blk_k,
            tensor_layout=layout,
        )
        return q_int8, q_scale, k_int8, k_scale, k_mean


def _prepare_kernel_lse(
    q: torch.Tensor,
    lse: torch.Tensor,
    k_mean: torch.Tensor | None,
    layout: str,
) -> torch.Tensor:
    if k_mean is None:
        return lse
    head_dim_index = 1 if layout == "HND" else 2
    return (lse - _lse_correction(q, k_mean, layout, head_dim_index) * (q.size(-1) ** -0.5)).contiguous()


def _prepare_backward_workspaces(
    v: torch.Tensor,
    output: torch.Tensor,
    dout: torch.Tensor,
    layout: str,
    smooth_k: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        return preprocess_delta_zero_dq(output, dout, v, layout, smooth_k)


def _sage_kernel_only_backward(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    dq_accum: torch.Tensor,
    ds_sum: torch.Tensor,
    do_int8: torch.Tensor,
    do_scale: torch.Tensor,
    ds_q_factors: torch.Tensor,
    ds_k_factors: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    layout_i: int,
    sm_scale: float,
    block_config: BlockConfig,
    k_mean: torch.Tensor | None,
    smooth_k: bool,
) -> None:
    blk_q, blk_k, bwd_block_m, bwd_block_n = block_config
    with torch.inference_mode():
        _qattn_cutlass_sm80.qk_int8_sv_f16_accum_f32_attn_bwd_cutlass(
            q_int8,
            k_int8,
            q_scale,
            k_scale,
            v,
            dout,
            lse,
            delta,
            dq_accum,
            ds_sum,
            do_int8,
            do_scale,
            ds_q_factors,
            ds_k_factors,
            dk,
            dv,
            layout_i,
            sm_scale,
            blk_q,
            blk_k,
            bwd_block_m,
            bwd_block_n,
            smooth_k,
        )
        convert_dq(
            dq_accum,
            dq,
            "HND" if layout_i else "NHD",
            ds_sum if smooth_k else None,
            k_mean,
        )


def _sage_prequantized_end_to_end_backward(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    layout: str,
    block_config: BlockConfig,
    k_mean: torch.Tensor | None,
    smooth_k: bool,
) -> None:
    with torch.inference_mode():
        delta, dq_accum, ds_sum, do_int8, do_scale, ds_q_factors, ds_k_factors = _prepare_backward_workspaces(
            v,
            output,
            dout,
            layout,
            smooth_k,
        )
        _sage_kernel_only_backward(
            q_int8,
            k_int8,
            q_scale,
            k_scale,
            v,
            output,
            dout,
            lse,
            delta,
            dq_accum,
            ds_sum,
            do_int8,
            do_scale,
            ds_q_factors,
            ds_k_factors,
            torch.empty_like(q),
            torch.empty_like(k),
            torch.empty_like(v),
            1 if layout == "HND" else 0,
            q.size(-1) ** -0.5,
            block_config,
            k_mean,
            smooth_k,
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
    smooth_k: bool,
    block_config: BlockConfig,
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
        "smooth_k": smooth_k,
        "block_config": _format_block_config(block_config),
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
    groups = {(str(row["kind"]), bool(row["smooth_k"])) for row in rows}
    for kind, smooth_k in groups:
        kind_rows = [row for row in rows if row["kind"] == kind and row["smooth_k"] == smooth_k]
        for rank, row in enumerate(sorted(kind_rows, key=lambda item: cast(float, item["sage_ms"])), start=1):
            row["sage_rank"] = rank


def _benchmark_case(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    layout: str,
    smooth_k: bool,
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
    for block_config in candidates:
        output, lse = _prepare_sage_forward(q, k, v, layout, smooth_k)
        q_int8, q_scale, k_int8, k_scale, k_mean = _prepare_quantized_qk(q, k, layout, block_config, smooth_k)
        kernel_lse = _prepare_kernel_lse(q, lse, k_mean, layout)

        if include_end_to_end:
            sage_stats = _bench(
                _sage_prequantized_end_to_end_backward,
                q_int8,
                k_int8,
                q_scale,
                k_scale,
                q,
                k,
                v,
                output,
                dout,
                kernel_lse,
                layout,
                block_config,
                k_mean,
                smooth_k,
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
                    smooth_k=smooth_k,
                    block_config=block_config,
                    flops=flops,
                    flash_stats=flash_stats,
                    sage_stats=sage_stats,
                )
            )

        if include_kernel_only:
            delta, dq_accum, ds_sum, do_int8, do_scale, ds_q_factors, ds_k_factors = _prepare_backward_workspaces(
                v,
                output,
                dout,
                layout,
                smooth_k,
            )
            layout_i = 1 if layout == "HND" else 0
            sm_scale = q.size(-1) ** -0.5
            sage_stats = _bench(
                _sage_kernel_only_backward,
                q_int8,
                k_int8,
                q_scale,
                k_scale,
                v,
                output,
                dout,
                kernel_lse,
                delta,
                dq_accum,
                ds_sum,
                do_int8,
                do_scale,
                ds_q_factors,
                ds_k_factors,
                torch.empty_like(q),
                torch.empty_like(k),
                torch.empty_like(v),
                layout_i,
                sm_scale,
                block_config,
                k_mean,
                smooth_k,
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
                    smooth_k=smooth_k,
                    block_config=block_config,
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
        "{kind},{batch_size},{num_heads},{head_dim},{seq_len},{layout},{smooth_k},{block_config},{backward_flops},"
        "{flash_ms:.4f},{flash_mean_ms:.4f},{flash_stdev_ms:.4f},{flash_tflops:.3f},"
        "{sage_ms:.4f},{sage_mean_ms:.4f},{sage_stdev_ms:.4f},{sage_tflops:.3f},"
        "{sage_speedup:.3f},{sage_rank}".format(**row)
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if any(value <= 0 for value in (*args.num_heads, *args.seq_lens)):
        raise ValueError("number of heads and sequence lengths must be positive")
    if any(head_dim not in _BWD_CONFIGS_BY_HEAD_DIM for head_dim in args.head_dims):
        raise ValueError("CUTLASS backward currently supports head dimensions 64 and 128")
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
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--mode", choices=["all", "end-to-end", "kernel-only"], default="all")
    parser.add_argument("--smooth-k", nargs="+", choices=["false", "true"], default=["false"])
    parser.add_argument("--csv", type=Path, help="Optional path to write benchmark rows as CSV.")
    args = parser.parse_args()

    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    block_configs = _parse_block_configs(args.block_configs)
    include_end_to_end = args.mode in ("all", "end-to-end")
    include_kernel_only = args.mode in ("all", "kernel-only")
    smooth_k_values = tuple(value == "true" for value in args.smooth_k)

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(f"device={props.name} capability=sm{props.major}{props.minor}")
    print(
        f"batch={args.batch_size} heads={args.num_heads} head_dims={args.head_dims} "
        f"seq_lens={args.seq_lens} layout={args.layout} warmup={args.warmup} repeats={args.repeats}"
    )
    print(f"block_configs={[_format_block_config(config) for config in block_configs]}")
    print("dS scale policy=precomputed separable predictor")
    print(f"smooth_k={list(smooth_k_values)}")
    print("FlashAttention baseline=library-selected best available kernel")
    print("TFLOPS use 8*batch*heads*head_dim*seq_len^2 effective dense backward FLOPs and median CUDA-event time")
    print(
        "end_to_end excludes Q/K quantization: backward-compatible Q/K INT8 tensors and scales are prepared outside timing; it includes preprocessing, output/workspace allocation, the main kernel, and dQ conversion"
    )
    print(
        "Sage forward setup is not timed and uses the fixed Q32/K64 artifact domain; its saved tensors are still discarded, so benchmark preparation remains a reuse proxy until forward state is routed into backward; sage_speedup > 1 means Sage is faster"
    )
    print(
        "kind,batch_size,num_heads,head_dim,seq_len,layout,smooth_k,block_config,backward_flops,"
        "flash_ms,flash_mean_ms,flash_stdev_ms,flash_tflops,"
        "sage_ms,sage_mean_ms,sage_stdev_ms,sage_tflops,sage_speedup,sage_rank"
    )

    rows: list[dict[str, object]] = []
    for num_heads in args.num_heads:
        for head_dim in args.head_dims:
            candidates = _make_head_dim_candidates(head_dim, block_configs)
            if not candidates:
                selected = ", ".join(_format_block_config(config) for config in block_configs)
                raise ValueError(f"No selected backward block config supports head dimension {head_dim}: {selected}")
            for seq_len in args.seq_lens:
                q, k, v, dout = _make_inputs(args.batch_size, num_heads, seq_len, head_dim, args.layout)
                for smooth_k in smooth_k_values:
                    case_rows = _benchmark_case(
                        q=q,
                        k=k,
                        v=v,
                        dout=dout,
                        layout=args.layout,
                        smooth_k=smooth_k,
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

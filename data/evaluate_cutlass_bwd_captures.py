"""Replay retained captures and reconstruct active-kernel quantization stages.

The P/dS statistics and FP32 decomposition model the CUDA source arithmetic; they are
not values exported by the kernel. Sequence lengths above a capture's source length
are deterministic tiled controls and are labeled as such in both CSV outputs.
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func

from sageattention.cutlass_bwd import sageattn_cutlass_bwd
from sageattention.triton.cutlass_bwd import preprocess_delta_zero_dq
from sageattention.triton.quant_per_block import per_block_int8
from sageattention.utils import _lse_correction

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_CAPTURES = (
    DATA_DIR / "sdxl_periodic_ds_inputs_training_dout_latent128.pt",
    DATA_DIR / "sdxl_periodic_ds_inputs_training_dout_latent128_seed2.pt",
)
DEFAULT_OUTPUT = DATA_DIR / "cutlass_bwd_predictor_capture_accuracy.csv"
DEFAULT_HISTOGRAM_OUTPUT = DATA_DIR / "cutlass_bwd_predictor_scale_histograms.csv"
DEFAULT_SEQUENCE_LENGTHS = (512, 513, 4096, 8192)

Q_BLOCK = 32
K_BLOCK = 64
P_K_BLOCK = 16
DS_PREDICTOR_GUARD = 1.5
INT8_SCALE_INV = float.fromhex("0x1.010122p-7")
INT8_SCALE_FLOOR = 2.0**-126
STRICT_COS_MIN = 0.998
STRICT_REL_MAX = 0.06
STRICT_MAX_ABS = 0.2
HISTOGRAM_EDGES = tuple(float(value) for value in range(-128, 9, 4))
QUANTILE_POINTS = (
    ("min", 0.0),
    ("p01", 0.01),
    ("p10", 0.10),
    ("p50", 0.50),
    ("p90", 0.90),
    ("p99", 0.99),
    ("p999", 0.999),
    ("max", 1.0),
)


def metric(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    actual_float = actual.float()
    expected_float = expected.float()
    diff = actual_float - expected_float
    cosine = F.cosine_similarity(actual_float.flatten(), expected_float.flatten(), dim=0).item()
    relative = (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected_float).clamp_min(1e-6)).item()
    return cosine, relative, diff.abs().max().item()


def sequence_variant(tensor: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, str]:
    source_seq_len = tensor.size(1)
    if seq_len <= source_seq_len:
        kind = "exact" if seq_len == source_seq_len else "slice"
        return tensor[:, :seq_len].contiguous(), kind
    repeats = math.ceil(seq_len / source_seq_len)
    return tensor.repeat(1, repeats, 1, 1)[:, :seq_len].contiguous(), "tiled"


def head_tensor(tensor: torch.Tensor, head: int) -> torch.Tensor:
    return tensor if head < 0 else tensor[:, :, head : head + 1, :]


def quantile_summary(values: torch.Tensor, prefix: str) -> dict[str, float]:
    values = values.float().flatten()
    if values.numel() == 0:
        return {f"{prefix}_{name}": float("nan") for name, _ in QUANTILE_POINTS}
    points = torch.tensor([point for _, point in QUANTILE_POINTS], dtype=torch.float32)
    quantiles = torch.quantile(values, points)
    return {f"{prefix}_{name}": float(quantiles[index]) for index, (name, _) in enumerate(QUANTILE_POINTS)}


def scale_histogram_rows(
    values: torch.Tensor,
    metadata: dict[str, Any],
    head: int,
    scale_kind: str,
) -> list[dict[str, Any]]:
    values = values.float().flatten().clamp_min(INT8_SCALE_FLOOR)
    log2_values = torch.log2(values)
    edges = torch.tensor(HISTOGRAM_EDGES, dtype=torch.float32)
    log2_values = log2_values.clamp(min=edges[0].item(), max=edges[-1].item() - 1e-5)
    counts = torch.histogram(log2_values, bins=edges).hist.to(torch.int64)
    total = max(int(counts.sum()), 1)
    return [
        {
            **metadata,
            "head": head,
            "scale_kind": scale_kind,
            "bin_index": index,
            "log2_lower": HISTOGRAM_EDGES[index],
            "log2_upper": HISTOGRAM_EDGES[index + 1],
            "count": int(count),
            "rate": int(count) / total,
        }
        for index, count in enumerate(counts)
    ]


def strict_gate_failures(
    dq: tuple[float, float, float],
    dk: tuple[float, float, float],
    dv: tuple[float, float, float],
    finite: bool,
) -> list[str]:
    failures: list[str] = []
    if not finite:
        failures.append("nonfinite")
    for name, values in (("dq", dq), ("dk", dk), ("dv", dv)):
        cosine, relative, maximum = values
        if cosine <= STRICT_COS_MIN:
            failures.append(f"{name}_cos")
        if relative >= STRICT_REL_MAX:
            failures.append(f"{name}_rel")
        if maximum >= STRICT_MAX_ABS:
            failures.append(f"{name}_max_abs")
    return failures


def quantization_telemetry(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    reference_grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    actual_grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    metadata: dict[str, Any],
    smooth_k: bool,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    seq_len = q.size(1)
    heads = q.size(2)
    head_dim = q.size(3)
    sm_scale = head_dim**-0.5
    q_blocks = math.ceil(seq_len / Q_BLOCK)
    k_blocks = math.ceil(seq_len / K_BLOCK)
    p_k_blocks = math.ceil(seq_len / P_K_BLOCK)

    k_mean = k.mean(dim=1, keepdim=True) if smooth_k else None
    q_i8, q_scale, k_i8, k_scale = per_block_int8(
        q,
        k,
        km=k_mean,
        BLKQ=Q_BLOCK,
        BLKK=K_BLOCK,
        tensor_layout="NHD",
    )
    delta, dq_accum, ds_sum, do_i8, do_scale, ds_q_factors, ds_k_factors = preprocess_delta_zero_dq(
        output,
        dout,
        v,
        "NHD",
    )
    del dq_accum, ds_sum

    q_i8_h = q_i8[0].permute(1, 0, 2).contiguous()
    k_i8_h = k_i8[0].permute(1, 0, 2).contiguous()
    q_scale_h = q_scale[0]
    k_scale_h = k_scale[0]
    dout_h = dout[0].permute(1, 0, 2).float()
    value_h = v[0].permute(1, 0, 2).float()
    kernel_lse = lse
    if k_mean is not None:
        kernel_lse = lse - _lse_correction(q, k_mean, "NHD", 2) * sm_scale
    lse_h = kernel_lse[0].float()
    delta_h = delta[0]
    ds_q_h = ds_q_factors[0]
    ds_k_h = ds_k_factors[0]

    q_factor_indices = torch.arange(q_blocks, device=q.device)
    q_factor_next = (q_factor_indices + 1).clamp_max(q_blocks - 1)
    do_l2_max = torch.maximum(ds_q_h[:, q_factor_indices, 0], ds_q_h[:, q_factor_next, 0])
    delta_abs_max = torch.maximum(ds_q_h[:, q_factor_indices, 1], ds_q_h[:, q_factor_next, 1])
    predicted_ds_max = (
        DS_PREDICTOR_GUARD
        * sm_scale
        / seq_len
        * (do_l2_max[:, :, None] * ds_k_h[:, None, :] + delta_abs_max[:, :, None])
    )
    ds_scales = predicted_ds_max * INT8_SCALE_INV + INT8_SCALE_FLOOR

    q_scale_by_row = q_scale_h.repeat_interleave(Q_BLOCK, dim=1)[:, :seq_len]
    k_scale_by_row = k_scale_h.repeat_interleave(K_BLOCK, dim=1)[:, :seq_len]
    do_scale_by_element = do_scale[0].repeat_interleave(Q_BLOCK, dim=1)[:, :seq_len]
    do_scale_by_element = do_scale_by_element.repeat_interleave(16, dim=2)
    q_dequant = q_i8_h.float() * q_scale_by_row[:, :, None]
    k_dequant = k_i8_h.float() * k_scale_by_row[:, :, None]
    do_dequant = do_i8[0].float() * do_scale_by_element
    stage_accumulators = {
        name: torch.zeros_like(q_dequant)
        for name in (
            "qk_fp32_dq",
            "qk_fp32_dk",
            "qk_fp32_dv",
            "dynamic_ds_fp32_dq",
            "dynamic_ds_fp32_dk",
            "predictor_ds_fp32_dq",
            "predictor_ds_fp32_dk",
            "p_quant_fp32_dv",
            "p_do_quant_fp32_dv",
        )
    }

    q_zero = (q_i8_h == 0).sum(dim=(1, 2), dtype=torch.int64)
    k_zero = (k_i8_h == 0).sum(dim=(1, 2), dtype=torch.int64)
    do_zero = (do_i8[0] == 0).sum(dim=(1, 2), dtype=torch.int64)
    q_count = seq_len * head_dim
    k_count = seq_len * head_dim
    do_count = seq_len * head_dim

    p_count = torch.zeros(heads, device=q.device, dtype=torch.int64)
    p_zero = torch.zeros_like(p_count)
    p_saturation = torch.zeros_like(p_count)
    p_clipping = torch.zeros_like(p_count)
    ds_count = torch.zeros_like(p_count)
    ds_zero = torch.zeros_like(p_count)
    ds_saturation = torch.zeros_like(p_count)
    ds_clipping = torch.zeros_like(p_count)
    telemetry_finite = torch.ones(heads, device=q.device, dtype=torch.bool)
    p_scale_batches: list[torch.Tensor] = []
    ds_ratio_batches: list[torch.Tensor] = []

    k_float_transposed = k_i8_h.float().transpose(1, 2).contiguous()
    value_transposed = value_h.transpose(1, 2).contiguous()
    k_scale_by_col = k_scale_h.repeat_interleave(K_BLOCK, dim=1)[:, :seq_len]

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for q_block, row0 in enumerate(range(0, seq_len, Q_BLOCK)):
            row1 = min(row0 + Q_BLOCK, seq_len)
            rows = row1 - row0
            raw = torch.matmul(q_i8_h[:, row0:row1].float(), k_float_transposed)
            score_scale = q_scale_h[:, q_block, None, None] * k_scale_by_col[:, None, :]
            p = torch.exp(raw * score_scale * sm_scale - lse_h[:, row0:row1, None])
            dp = torch.matmul(dout_h[:, row0:row1], value_transposed)
            ds = p * (dp - delta_h[:, row0:row1, None]) * sm_scale

            p_padded = F.pad(p, (0, p_k_blocks * P_K_BLOCK - seq_len, 0, Q_BLOCK - rows))
            p_blocks = p_padded.reshape(heads, Q_BLOCK, p_k_blocks, P_K_BLOCK).permute(0, 2, 1, 3)
            p_max = p_blocks.abs().amax(dim=(2, 3))
            p_scale = p_max * INT8_SCALE_INV + INT8_SCALE_FLOOR
            p_scale_batches.append(p_scale)
            p_column_scale = p_scale.repeat_interleave(P_K_BLOCK, dim=1)[:, :seq_len]
            inv_p_scale = (1.0 / p_scale).repeat_interleave(P_K_BLOCK, dim=1)[:, :seq_len]
            p_unclamped = torch.round(p * inv_p_scale[:, None, :])
            p_quantized = p_unclamped.clamp(-128, 127)
            p_reconstructed = p_quantized * p_column_scale[:, None, :]
            p_count += rows * seq_len
            p_zero += (p_quantized == 0).sum(dim=(1, 2), dtype=torch.int64)
            p_saturation += ((p_quantized == -128) | (p_quantized == 127)).sum(dim=(1, 2), dtype=torch.int64)
            p_clipping += ((p_unclamped < -128) | (p_unclamped > 127)).sum(dim=(1, 2), dtype=torch.int64)

            ds_scale = ds_scales[:, q_block]
            ds_column_scale = ds_scale.repeat_interleave(K_BLOCK, dim=1)[:, :seq_len]
            inv_ds_scale = (1.0 / ds_scale).repeat_interleave(K_BLOCK, dim=1)[:, :seq_len]
            ds_unclamped = torch.round(ds * inv_ds_scale[:, None, :])
            ds_quantized = ds_unclamped.clamp(-128, 127)
            ds_reconstructed = ds_quantized * ds_column_scale[:, None, :]
            ds_count += rows * seq_len
            ds_zero += (ds_quantized == 0).sum(dim=(1, 2), dtype=torch.int64)
            ds_saturation += ((ds_quantized == -128) | (ds_quantized == 127)).sum(dim=(1, 2), dtype=torch.int64)
            ds_clipping += ((ds_unclamped < -128) | (ds_unclamped > 127)).sum(dim=(1, 2), dtype=torch.int64)

            ds_padded = F.pad(ds, (0, k_blocks * K_BLOCK - seq_len, 0, Q_BLOCK - rows))
            ds_blocks = ds_padded.reshape(heads, Q_BLOCK, k_blocks, K_BLOCK).permute(0, 2, 1, 3)
            true_ds_max = ds_blocks.abs().amax(dim=(2, 3))
            dynamic_ds_scale = true_ds_max * INT8_SCALE_INV + INT8_SCALE_FLOOR
            dynamic_column_scale = dynamic_ds_scale.repeat_interleave(K_BLOCK, dim=1)[:, :seq_len]
            dynamic_inv_scale = (1.0 / dynamic_ds_scale).repeat_interleave(K_BLOCK, dim=1)[:, :seq_len]
            dynamic_ds_quantized = torch.round(ds * dynamic_inv_scale[:, None, :]).clamp(-128, 127)
            dynamic_ds_reconstructed = dynamic_ds_quantized * dynamic_column_scale[:, None, :]

            q_block_dequant = q_dequant[:, row0:row1]
            dout_block = dout_h[:, row0:row1]
            do_dequant_block = do_dequant[:, row0:row1]
            stage_accumulators["qk_fp32_dq"][:, row0:row1] += torch.matmul(ds, k_dequant)
            stage_accumulators["qk_fp32_dk"] += torch.matmul(ds.transpose(1, 2), q_block_dequant)
            stage_accumulators["qk_fp32_dv"] += torch.matmul(p.transpose(1, 2), dout_block)
            stage_accumulators["dynamic_ds_fp32_dq"][:, row0:row1] += torch.matmul(dynamic_ds_reconstructed, k_dequant)
            stage_accumulators["dynamic_ds_fp32_dk"] += torch.matmul(
                dynamic_ds_reconstructed.transpose(1, 2), q_block_dequant
            )
            stage_accumulators["predictor_ds_fp32_dq"][:, row0:row1] += torch.matmul(ds_reconstructed, k_dequant)
            stage_accumulators["predictor_ds_fp32_dk"] += torch.matmul(
                ds_reconstructed.transpose(1, 2), q_block_dequant
            )
            stage_accumulators["p_quant_fp32_dv"] += torch.matmul(p_reconstructed.transpose(1, 2), dout_block)
            stage_accumulators["p_do_quant_fp32_dv"] += torch.matmul(p_reconstructed.transpose(1, 2), do_dequant_block)

            ds_ratio_batches.append(true_ds_max / predicted_ds_max[:, q_block].clamp_min(INT8_SCALE_FLOOR))
            telemetry_finite &= (
                torch.isfinite(p).all(dim=(1, 2))
                & torch.isfinite(ds).all(dim=(1, 2))
                & torch.isfinite(p_scale).all(dim=1)
                & torch.isfinite(ds_scale).all(dim=1)
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    scale_tensors = {
        "q_scale": q_scale_h.detach().float().cpu(),
        "k_scale": k_scale_h.detach().float().cpu(),
        "do_scale": do_scale[0].detach().float().cpu(),
        "p_scale": torch.cat(p_scale_batches, dim=1).detach().float().cpu(),
        "ds_scale": ds_scales.reshape(heads, -1).detach().float().cpu(),
    }
    ds_ratios = torch.cat(ds_ratio_batches, dim=1).detach().float().cpu()
    counts = {
        "q_zero": q_zero.cpu(),
        "k_zero": k_zero.cpu(),
        "do_zero": do_zero.cpu(),
        "p_count": p_count.cpu(),
        "p_zero": p_zero.cpu(),
        "p_saturation": p_saturation.cpu(),
        "p_clipping": p_clipping.cpu(),
        "ds_count": ds_count.cpu(),
        "ds_zero": ds_zero.cpu(),
        "ds_saturation": ds_saturation.cpu(),
        "ds_clipping": ds_clipping.cpu(),
        "telemetry_finite": telemetry_finite.cpu(),
    }

    reference_h = tuple(grad[0].permute(1, 0, 2).float() for grad in reference_grads)
    actual_h = tuple(grad[0].permute(1, 0, 2).float() for grad in actual_grads)
    reference_by_grad = dict(zip(("dq", "dk", "dv"), reference_h, strict=True))
    decomposition_by_head: dict[int, dict[str, Any]] = {}
    for head in (-1, *range(heads)):

        def select(tensor: torch.Tensor, selected_head: int = head) -> torch.Tensor:
            return tensor if selected_head < 0 else tensor[selected_head : selected_head + 1]

        stage_metrics: dict[str, tuple[float, float, float]] = {}
        for stage_name, stage_tensor in stage_accumulators.items():
            grad_name = stage_name.rsplit("_", 1)[-1]
            stage_metrics[stage_name] = metric(select(stage_tensor), select(reference_by_grad[grad_name]))

        transition_pairs = {
            "dynamic_vs_qk_dq": (stage_accumulators["dynamic_ds_fp32_dq"], stage_accumulators["qk_fp32_dq"]),
            "dynamic_vs_qk_dk": (stage_accumulators["dynamic_ds_fp32_dk"], stage_accumulators["qk_fp32_dk"]),
            "predictor_vs_dynamic_dq": (
                stage_accumulators["predictor_ds_fp32_dq"],
                stage_accumulators["dynamic_ds_fp32_dq"],
            ),
            "predictor_vs_dynamic_dk": (
                stage_accumulators["predictor_ds_fp32_dk"],
                stage_accumulators["dynamic_ds_fp32_dk"],
            ),
            "actual_vs_dynamic_dq": (actual_h[0], stage_accumulators["dynamic_ds_fp32_dq"]),
            "actual_vs_dynamic_dk": (actual_h[1], stage_accumulators["dynamic_ds_fp32_dk"]),
            "actual_vs_predictor_dq": (actual_h[0], stage_accumulators["predictor_ds_fp32_dq"]),
            "actual_vs_predictor_dk": (actual_h[1], stage_accumulators["predictor_ds_fp32_dk"]),
            "p_quant_vs_qk_dv": (stage_accumulators["p_quant_fp32_dv"], stage_accumulators["qk_fp32_dv"]),
            "p_do_vs_p_quant_dv": (
                stage_accumulators["p_do_quant_fp32_dv"],
                stage_accumulators["p_quant_fp32_dv"],
            ),
            "actual_vs_p_do_dv": (actual_h[2], stage_accumulators["p_do_quant_fp32_dv"]),
        }
        transition_metrics = {
            name: metric(select(actual), select(expected)) for name, (actual, expected) in transition_pairs.items()
        }
        decomposition: dict[str, Any] = {}
        for name, values in {**stage_metrics, **transition_metrics}.items():
            decomposition[f"{name}_cos"] = values[0]
            decomposition[f"{name}_rel"] = values[1]
            decomposition[f"{name}_max_abs"] = values[2]
        decomposition["qk_fp32_strict_pass"] = not strict_gate_failures(
            stage_metrics["qk_fp32_dq"],
            stage_metrics["qk_fp32_dk"],
            stage_metrics["qk_fp32_dv"],
            True,
        )
        decomposition["dynamic_ds_fp32_strict_pass"] = not strict_gate_failures(
            stage_metrics["dynamic_ds_fp32_dq"],
            stage_metrics["dynamic_ds_fp32_dk"],
            stage_metrics["qk_fp32_dv"],
            True,
        )
        decomposition["quantized_fp32_strict_pass"] = not strict_gate_failures(
            stage_metrics["predictor_ds_fp32_dq"],
            stage_metrics["predictor_ds_fp32_dk"],
            stage_metrics["p_do_quant_fp32_dv"],
            True,
        )
        decomposition_by_head[head] = decomposition

    telemetry: dict[int, dict[str, Any]] = {}
    histogram_rows: list[dict[str, Any]] = []
    for head in (-1, *range(heads)):
        if head < 0:
            p_denominator = int(counts["p_count"].sum())
            ds_denominator = int(counts["ds_count"].sum())
            values = {
                "q_int8_zero_rate": int(counts["q_zero"].sum()) / (heads * q_count),
                "k_int8_zero_rate": int(counts["k_zero"].sum()) / (heads * k_count),
                "do_int8_zero_rate": int(counts["do_zero"].sum()) / (heads * do_count),
                "p_int8_count": p_denominator,
                "p_int8_zero_count": int(counts["p_zero"].sum()),
                "p_int8_saturation_count": int(counts["p_saturation"].sum()),
                "p_int8_clipping_count": int(counts["p_clipping"].sum()),
                "ds_int8_count": ds_denominator,
                "ds_int8_zero_count": int(counts["ds_zero"].sum()),
                "ds_int8_saturation_count": int(counts["ds_saturation"].sum()),
                "ds_int8_clipping_count": int(counts["ds_clipping"].sum()),
                "telemetry_finite": bool(counts["telemetry_finite"].all()),
            }
            head_scales = {name: tensor.flatten() for name, tensor in scale_tensors.items()}
            head_ratios = ds_ratios.flatten()
        else:
            p_denominator = int(counts["p_count"][head])
            ds_denominator = int(counts["ds_count"][head])
            values = {
                "q_int8_zero_rate": int(counts["q_zero"][head]) / q_count,
                "k_int8_zero_rate": int(counts["k_zero"][head]) / k_count,
                "do_int8_zero_rate": int(counts["do_zero"][head]) / do_count,
                "p_int8_count": p_denominator,
                "p_int8_zero_count": int(counts["p_zero"][head]),
                "p_int8_saturation_count": int(counts["p_saturation"][head]),
                "p_int8_clipping_count": int(counts["p_clipping"][head]),
                "ds_int8_count": ds_denominator,
                "ds_int8_zero_count": int(counts["ds_zero"][head]),
                "ds_int8_saturation_count": int(counts["ds_saturation"][head]),
                "ds_int8_clipping_count": int(counts["ds_clipping"][head]),
                "telemetry_finite": bool(counts["telemetry_finite"][head]),
            }
            head_scales = {name: tensor[head].flatten() for name, tensor in scale_tensors.items()}
            head_ratios = ds_ratios[head].flatten()

        values.update(
            {
                "p_int8_zero_rate": values["p_int8_zero_count"] / max(p_denominator, 1),
                "p_int8_saturation_rate": values["p_int8_saturation_count"] / max(p_denominator, 1),
                "p_int8_clipping_rate": values["p_int8_clipping_count"] / max(p_denominator, 1),
                "ds_int8_zero_rate": values["ds_int8_zero_count"] / max(ds_denominator, 1),
                "ds_int8_saturation_rate": values["ds_int8_saturation_count"] / max(ds_denominator, 1),
                "ds_int8_clipping_rate": values["ds_int8_clipping_count"] / max(ds_denominator, 1),
                "ds_clip_free": values["ds_int8_clipping_count"] == 0,
            }
        )
        values.update(quantile_summary(head_scales["p_scale"], "p_scale"))
        values.update(quantile_summary(head_scales["ds_scale"], "ds_scale"))
        values.update(quantile_summary(head_ratios, "ds_true_over_predicted"))
        values.update(decomposition_by_head[head])
        telemetry[head] = values
        for scale_kind, scale_values in head_scales.items():
            histogram_rows.extend(scale_histogram_rows(scale_values, metadata, head, scale_kind))

    del q_i8, k_i8, q_scale, k_scale, delta, do_i8, do_scale, ds_q_factors, ds_k_factors, kernel_lse
    del q_i8_h, k_i8_h, p_scale_batches, ds_ratio_batches, stage_accumulators
    return telemetry, histogram_rows


def evaluate_record(
    record: dict[str, Any],
    capture: Path,
    target_seq_len: int,
    max_heads: int | None,
    smooth_k: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_seq_len = int(record["seq_len"])
    available_heads = int(record["heads"])
    heads = available_heads if max_heads is None else min(max_heads, available_heads)

    q_cpu, shape_kind = sequence_variant(record["q"][:, :, :heads], target_seq_len)
    k_cpu, _ = sequence_variant(record["k"][:, :, :heads], target_seq_len)
    v_cpu, _ = sequence_variant(record["v"][:, :, :heads], target_seq_len)
    dout_cpu, _ = sequence_variant(record["dout"][:, :, :heads], target_seq_len)
    q = q_cpu.cuda(non_blocking=True)
    k = k_cpu.cuda(non_blocking=True)
    v = v_cpu.cuda(non_blocking=True)
    dout = dout_cpu.cuda(non_blocking=True)
    del q_cpu, k_cpu, v_cpu, dout_cpu

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    output, lse, _ = flash_attn_func(
        q_ref,
        k_ref,
        v_ref,
        dropout_p=0.0,
        softmax_scale=q.size(-1) ** -0.5,
        causal=False,
        return_attn_probs=True,
    )
    output.backward(dout)
    if q_ref.grad is None or k_ref.grad is None or v_ref.grad is None:
        raise RuntimeError("FlashAttention did not produce all reference gradients.")

    with torch.inference_mode():
        dq, dk, dv = sageattn_cutlass_bwd(q, k, v, output.detach(), dout, lse.detach(), "NHD", smooth_k=smooth_k)
        histogram_metadata = {
            "capture": capture.name,
            "record_index": int(record["index"]),
            "source_seq_len": source_seq_len,
            "seq_len": target_seq_len,
            "shape_kind": shape_kind,
            "heads": heads,
            "sigma_index": int(record.get("sigma_index", -1)),
            "sigma": float(record.get("sigma", float("nan"))),
            "smooth_k": smooth_k,
        }
        telemetry, histogram_rows = quantization_telemetry(
            q,
            k,
            v,
            output.detach(),
            dout,
            lse.detach(),
            (q_ref.grad, k_ref.grad, v_ref.grad),
            (dq, dk, dv),
            histogram_metadata,
            smooth_k,
        )

    rows: list[dict[str, Any]] = []
    for head in (-1, *range(heads)):
        dq_metric = metric(head_tensor(dq, head), head_tensor(q_ref.grad, head))
        dk_metric = metric(head_tensor(dk, head), head_tensor(k_ref.grad, head))
        dv_metric = metric(head_tensor(dv, head), head_tensor(v_ref.grad, head))
        gradient_finite = all(
            torch.isfinite(head_tensor(tensor, head)).all().item()
            for tensor in (dq, dk, dv, q_ref.grad, k_ref.grad, v_ref.grad)
        )
        finite = gradient_finite and bool(telemetry[head]["telemetry_finite"])
        failures = strict_gate_failures(dq_metric, dk_metric, dv_metric, finite)
        rows.append(
            {
                **histogram_metadata,
                "head": head,
                "dq_cos": dq_metric[0],
                "dq_rel": dq_metric[1],
                "dq_max_abs": dq_metric[2],
                "dk_cos": dk_metric[0],
                "dk_rel": dk_metric[1],
                "dk_max_abs": dk_metric[2],
                "dv_cos": dv_metric[0],
                "dv_rel": dv_metric[1],
                "dv_max_abs": dv_metric[2],
                "gradient_finite": gradient_finite,
                **telemetry[head],
                "finite": finite,
                "strict_cos_min": STRICT_COS_MIN,
                "strict_rel_max": STRICT_REL_MAX,
                "strict_max_abs": STRICT_MAX_ABS,
                "strict_gate_pass": not failures,
                "strict_gate_failures": ";".join(failures),
            }
        )

    del q, k, v, dout, q_ref, k_ref, v_ref, output, lse, dq, dk, dv, telemetry
    torch.cuda.empty_cache()
    return rows, histogram_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate active CUTLASS backward accuracy and quantization telemetry on retained captures."
    )
    parser.add_argument("captures", nargs="*", type=Path, default=list(DEFAULT_CAPTURES))
    parser.add_argument("--max-heads", type=int, default=0, help="Maximum heads; zero evaluates every captured head.")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=list(DEFAULT_SEQUENCE_LENGTHS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--histogram-output", type=Path, default=DEFAULT_HISTOGRAM_OUTPUT)
    parser.add_argument("--smooth-k", nargs="+", choices=["false", "true"], default=["false", "true"])
    args = parser.parse_args()

    if args.max_heads < 0:
        raise ValueError("max-heads must be nonnegative")
    if args.max_records is not None and args.max_records <= 0:
        raise ValueError("max-records must be positive")
    if not args.sequence_lengths or any(seq_len <= 0 for seq_len in args.sequence_lengths):
        raise ValueError("sequence-lengths must contain positive values")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    max_heads = args.max_heads or None
    smooth_k_values = tuple(value == "true" for value in args.smooth_k)
    rows: list[dict[str, Any]] = []
    histogram_rows: list[dict[str, Any]] = []
    for capture in args.captures:
        payload: Any = torch.load(capture, map_location="cpu", weights_only=False)
        records = payload["records"] if isinstance(payload, dict) else payload
        if args.max_records is not None:
            records = records[: args.max_records]
        for record in records:
            for seq_len in args.sequence_lengths:
                for smooth_k in smooth_k_values:
                    print(
                        f"running capture={capture.name} record={record['index']} "
                        f"source_seq={record['seq_len']} seq={seq_len} smooth_k={smooth_k} "
                        f"heads={record['heads'] if max_heads is None else min(max_heads, record['heads'])}"
                    )
                    case_rows, case_histograms = evaluate_record(record, capture, seq_len, max_heads, smooth_k)
                    rows.extend(case_rows)
                    histogram_rows.extend(case_histograms)
                    aggregate = case_rows[0]
                    print(
                        f"  dQ={aggregate['dq_cos']:.6f}/{aggregate['dq_rel']:.5f} "
                        f"dK={aggregate['dk_cos']:.6f}/{aggregate['dk_rel']:.5f} "
                        f"dV={aggregate['dv_cos']:.6f}/{aggregate['dv_rel']:.5f} "
                        f"dS_clip_ppm={1e6 * aggregate['ds_int8_clipping_rate']:.3f} "
                        f"strict={aggregate['strict_gate_pass']} finite={aggregate['finite']}"
                    )

    write_csv(args.output, rows)
    write_csv(args.histogram_output, histogram_rows)
    aggregate_rows = [row for row in rows if row["head"] == -1]
    strict_passes = sum(bool(row["strict_gate_pass"]) for row in rows)
    print(
        f"strict rows={strict_passes}/{len(rows)} aggregate={sum(bool(row['strict_gate_pass']) for row in aggregate_rows)}/"
        f"{len(aggregate_rows)}"
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.histogram_output}")


if __name__ == "__main__":
    main()

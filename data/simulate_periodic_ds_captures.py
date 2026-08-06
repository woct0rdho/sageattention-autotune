import math
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func

from sageattention.triton.quant_per_block import per_block_int8

INTERVALS = (16, 32, 64, 128)
GUARDS = (1.5, 1.75, 1.875, 2.0, 2.125, 2.25, 2.5, 3.0)
MAX_HEADS = 2
INT8_SCALE_INV = float.fromhex("0x1.010122p-7")
INT8_SCALE_FLOOR = 2.0**-126


def metric(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = actual - expected
    cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    relative = (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected).clamp_min(1e-6)).item()
    maximum = diff.abs().max().item()
    return cosine, relative, maximum


def ratio_quantiles(values: list[torch.Tensor]) -> dict[str, float]:
    if not values:
        return {name: 1.0 for name in ("p50", "p90", "p95", "p99", "p999", "max")}
    merged = torch.cat(values).float()
    result = {
        "p50": torch.quantile(merged, 0.50).item(),
        "p90": torch.quantile(merged, 0.90).item(),
        "p95": torch.quantile(merged, 0.95).item(),
        "p99": torch.quantile(merged, 0.99).item(),
        "p999": torch.quantile(merged, 0.999).item(),
        "max": merged.max().item(),
    }
    return result


def run_case(
    record: dict[str, Any], capture_name: str, dout_seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seq_len = int(record["seq_len"])
    head_count = min(MAX_HEADS, int(record["heads"]))
    q = record["q"][:, :, :head_count].cuda(non_blocking=True)
    k = record["k"][:, :, :head_count].cuda(non_blocking=True)
    v = record["v"][:, :, :head_count].cuda(non_blocking=True)

    if "dout" in record:
        dout = record["dout"][:, :, :head_count].cuda(non_blocking=True)
    else:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(100000 * int(record["index"]) + dout_seed)
        dout = torch.randn(q.shape, device="cuda", dtype=torch.float16, generator=generator)

    output, lse, _ = flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=64**-0.5,
        causal=False,
        return_attn_probs=True,
    )
    q_i8, q_scale, k_i8, k_scale = per_block_int8(
        q,
        k,
        km=None,
        BLKQ=32,
        BLKK=64,
        tensor_layout="NHD",
    )
    q_i8 = q_i8[0].permute(1, 0, 2).contiguous()
    k_i8 = k_i8[0].permute(1, 0, 2).contiguous()
    q_scale = q_scale[0]
    k_scale = k_scale[0]
    v_h = v[0].permute(1, 0, 2).float()
    do_h = dout[0].permute(1, 0, 2).float()
    out_h = output[0].permute(1, 0, 2).float()
    lse_h = lse[0].float()
    q_dequant = q_i8.float() * q_scale.repeat_interleave(32, dim=-1)[:, :seq_len, None]
    k_dequant = k_i8.float() * k_scale.repeat_interleave(64, dim=-1)[:, :seq_len, None]
    delta = (out_h * do_h).sum(dim=-1)
    sm_scale = 64**-0.5
    k_blocks = math.ceil(seq_len / 64)

    dq_exact = torch.zeros((head_count, seq_len, 64), device="cuda")
    dk_exact = torch.zeros_like(dq_exact)
    dq_dynamic = torch.zeros_like(dq_exact)
    dk_dynamic = torch.zeros_like(dq_exact)

    candidates = tuple((interval, guard) for interval in INTERVALS for guard in GUARDS)
    accumulators = {candidate: (torch.zeros_like(dq_exact), torch.zeros_like(dq_exact)) for candidate in candidates}
    calibration_scales: dict[tuple[int, float], torch.Tensor | None] = {candidate: None for candidate in candidates}
    stats = {
        candidate: {
            "count": 0,
            "zero": 0,
            "sat": 0,
            "ds_error_sq": 0.0,
            "ds_energy": 0.0,
            "sat_energy": 0.0,
        }
        for candidate in candidates
    }
    dynamic_stats = {"count": 0, "zero": 0}
    scale_ratios: dict[int, list[torch.Tensor]] = defaultdict(list)
    interval_calibration: dict[int, torch.Tensor | None] = {interval: None for interval in INTERVALS}

    for q_block, row0 in enumerate(range(0, seq_len, 32)):
        row1 = min(row0 + 32, seq_len)
        rows = row1 - row0
        raw = torch.matmul(q_i8[:, row0:row1, :].float(), k_i8.float().transpose(1, 2))
        score_scale = q_scale[:, q_block, None, None] * k_scale.repeat_interleave(64, dim=-1)[:, None, :]
        p = torch.exp(raw * score_scale * sm_scale - lse_h[:, row0:row1, None])
        dp = torch.matmul(do_h[:, row0:row1, :], v_h.transpose(1, 2))
        ds = p * (dp - delta[:, row0:row1, None]) * sm_scale

        padded = torch.zeros((head_count, 32, k_blocks * 64), device="cuda")
        padded[:, :rows, :seq_len] = ds
        ds_blocks = padded.reshape(head_count, 32, k_blocks, 64).permute(0, 2, 1, 3).contiguous()
        max_abs = ds_blocks.abs().amax(dim=(2, 3), keepdim=True)
        dynamic_scale = max_abs * INT8_SCALE_INV + INT8_SCALE_FLOOR
        dynamic_int = torch.round(ds_blocks / dynamic_scale).clamp(-128, 127)
        dynamic_rec_blocks = dynamic_int * dynamic_scale
        dynamic_rec = dynamic_rec_blocks.permute(0, 2, 1, 3).reshape(head_count, 32, k_blocks * 64)
        dynamic_rec = dynamic_rec[:, :rows, :seq_len]

        q_block_dequant = q_dequant[:, row0:row1, :]
        dq_exact[:, row0:row1, :] += torch.matmul(ds, k_dequant)
        dk_exact += torch.matmul(ds.transpose(1, 2), q_block_dequant)
        dq_dynamic[:, row0:row1, :] += torch.matmul(dynamic_rec, k_dequant)
        dk_dynamic += torch.matmul(dynamic_rec.transpose(1, 2), q_block_dequant)
        dynamic_stats["count"] += head_count * rows * seq_len
        dynamic_stats["zero"] += (dynamic_int[:, :, :rows, :] == 0).sum().item()

        for interval in INTERVALS:
            if q_block % interval == 0:
                interval_calibration[interval] = dynamic_scale
            else:
                calibration = interval_calibration[interval]
                assert calibration is not None
                scale_ratios[interval].append((dynamic_scale / calibration).flatten().cpu())

        valid_ds_blocks = ds_blocks[:, :, :rows, :]
        for candidate in candidates:
            interval, guard = candidate
            if q_block % interval == 0:
                calibration_scales[candidate] = dynamic_scale
                scale = dynamic_scale
            else:
                calibration = calibration_scales[candidate]
                assert calibration is not None
                scale = calibration * guard

            unclamped = torch.round(ds_blocks / scale)
            overflow = (unclamped > 127) | (unclamped < -128)
            quantized = unclamped.clamp(-128, 127)
            reconstructed_blocks = quantized * scale
            reconstructed = reconstructed_blocks.permute(0, 2, 1, 3).reshape(head_count, 32, k_blocks * 64)
            reconstructed = reconstructed[:, :rows, :seq_len]
            dq_candidate, dk_candidate = accumulators[candidate]
            dq_candidate[:, row0:row1, :] += torch.matmul(reconstructed, k_dequant)
            dk_candidate += torch.matmul(reconstructed.transpose(1, 2), q_block_dequant)

            valid_quantized = quantized[:, :, :rows, :]
            valid_overflow = overflow[:, :, :rows, :]
            valid_reconstructed = reconstructed_blocks[:, :, :rows, :]
            values = stats[candidate]
            values["count"] += head_count * rows * seq_len
            values["zero"] += (valid_quantized == 0).sum().item()
            values["sat"] += valid_overflow.sum().item()
            values["ds_error_sq"] += ((valid_reconstructed - valid_ds_blocks) ** 2).sum().item()
            values["ds_energy"] += (valid_ds_blocks**2).sum().item()
            values["sat_energy"] += (valid_ds_blocks.square() * valid_overflow).sum().item()

    dynamic_dq_metric = metric(dq_dynamic, dq_exact)
    dynamic_dk_metric = metric(dk_dynamic, dk_exact)
    rows_out: list[dict[str, Any]] = []
    for candidate in candidates:
        interval, guard = candidate
        dq_candidate, dk_candidate = accumulators[candidate]
        dq_exact_metric = metric(dq_candidate, dq_exact)
        dk_exact_metric = metric(dk_candidate, dk_exact)
        dq_dynamic_metric = metric(dq_candidate, dq_dynamic)
        dk_dynamic_metric = metric(dk_candidate, dk_dynamic)
        values = stats[candidate]
        rows_out.append(
            {
                "capture": capture_name,
                "record_index": int(record["index"]),
                "seq_len": seq_len,
                "heads": head_count,
                "dout_seed": dout_seed,
                "interval": interval,
                "guard": guard,
                "dynamic_dq_cos": dynamic_dq_metric[0],
                "dynamic_dq_rel": dynamic_dq_metric[1],
                "dynamic_dk_cos": dynamic_dk_metric[0],
                "dynamic_dk_rel": dynamic_dk_metric[1],
                "dq_cos_vs_exact": dq_exact_metric[0],
                "dq_rel_vs_exact": dq_exact_metric[1],
                "dq_max_vs_exact": dq_exact_metric[2],
                "dk_cos_vs_exact": dk_exact_metric[0],
                "dk_rel_vs_exact": dk_exact_metric[1],
                "dk_max_vs_exact": dk_exact_metric[2],
                "dq_cos_vs_dynamic": dq_dynamic_metric[0],
                "dq_rel_vs_dynamic": dq_dynamic_metric[1],
                "dk_cos_vs_dynamic": dk_dynamic_metric[0],
                "dk_rel_vs_dynamic": dk_dynamic_metric[1],
                "zero_rate": values["zero"] / values["count"],
                "sat_rate": values["sat"] / values["count"],
                "ds_rel_rmse": math.sqrt(values["ds_error_sq"] / max(values["ds_energy"], 1.0e-30)),
                "sat_energy_rate": values["sat_energy"] / max(values["ds_energy"], 1.0e-30),
            }
        )

    ratio_rows: list[dict[str, Any]] = []
    for interval in INTERVALS:
        quantiles = ratio_quantiles(scale_ratios[interval])
        ratio_rows.append(
            {
                "capture": capture_name,
                "record_index": int(record["index"]),
                "seq_len": seq_len,
                "heads": head_count,
                "dout_seed": dout_seed,
                "interval": interval,
                **quantiles,
            }
        )

    del q, k, v, dout, output, lse, q_i8, k_i8, q_scale, k_scale
    del dq_exact, dk_exact, dq_dynamic, dk_dynamic, accumulators
    torch.cuda.empty_cache()
    return rows_out, ratio_rows

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func

from sageattention.triton.quant_per_block import per_block_int8

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_CAPTURES = (
    DATA_DIR / "sdxl_periodic_ds_inputs_training_dout_latent128.pt",
    DATA_DIR / "sdxl_periodic_ds_inputs_training_dout_latent128_seed2.pt",
)
PREDICTOR_GUARDS = {
    "forward_saved_pmax": (0.25, 0.50, 0.75, 1.00),
    "forward_row_pmax": (0.0625, 0.125, 0.25, 0.50),
    "forward_separable": (0.50, 1.00, 2.00, 4.00),
    "backward_l2_bound": (0.03125, 0.0625, 0.125, 0.25),
    "backward_row_pmax": (0.03125, 0.0625, 0.125, 0.25),
    "backward_separable": (0.25, 0.50, 1.00, 2.00),
    "backward_dout_max": (0.25, 0.50, 1.00, 2.00),
    "backward_existing_max": (0.25, 0.50, 1.00, 2.00),
    "backward_rms": (0.25, 0.50, 0.75, 1.00),
}
PREDICTORS = tuple(PREDICTOR_GUARDS)
SM_SCALE = 64**-0.5
INT8_SCALE_INV = float.fromhex("0x1.010122p-7")
INT8_SCALE_FLOOR = 2.0**-126


def metric(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = actual - expected
    cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    relative = (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected).clamp_min(1e-6)).item()
    return cosine, relative


def quantiles(values: list[torch.Tensor]) -> dict[str, float]:
    if not values:
        return {name: 1.0 for name in ("p01", "p10", "p50", "p90", "p99", "p999", "max")}
    merged = torch.cat(values).float()
    result = {
        "p01": torch.quantile(merged, 0.01).item(),
        "p10": torch.quantile(merged, 0.10).item(),
        "p50": torch.quantile(merged, 0.50).item(),
        "p90": torch.quantile(merged, 0.90).item(),
        "p99": torch.quantile(merged, 0.99).item(),
        "p999": torch.quantile(merged, 0.999).item(),
        "max": merged.max().item(),
    }
    return result


def block_pad(tensor: torch.Tensor, blocks: int, block_size: int) -> torch.Tensor:
    padded = torch.zeros(
        (*tensor.shape[:-2], blocks * block_size, tensor.shape[-1]),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    padded[..., : tensor.shape[-2], :] = tensor
    return padded


def run_record(
    record: dict[str, Any], capture_name: str, max_heads: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seq_len = int(record["seq_len"])
    heads = min(max_heads, int(record["heads"]))
    q = record["q"][:, :, :heads].cuda(non_blocking=True)
    k = record["k"][:, :, :heads].cuda(non_blocking=True)
    v = record["v"][:, :, :heads].cuda(non_blocking=True)
    if "dout" in record:
        dout = record["dout"][:, :, :heads].cuda(non_blocking=True)
    else:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(100000 * int(record["index"]))
        dout = torch.randn(q.shape, device="cuda", dtype=torch.float16, generator=generator)

    output, lse, _ = flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=SM_SCALE,
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

    k_blocks = (seq_len + 63) // 64
    do_global_rms = do_h.square().mean().sqrt()
    v_padded = block_pad(v_h, k_blocks, 64).reshape(heads, k_blocks, 64, 64)
    v_l2_max = v_padded.square().sum(dim=-1).sqrt().amax(dim=-1)
    v_abs_max = v_padded.abs().amax(dim=(2, 3))
    v_rms = v_padded.square().mean(dim=(2, 3)).sqrt()
    valid_cols = (torch.arange(k_blocks * 64, device="cuda") < seq_len).reshape(1, k_blocks, 1, 64)
    valid_ds = torch.arange(32, device="cuda").reshape(1, 1, 32, 1)

    dq_exact = torch.zeros((heads, seq_len, 64), device="cuda")
    dk_exact = torch.zeros_like(dq_exact)
    dq_dynamic = torch.zeros_like(dq_exact)
    dk_dynamic = torch.zeros_like(dq_exact)

    scale_accumulators = {
        (predictor, guard): (torch.zeros_like(dq_exact), torch.zeros_like(dq_exact))
        for predictor in PREDICTORS
        for guard in PREDICTOR_GUARDS[predictor]
    }
    scale_counts = {
        (predictor, guard): {"count": 0, "sat": 0, "zero": 0}
        for predictor in PREDICTORS
        for guard in PREDICTOR_GUARDS[predictor]
    }
    scale_ratios: dict[str, list[torch.Tensor]] = defaultdict(list)

    random_dout_scale_ratios: list[torch.Tensor] = []

    for q_block, row0 in enumerate(range(0, seq_len, 32)):
        row1 = min(row0 + 32, seq_len)
        rows = row1 - row0
        q_block_dequant = q_dequant[:, row0:row1, :]
        do_tile = do_h[:, row0:row1, :]
        delta_tile = (out_h[:, row0:row1, :] * do_tile).sum(dim=-1)
        do_l2_max = do_tile.square().sum(dim=-1).sqrt().amax(dim=-1)
        do_abs_max = do_tile.abs().amax(dim=(1, 2))
        do_rms = do_tile.square().mean(dim=(1, 2)).sqrt()
        delta_max = delta_tile.abs().amax(dim=-1)
        delta_rms = delta_tile.square().mean(dim=1).sqrt()
        out_l2_max = out_h[:, row0:row1, :].square().sum(dim=-1).sqrt().amax(dim=-1)

        raw = torch.matmul(q_i8[:, row0:row1, :].float(), k_i8.float().transpose(1, 2))
        score_scale = q_scale[:, q_block, None, None] * k_scale.repeat_interleave(64, dim=-1)[:, None, :]
        p = torch.exp(raw * score_scale * SM_SCALE - lse_h[:, row0:row1, None])
        dp = torch.matmul(do_tile, v_h.transpose(1, 2))
        ds = p * (dp - delta_tile[:, :, None]) * SM_SCALE

        padded = torch.zeros((heads, 32, k_blocks * 64), device="cuda")
        padded[:, :rows, :seq_len] = ds
        ds_blocks = padded.reshape(heads, 32, k_blocks, 64).permute(0, 2, 1, 3).contiguous()
        true_max = ds_blocks.abs().amax(dim=(2, 3))
        p_padded = torch.zeros((heads, 32, k_blocks * 64), device="cuda")
        p_padded[:, :rows, :seq_len] = p
        p_blocks = p_padded.reshape(heads, 32, k_blocks, 64).permute(0, 2, 1, 3).contiguous()
        p_max = p_blocks.amax(dim=(2, 3))
        row_p_max = p.amax(dim=(1, 2))
        forward_base = SM_SCALE * do_global_rms * (v_l2_max + out_l2_max[:, None])
        backward_base = SM_SCALE * (do_l2_max[:, None] * v_l2_max + delta_max[:, None])

        predictors = {
            "forward_saved_pmax": p_max * forward_base,
            "forward_row_pmax": row_p_max[:, None] * forward_base,
            "forward_separable": forward_base / seq_len,
            "backward_l2_bound": p_max * backward_base,
            "backward_row_pmax": row_p_max[:, None] * backward_base,
            "backward_separable": backward_base / seq_len,
            "backward_dout_max": SM_SCALE * (3.0 * do_abs_max[:, None] * v_l2_max + delta_max[:, None]) / seq_len,
            "backward_existing_max": SM_SCALE * (8.0 * do_abs_max[:, None] * v_abs_max + delta_max[:, None]) / seq_len,
            "backward_rms": SM_SCALE * p_max * (do_rms[:, None] * v_rms + delta_rms[:, None]),
        }
        for predictor, predicted_max in predictors.items():
            scale_ratios[predictor].append(
                (predicted_max / true_max.clamp_min(1e-12)).masked_select(true_max > 1e-12).detach()
            )

        random_dout = torch.randn_like(do_tile)
        random_dout = random_dout * (do_global_rms / random_dout.square().mean().sqrt().clamp_min(1e-12))
        random_delta = (out_h[:, row0:row1, :] * random_dout).sum(dim=-1)
        random_dp = torch.matmul(random_dout, v_h.transpose(1, 2))
        random_ds = p * (random_dp - random_delta[:, :, None]) * SM_SCALE
        random_padded = torch.zeros((heads, 32, k_blocks * 64), device="cuda")
        random_padded[:, :rows, :seq_len] = random_ds
        random_blocks = random_padded.reshape(heads, 32, k_blocks, 64).permute(0, 2, 1, 3)
        random_max = random_blocks.abs().amax(dim=(2, 3))
        random_dout_scale_ratios.append(
            (random_max / true_max.clamp_min(1e-12)).masked_select(true_max > 1e-12).detach()
        )

        dq_exact[:, row0:row1, :] += torch.matmul(ds, k_dequant)
        dk_exact += torch.matmul(ds.transpose(1, 2), q_block_dequant)
        dynamic_scale = true_max[:, :, None, None] * INT8_SCALE_INV + INT8_SCALE_FLOOR
        dynamic_int = torch.round(ds_blocks / dynamic_scale).clamp(-128, 127)
        dynamic_rec = dynamic_int * dynamic_scale
        dynamic_rec = dynamic_rec.permute(0, 2, 1, 3).reshape(heads, 32, k_blocks * 64)
        dynamic_rec = dynamic_rec[:, :rows, :seq_len]
        dq_dynamic[:, row0:row1, :] += torch.matmul(dynamic_rec, k_dequant)
        dk_dynamic += torch.matmul(dynamic_rec.transpose(1, 2), q_block_dequant)

        valid_mask = valid_cols & (valid_ds < rows)
        for predictor, predicted_max in predictors.items():
            for guard in PREDICTOR_GUARDS[predictor]:
                scale = predicted_max[:, :, None, None] * guard * INT8_SCALE_INV + INT8_SCALE_FLOOR
                unclamped = torch.round(ds_blocks / scale)
                quantized = unclamped.clamp(-128, 127)
                reconstructed_blocks = quantized * scale
                reconstructed = reconstructed_blocks.permute(0, 2, 1, 3).reshape(heads, 32, k_blocks * 64)
                reconstructed = reconstructed[:, :rows, :seq_len]
                dq_candidate, dk_candidate = scale_accumulators[(predictor, guard)]
                dq_candidate[:, row0:row1, :] += torch.matmul(reconstructed, k_dequant)
                dk_candidate += torch.matmul(reconstructed.transpose(1, 2), q_block_dequant)
                stats = scale_counts[(predictor, guard)]
                stats["count"] += heads * valid_mask.sum().item()
                stats["sat"] += ((unclamped.abs() > 127) & valid_mask).sum().item()
                stats["zero"] += ((quantized == 0) & valid_mask).sum().item()

    scale_rows: list[dict[str, Any]] = []
    dynamic_dq = metric(dq_dynamic, dq_exact)
    dynamic_dk = metric(dk_dynamic, dk_exact)
    ratio_quantile_rows = []
    for predictor in PREDICTORS:
        ratio_quantile_rows.append(
            {
                "capture": capture_name,
                "record_index": int(record["index"]),
                "seq_len": seq_len,
                "heads": heads,
                "sigma_index": int(record.get("sigma_index", -1)),
                "sigma": float(record.get("sigma", float("nan"))),
                "predictor": predictor,
                "ratio_kind": "captured_dout",
                **quantiles(scale_ratios[predictor]),
            }
        )
        for guard in PREDICTOR_GUARDS[predictor]:
            dq_candidate, dk_candidate = scale_accumulators[(predictor, guard)]
            dq_metrics = metric(dq_candidate, dq_exact)
            dk_metrics = metric(dk_candidate, dk_exact)
            stats = scale_counts[(predictor, guard)]
            scale_rows.append(
                {
                    "capture": capture_name,
                    "record_index": int(record["index"]),
                    "seq_len": seq_len,
                    "heads": heads,
                    "sigma_index": int(record.get("sigma_index", -1)),
                    "sigma": float(record.get("sigma", float("nan"))),
                    "predictor": predictor,
                    "guard": guard,
                    "dq_cos": dq_metrics[0],
                    "dq_rel": dq_metrics[1],
                    "dk_cos": dk_metrics[0],
                    "dk_rel": dk_metrics[1],
                    "sat_rate": stats["sat"] / max(stats["count"], 1),
                    "zero_rate": stats["zero"] / max(stats["count"], 1),
                    "dynamic_dq_cos": dynamic_dq[0],
                    "dynamic_dq_rel": dynamic_dq[1],
                    "dynamic_dk_cos": dynamic_dk[0],
                    "dynamic_dk_rel": dynamic_dk[1],
                }
            )

    ratio_quantile_rows.append(
        {
            "capture": capture_name,
            "record_index": int(record["index"]),
            "seq_len": seq_len,
            "heads": heads,
            "sigma_index": int(record.get("sigma_index", -1)),
            "sigma": float(record.get("sigma", float("nan"))),
            "predictor": "random_dout_same_global_rms",
            "ratio_kind": "random_dout_over_captured_dout",
            **quantiles(random_dout_scale_ratios),
        }
    )

    del q, k, v, dout, output, lse, q_i8, k_i8, q_scale, k_scale
    del dq_exact, dk_exact, dq_dynamic, dk_dynamic, scale_accumulators
    torch.cuda.empty_cache()
    return scale_rows, ratio_quantile_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-heads", type=int, default=10)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--output-prefix", type=Path, default=DATA_DIR / "ds_scale_precompute")
    args = parser.parse_args()

    scale_rows: list[dict[str, Any]] = []
    ratio_rows: list[dict[str, Any]] = []
    seen_records = 0
    for capture_path in DEFAULT_CAPTURES:
        payload = torch.load(capture_path, map_location="cpu", weights_only=False)
        for record in payload["records"]:
            if args.limit_records and seen_records >= args.limit_records:
                break
            print(
                f"running capture={capture_path.name} record={record['index']} "
                f"seq={record['seq_len']} heads={min(args.max_heads, record['heads'])}"
            )
            case_scale, case_ratios = run_record(record, capture_path.name, args.max_heads)
            scale_rows.extend(case_scale)
            ratio_rows.extend(case_ratios)
            seen_records += 1
        if args.limit_records and seen_records >= args.limit_records:
            break

    write_csv(args.output_prefix.with_name(args.output_prefix.name + "_scales.csv"), scale_rows)
    write_csv(args.output_prefix.with_name(args.output_prefix.name + "_ratios.csv"), ratio_rows)

    print("\nBest scale predictors by mean dQ/dK relative error")
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in scale_rows:
        groups[(str(row["predictor"]), float(row["guard"]))].append(row)
    ranking = []
    for key, rows in groups.items():
        ranking.append(
            (
                sum(float(row["dq_rel"]) + float(row["dk_rel"]) for row in rows) / len(rows),
                key,
                rows,
            )
        )
    for _, (predictor, guard), rows in sorted(ranking)[:16]:
        print(
            f"predictor={predictor:22s} guard={guard:.2f} n={len(rows):2d} "
            f"dq_rel={sum(float(row['dq_rel']) for row in rows) / len(rows):.5f} "
            f"dk_rel={sum(float(row['dk_rel']) for row in rows) / len(rows):.5f} "
            f"dq_cos={sum(float(row['dq_cos']) for row in rows) / len(rows):.6f} "
            f"dk_cos={sum(float(row['dk_cos']) for row in rows) / len(rows):.6f} "
            f"sat_ppm={1e6 * max(float(row['sat_rate']) for row in rows):.2f}"
        )

    print("\nScale ratio quantiles")
    for row in ratio_rows:
        if row["ratio_kind"] == "random_dout_over_captured_dout":
            print(
                f"capture={row['capture']} sigma_index={row.get('sigma_index', -1)} "
                f"random_dout/captured_dout p01={float(row['p01']):.3f} "
                f"p50={float(row['p50']):.3f} p99={float(row['p99']):.3f} max={float(row['max']):.3f}"
            )

    print(f"wrote {args.output_prefix.with_name(args.output_prefix.name + '_scales.csv')}")
    print(f"wrote {args.output_prefix.with_name(args.output_prefix.name + '_ratios.csv')}")


if __name__ == "__main__":
    main()

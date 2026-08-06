from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, TypedDict

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func

from sageattention.cutlass_bwd import sageattn_cutlass_bwd

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_CAPTURES = (
    DATA_DIR / "sdxl_periodic_ds_inputs_training_dout_latent128.pt",
    DATA_DIR / "sdxl_periodic_ds_inputs_training_dout_latent128_seed2.pt",
)
DEFAULT_OUTPUT = DATA_DIR / "cutlass_bwd_predictor_capture_accuracy.csv"


class AccuracyRow(TypedDict):
    capture: str
    record_index: int
    seq_len: int
    heads: int
    sigma_index: int
    sigma: float
    dq_cos: float
    dq_rel: float
    dq_max_abs: float
    dk_cos: float
    dk_rel: float
    dk_max_abs: float
    dv_cos: float
    dv_rel: float
    dv_max_abs: float
    finite: bool


def metric(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    actual_float = actual.float()
    expected_float = expected.float()
    diff = actual_float - expected_float
    cosine = F.cosine_similarity(actual_float.flatten(), expected_float.flatten(), dim=0).item()
    relative = (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected_float).clamp_min(1e-6)).item()
    return cosine, relative, diff.abs().max().item()


def evaluate_record(record: dict[str, Any], capture: Path, max_heads: int) -> AccuracyRow:
    heads = min(max_heads, int(record["heads"]))
    q = record["q"][:, :, :heads].cuda(non_blocking=True)
    k = record["k"][:, :, :heads].cuda(non_blocking=True)
    v = record["v"][:, :, :heads].cuda(non_blocking=True)
    dout = record["dout"][:, :, :heads].cuda(non_blocking=True)

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
        dq, dk, dv = sageattn_cutlass_bwd(q, k, v, output.detach(), dout, lse.detach(), "NHD")

    dq_metric = metric(dq, q_ref.grad)
    dk_metric = metric(dk, k_ref.grad)
    dv_metric = metric(dv, v_ref.grad)
    finite = all(torch.isfinite(tensor).all().item() for tensor in (dq, dk, dv))
    row: AccuracyRow = {
        "capture": capture.name,
        "record_index": int(record["index"]),
        "seq_len": int(record["seq_len"]),
        "heads": heads,
        "sigma_index": int(record.get("sigma_index", -1)),
        "sigma": float(record.get("sigma", float("nan"))),
        "dq_cos": dq_metric[0],
        "dq_rel": dq_metric[1],
        "dq_max_abs": dq_metric[2],
        "dk_cos": dk_metric[0],
        "dk_rel": dk_metric[1],
        "dk_max_abs": dk_metric[2],
        "dv_cos": dv_metric[0],
        "dv_rel": dv_metric[1],
        "dv_max_abs": dv_metric[2],
        "finite": finite,
    }

    del q, k, v, dout, q_ref, k_ref, v_ref, output, lse, dq, dk, dv
    torch.cuda.empty_cache()
    return row


def write_csv(path: Path, rows: list[AccuracyRow]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the active CUTLASS backward path on retained captures.")
    parser.add_argument("captures", nargs="*", type=Path, default=list(DEFAULT_CAPTURES))
    parser.add_argument("--max-heads", type=int, default=10)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.max_heads <= 0:
        raise ValueError("max-heads must be positive")
    if args.max_records is not None and args.max_records <= 0:
        raise ValueError("max-records must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    rows: list[AccuracyRow] = []
    for capture in args.captures:
        payload: Any = torch.load(capture, map_location="cpu", weights_only=False)
        records = payload["records"] if isinstance(payload, dict) else payload
        if args.max_records is not None:
            records = records[: args.max_records]
        for record in records:
            row = evaluate_record(record, capture, args.max_heads)
            rows.append(row)
            print(
                f"{row['capture']} sigma={row['sigma_index']} heads={row['heads']} "
                f"dQ={row['dq_cos']:.6f}/{row['dq_rel']:.5f} "
                f"dK={row['dk_cos']:.6f}/{row['dk_rel']:.5f} "
                f"dV={row['dv_cos']:.6f}/{row['dv_rel']:.5f} finite={row['finite']}"
            )

    write_csv(args.output, rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

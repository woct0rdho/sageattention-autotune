import argparse
import math

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func

from sageattention.triton.quant_per_block import per_block_int8

CANDIDATES = tuple((interval, guard) for interval in (16, 32, 64, 128) for guard in (1.5, 1.75, 2.0, 2.25))
INT8_SCALE_INV = float.fromhex("0x1.010122p-7")
INT8_SCALE_FLOOR = 2.0**-126


def metric(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = actual - expected
    cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    relative = (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected).clamp_min(1e-6)).item()
    return cosine, relative, diff.abs().max().item()


def run(seq_len: int, heads: int, seed: int) -> None:
    torch.manual_seed(seed)
    shape = (1, seq_len, heads, 64)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q)
    q_ref = q.detach().requires_grad_(True)
    k_ref = k.detach().requires_grad_(True)
    v_ref = v.detach().requires_grad_(True)
    output, lse, _ = flash_attn_func(
        q_ref,
        k_ref,
        v_ref,
        dropout_p=0.0,
        softmax_scale=64**-0.5,
        causal=False,
        return_attn_probs=True,
    )
    q_i8, q_scale, k_i8, k_scale = per_block_int8(q, k, km=None, BLKQ=32, BLKK=64, tensor_layout="NHD")
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

    dq_exact = torch.zeros((heads, seq_len, 64), device="cuda")
    dk_exact = torch.zeros_like(dq_exact)
    dq_dynamic = torch.zeros_like(dq_exact)
    dk_dynamic = torch.zeros_like(dq_exact)
    accumulators = {candidate: (torch.zeros_like(dq_exact), torch.zeros_like(dq_exact)) for candidate in CANDIDATES}
    calibration_scales = {candidate: None for candidate in CANDIDATES}
    stats = {candidate: {"count": 0, "zero": 0, "sat": 0} for candidate in CANDIDATES}
    dynamic_stats = {"count": 0, "zero": 0}

    for q_block, row0 in enumerate(range(0, seq_len, 32)):
        row1 = min(row0 + 32, seq_len)
        rows = row1 - row0
        raw = torch.matmul(q_i8[:, row0:row1, :].float(), k_i8.float().transpose(1, 2))
        score_scale = q_scale[:, q_block, None, None] * k_scale.repeat_interleave(64, dim=-1)[:, None, :]
        p = torch.exp(raw * score_scale * sm_scale - lse_h[:, row0:row1, None])
        dp = torch.matmul(do_h[:, row0:row1, :], v_h.transpose(1, 2))
        ds = p * (dp - delta[:, row0:row1, None]) * sm_scale
        padded = torch.zeros((heads, 32, k_blocks * 64), device="cuda")
        padded[:, :rows, :seq_len] = ds
        ds_blocks = padded.reshape(heads, 32, k_blocks, 64).permute(0, 2, 1, 3).contiguous()
        max_abs = ds_blocks.abs().amax(dim=(2, 3), keepdim=True)
        dynamic_scale = max_abs * INT8_SCALE_INV + INT8_SCALE_FLOOR
        dynamic_int = torch.round(ds_blocks / dynamic_scale).clamp(-128, 127)
        dynamic_rec = dynamic_int.mul(dynamic_scale).permute(0, 2, 1, 3).reshape(heads, 32, k_blocks * 64)
        dynamic_rec = dynamic_rec[:, :rows, :seq_len]

        q_block_dequant = q_dequant[:, row0:row1, :]
        dq_exact[:, row0:row1, :] += torch.matmul(ds, k_dequant)
        dk_exact += torch.matmul(ds.transpose(1, 2), q_block_dequant)
        dq_dynamic[:, row0:row1, :] += torch.matmul(dynamic_rec, k_dequant)
        dk_dynamic += torch.matmul(dynamic_rec.transpose(1, 2), q_block_dequant)
        dynamic_stats["count"] += heads * rows * seq_len
        dynamic_stats["zero"] += (dynamic_int[:, :, :rows, :].reshape(heads, k_blocks, -1) == 0).sum().item()

        for candidate in CANDIDATES:
            interval, guard = candidate
            if q_block % interval == 0:
                calibration_scales[candidate] = dynamic_scale
                scale = dynamic_scale
            else:
                calibration = calibration_scales[candidate]
                assert calibration is not None
                scale = calibration * guard
            unclamped = torch.round(ds_blocks / scale)
            quantized = unclamped.clamp(-128, 127)
            reconstructed = quantized.mul(scale).permute(0, 2, 1, 3).reshape(heads, 32, k_blocks * 64)
            reconstructed = reconstructed[:, :rows, :seq_len]
            dq, dk = accumulators[candidate]
            dq[:, row0:row1, :] += torch.matmul(reconstructed, k_dequant)
            dk += torch.matmul(reconstructed.transpose(1, 2), q_block_dequant)
            valid_unclamped = unclamped[:, :, :rows, :].reshape(heads, k_blocks, -1)
            valid_quantized = quantized[:, :, :rows, :].reshape(heads, k_blocks, -1)
            stats[candidate]["count"] += heads * rows * seq_len
            stats[candidate]["zero"] += (valid_quantized == 0).sum().item()
            stats[candidate]["sat"] += (valid_unclamped.abs() > 127).sum().item()

    print(f"seq={seq_len} heads={heads} seed={seed}")
    print(
        "  dynamic",
        "dq",
        metric(dq_dynamic, dq_exact),
        "dk",
        metric(dk_dynamic, dk_exact),
        f"zero={dynamic_stats['zero'] / dynamic_stats['count']:.6f}",
    )
    for candidate in CANDIDATES:
        dq, dk = accumulators[candidate]
        values = stats[candidate]
        print(
            f"  interval={candidate[0]:2d} guard={candidate[1]:.3f}",
            "dq",
            metric(dq, dq_exact),
            "dk",
            metric(dk, dk_exact),
            f"zero={values['zero'] / values['count']:.6f}",
            f"sat={values['sat'] / values['count']:.3e}",
        )


parser = argparse.ArgumentParser()
parser.add_argument("--seq-lens", nargs="+", type=int, default=[512, 4096])
parser.add_argument("--heads", type=int, default=2)
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
args = parser.parse_args()
for seed in args.seeds:
    for seq_len in args.seq_lens:
        run(seq_len, args.heads, seed)

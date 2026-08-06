from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import torch
from flash_attn import flash_attn_func

from sageattention.cutlass_bwd import _BWD_CONFIG, _sageattn_cutlass_bwd_configured

CAPTURE = Path(__file__).resolve().parent / "sdxl_periodic_ds_inputs_training_dout_latent128.pt"


def bench(function: Callable[[], object], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-multiplier", type=int, choices=(1, 2), default=1)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    payload = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    record = payload["records"][0]
    multiplier = args.sequence_multiplier
    q = torch.cat([record["q"]] * multiplier, dim=1).cuda()
    k = torch.cat([record["k"]] * multiplier, dim=1).cuda()
    v = torch.cat([record["v"]] * multiplier, dim=1).cuda()
    dout = torch.cat([record["dout"]] * multiplier, dim=1).cuda()
    output, lse, _ = flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=64**-0.5,
        causal=False,
        return_attn_probs=True,
    )
    heads = q.shape[2]
    seq_len = q.shape[1]
    do_h = dout[0].permute(1, 0, 2).float().contiguous()
    out_h = output[0].permute(1, 0, 2).float().contiguous()
    v_h = v[0].permute(1, 0, 2).float().contiguous()

    def do_delta_summaries() -> tuple[torch.Tensor, torch.Tensor]:
        do_blocks = do_h.reshape(heads, seq_len // 32, 32, 64)
        out_blocks = out_h.reshape(heads, seq_len // 32, 32, 64)
        do_l2_max = do_blocks.square().sum(dim=-1).sqrt().amax(dim=-1)
        delta_max = (do_blocks * out_blocks).sum(dim=-1).abs().amax(dim=-1)
        return do_l2_max, delta_max

    def v_summaries() -> torch.Tensor:
        v_blocks = v_h.reshape(heads, seq_len // 64, 64, 64)
        return v_blocks.square().sum(dim=-1).sqrt().amax(dim=-1)

    def summaries() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        do_l2_max, delta_max = do_delta_summaries()
        return do_l2_max, delta_max, v_summaries()

    def backward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _sageattn_cutlass_bwd_configured(
            q,
            k,
            v,
            output,
            dout,
            lse,
            "NHD",
            _BWD_CONFIG,
        )

    iterations = args.iterations
    summary_ms = bench(summaries, 20, iterations)
    do_delta_ms = bench(do_delta_summaries, 20, iterations)
    v_ms = bench(v_summaries, 20, iterations)
    bwd_ms = bench(backward, 5, max(10, iterations // 4))
    forward_ms = bench(
        lambda: flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=64**-0.5, causal=False),
        10,
        max(20, iterations // 2),
    )
    print(
        f"shape={tuple(q.shape)} eager_summary_ms={summary_ms:.6f} "
        f"do_delta_ms={do_delta_ms:.6f} v_ms={v_ms:.6f} "
        f"cutlass_bwd_ms={bwd_ms:.6f} flash_fwd_ms={forward_ms:.6f}"
    )
    print(f"summary/bwd={summary_ms / bwd_ms:.4f} summary/(fwd+bwd)={summary_ms / (forward_ms + bwd_ms):.4f}")


if __name__ == "__main__":
    main()

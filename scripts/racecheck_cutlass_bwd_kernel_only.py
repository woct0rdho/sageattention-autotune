"""Launch only the native CUTLASS backward kernel for Compute Sanitizer.

All workspaces are created and populated before ``cudaProfilerStart`` without
Triton. Compute Sanitizer should be invoked with ``--target-processes
application-only`` and ``--kernel-name`` matching the CUTLASS kernel so that
PyTorch setup kernels are not instrumented as part of the checked region.
"""

import argparse
import importlib

import torch

_INT8_SCALE_INV = float.fromhex("0x1.010122p-7")
_INT8_SCALE_FLOOR = 2.0**-126


def _to_nhd(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    return tensor if layout == "NHD" else tensor.transpose(1, 2).contiguous()


def _quantize_blocks(tensor: torch.Tensor, block: int, layout: str) -> tuple[torch.Tensor, torch.Tensor]:
    logical = _to_nhd(tensor, layout).float()
    batch, seq_len, heads, head_dim = logical.shape
    blocks = (seq_len + block - 1) // block
    padded = torch.nn.functional.pad(logical, (0, 0, 0, 0, 0, blocks * block - seq_len))
    tiles = padded.permute(0, 2, 1, 3).reshape(batch, heads, blocks, block, head_dim)
    scale = tiles.abs().amax(dim=(-2, -1)) * _INT8_SCALE_INV + _INT8_SCALE_FLOOR
    quantized = torch.round(tiles / scale[..., None, None]).clamp_(-128, 127).to(torch.int8)
    quantized_nhd = quantized.reshape(batch, heads, blocks * block, head_dim)[:, :, :seq_len].permute(0, 2, 1, 3)
    result = quantized_nhd.contiguous() if layout == "NHD" else quantized_nhd.transpose(1, 2).contiguous()
    return result, scale.contiguous()


def _preprocess(
    output: torch.Tensor,
    grad_output: torch.Tensor,
    value: torch.Tensor,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    output_nhd = _to_nhd(output, layout).float()
    grad_output_nhd = _to_nhd(grad_output, layout).float()
    value_nhd = _to_nhd(value, layout).float()
    batch, seq_len, heads, head_dim = output_nhd.shape

    output_bhsd = output_nhd.permute(0, 2, 1, 3)
    grad_output_bhsd = grad_output_nhd.permute(0, 2, 1, 3)
    value_bhsd = value_nhd.permute(0, 2, 1, 3)
    delta = (output_bhsd * grad_output_bhsd).sum(dim=-1).contiguous()

    q_blocks = (seq_len + 31) // 32
    q_padded_len = q_blocks * 32
    dq_accum = torch.zeros((batch, heads, q_padded_len, head_dim), device=output.device, dtype=torch.float32)
    ds_sum = torch.zeros((batch, heads, q_padded_len), device=output.device, dtype=torch.float32)
    do_padded = torch.nn.functional.pad(grad_output_bhsd, (0, 0, 0, q_padded_len - seq_len))
    do_tiles = do_padded.reshape(batch, heads, q_blocks, 32, 4, 16)
    do_scale = do_tiles.abs().amax(dim=(-3, -1)) * _INT8_SCALE_INV + _INT8_SCALE_FLOOR
    do_int8 = torch.round(do_tiles / do_scale[..., None, :, None]).clamp_(-128, 127).to(torch.int8)
    do_int8 = do_int8.reshape(batch, heads, q_padded_len, head_dim)[:, :, :seq_len].contiguous()

    do_rows = do_padded.reshape(batch, heads, q_blocks, 32, head_dim)
    delta_padded = torch.nn.functional.pad(delta, (0, q_padded_len - seq_len)).reshape(batch, heads, q_blocks, 32)
    ds_q_factors = torch.stack(
        (torch.linalg.vector_norm(do_rows, dim=-1).amax(dim=-1), delta_padded.abs().amax(dim=-1)),
        dim=-1,
    ).contiguous()

    k_blocks = (seq_len + 63) // 64
    k_padded_len = k_blocks * 64
    value_padded = torch.nn.functional.pad(value_bhsd, (0, 0, 0, k_padded_len - seq_len))
    ds_k_factors = (
        torch.linalg.vector_norm(value_padded.reshape(batch, heads, k_blocks, 64, head_dim), dim=-1)
        .amax(dim=-1)
        .contiguous()
    )
    return delta, dq_accum, ds_sum, do_int8, do_scale.contiguous(), ds_q_factors, ds_k_factors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--layout", choices=("NHD", "HND"), default="NHD")
    parser.add_argument("--block-n", type=int, choices=(128, 256), default=128)
    parser.add_argument("--smooth-k", action="store_true")
    args = parser.parse_args()

    importlib.import_module("sageattention._qattn_cutlass_sm80")
    operator = torch.ops.sageattention_qattn_cutlass_sm80.qk_int8_sv_f16_accum_f32_attn_bwd_cutlass
    shape = (1, args.seq_len, args.num_heads, 64) if args.layout == "NHD" else (1, args.num_heads, args.seq_len, 64)
    torch.manual_seed(0)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    value = torch.randn_like(q)
    output = torch.randn_like(q)
    grad_output = torch.randn_like(q)
    lse = torch.randn((1, args.num_heads, args.seq_len), device="cuda", dtype=torch.float32)

    q_int8, q_scale = _quantize_blocks(q, 32, args.layout)
    seq_dim = 1 if args.layout == "NHD" else 2
    k_for_quant = k - k.mean(dim=seq_dim, keepdim=True) if args.smooth_k else k
    k_int8, k_scale = _quantize_blocks(k_for_quant, 64, args.layout)
    delta, dq_accum, ds_sum, do_int8, do_scale, ds_q_factors, ds_k_factors = _preprocess(
        output, grad_output, value, args.layout
    )
    grad_key = torch.empty_like(k)
    grad_value = torch.empty_like(value)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    operator(
        q_int8,
        k_int8,
        q_scale,
        k_scale,
        value,
        grad_output,
        lse,
        delta,
        dq_accum,
        ds_sum,
        do_int8,
        do_scale,
        ds_q_factors,
        ds_k_factors,
        grad_key,
        grad_value,
        0 if args.layout == "NHD" else 1,
        64**-0.5,
        32,
        64,
        64,
        args.block_n,
        args.smooth_k,
    )
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()

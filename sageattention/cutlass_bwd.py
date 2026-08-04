import importlib

import torch
import torch.nn.functional as F

from .triton.quant_per_block import per_block_int8
from .utils import _pad_qkv

_BWD_QK_CONFIG = (32, 64, 32, 64)
_BWD_CONFIGS = (_BWD_QK_CONFIG, (128, 64, 128, 64), (128, 128, 128, 128))

importlib.import_module(f"{__package__}._qattn_cutlass_sm80")
_qattn_cutlass_sm80 = torch.ops.sageattention_qattn_cutlass_sm80

CutlassSageBwdResult = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@torch.library.register_fake("sageattention_qattn_cutlass_sm80::qk_int8_sv_f16_accum_f32_attn_bwd_cutlass")
def _bwd_fake_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    lse: torch.Tensor,
    grad_query: torch.Tensor,
    grad_key: torch.Tensor,
    grad_value: torch.Tensor,
    tensor_layout: int,
    sm_scale: float,
    blk_q: int,
    blk_k: int,
    warp_q: int,
    warp_k: int,
) -> None:
    return None


def _sageattn_cutlass_bwd_configured(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    lse: torch.Tensor,
    tensor_layout: str,
    config: tuple[int, int, int, int],
) -> CutlassSageBwdResult:
    if not all(tensor.is_cuda for tensor in (q, k, v, output, grad_output, lse)):
        raise ValueError("Input tensors must be CUDA tensors.")
    if any(tensor.dtype != torch.float16 for tensor in (q, k, v, output, grad_output)):
        raise ValueError("CUTLASS qattn backward currently supports fp16 tensors only.")
    if lse.dtype != torch.float32:
        raise ValueError("lse must be a float32 tensor.")
    if q.shape != k.shape or q.shape != v.shape or q.shape != output.shape or q.shape != grad_output.shape:
        raise ValueError("q, k, v, output, and grad_output must have the same shape.")
    if len({q.device, k.device, v.device, output.device, grad_output.device, lse.device}) != 1:
        raise ValueError("All tensors must be on the same device.")

    head_dim, q, k, v = _pad_qkv(q, k, v)
    if output.size(-1) != q.size(-1):
        output = F.pad(output, (0, q.size(-1) - output.size(-1)))
    if grad_output.size(-1) != q.size(-1):
        grad_output = F.pad(grad_output, (0, q.size(-1) - grad_output.size(-1)))
    if q.size(-1) != 64:
        raise ValueError("Focused CUTLASS qattn backward currently supports padded head_dim 64 only.")
    if any(tensor.stride(-1) != 1 for tensor in (q, k, v, output, grad_output)):
        raise ValueError("Last dimension of q, k, v, output, and grad_output must be contiguous.")

    if tensor_layout == "NHD":
        layout_i = 0
        seq_len = q.size(1)
        num_heads = q.size(2)
    elif tensor_layout == "HND":
        layout_i = 1
        num_heads = q.size(1)
        seq_len = q.size(2)
    else:
        raise ValueError("tensor_layout must be 'NHD' or 'HND'.")
    if lse.shape != (q.size(0), num_heads, seq_len):
        raise ValueError("lse must have shape (batch, heads, seq_len).")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    output = output.contiguous()
    grad_output = grad_output.contiguous()
    lse = lse.contiguous()
    blk_q, blk_k, warp_q, warp_k = config
    q_int8, q_scale, k_int8, k_scale = per_block_int8(
        q,
        k,
        km=None,
        BLKQ=blk_q,
        BLKK=blk_k,
        tensor_layout=tensor_layout,
    )
    grad_query = torch.empty_like(q)
    grad_key = torch.empty_like(k)
    grad_value = torch.empty_like(v)
    _qattn_cutlass_sm80.qk_int8_sv_f16_accum_f32_attn_bwd_cutlass(
        q_int8,
        k_int8,
        q_scale,
        k_scale,
        v,
        output,
        grad_output,
        lse,
        grad_query,
        grad_key,
        grad_value,
        layout_i,
        head_dim**-0.5,
        blk_q,
        blk_k,
        warp_q,
        warp_k,
    )
    return grad_query[..., :head_dim], grad_key[..., :head_dim], grad_value[..., :head_dim]


def sageattn_cutlass_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    lse: torch.Tensor,
    tensor_layout: str = "HND",
) -> CutlassSageBwdResult:
    return _sageattn_cutlass_bwd_configured(q, k, v, output, grad_output, lse, tensor_layout, _BWD_QK_CONFIG)

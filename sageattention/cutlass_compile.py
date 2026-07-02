import importlib

import torch

importlib.import_module(f"{__package__}._qattn_cutlass_sm80")
_qattn_cutlass_sm80 = torch.ops.sageattention_qattn_cutlass_sm80


def _empty_lse(query: torch.Tensor, tensor_layout: int, return_lse: bool) -> torch.Tensor:
    batch_size = query.size(0)

    if tensor_layout == 0:
        num_qo_heads = query.size(2)
        qo_len = query.size(1)
    else:
        num_qo_heads = query.size(1)
        qo_len = query.size(2)

    if return_lse:
        lse = torch.empty((batch_size, num_qo_heads, qo_len), device=query.device, dtype=torch.float32)
    else:
        lse = torch.empty((0,), device=query.device, dtype=torch.float32)
    return lse


def _fake_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    tensor_layout: int,
    sm_scale: float,
    blk_q: int,
    blk_k: int,
    warp_q: int,
    warp_k: int,
    return_lse: bool,
) -> torch.Tensor:
    return _empty_lse(query, tensor_layout, return_lse)


torch.library.register_fake("sageattention_qattn_cutlass_sm80::qk_int8_sv_f16_accum_f32_attn_cutlass")(_fake_impl)

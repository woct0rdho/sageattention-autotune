import torch

from . import _qattn_sm80

_qattn_sm80 = torch.ops.sageattention_qattn_sm80  # noqa: F811


def _sm80_qk_fake_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    tensor_layout: int,
    is_causal: int,
    qk_quant_gran: int,
    sm_scale: float,
    return_lse: int,
) -> torch.Tensor:
    batch_size = query.size(0)
    if tensor_layout == 0:
        qo_len = query.size(1)
        num_qo_heads = query.size(2)
    else:
        num_qo_heads = query.size(1)
        qo_len = query.size(2)

    if return_lse:
        return torch.empty((batch_size, num_qo_heads, qo_len), dtype=torch.float32, device=query.device)
    return torch.empty((0,), dtype=torch.float32, device=query.device)


torch.library.register_fake("sageattention_qattn_sm80::qk_int8_sv_f16_accum_f32_attn")(_sm80_qk_fake_impl)
torch.library.register_fake("sageattention_qattn_sm80::qk_int8_sv_f16_accum_f16_attn_inst_buf")(_sm80_qk_fake_impl)


@torch.library.register_fake("sageattention_qattn_sm80::qk_int8_sv_f16_accum_f16_attn")
def _qk_int8_sv_f16_accum_f16_attn_fake_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    tensor_layout: int,
    is_causal: int,
    qk_quant_gran: int,
    sm_scale: float,
    qk_tile_config: int,
    return_lse: int,
) -> torch.Tensor:
    return _sm80_qk_fake_impl(
        query,
        key,
        value,
        output,
        query_scale,
        key_scale,
        tensor_layout,
        is_causal,
        qk_quant_gran,
        sm_scale,
        return_lse,
    )


@torch.library.register_fake("sageattention_qattn_sm80::qk_int8_sv_f16_accum_f16_fuse_v_mean_attn")
def _qk_int8_sv_f16_accum_f16_fuse_v_mean_attn_fake_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value_mean: torch.Tensor,
    tensor_layout: int,
    is_causal: int,
    qk_quant_gran: int,
    sm_scale: float,
    return_lse: int,
) -> torch.Tensor:
    return _sm80_qk_fake_impl(
        query,
        key,
        value,
        output,
        query_scale,
        key_scale,
        tensor_layout,
        is_causal,
        qk_quant_gran,
        sm_scale,
        return_lse,
    )

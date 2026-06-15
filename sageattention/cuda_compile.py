import importlib

import torch

from .utils import _env_flag_enabled

importlib.import_module(f"{__package__}._qattn_sm80")
_qattn_sm80 = torch.ops.sageattention_qattn_sm80

# The fp8 (sm89/sm120) extension is optional: it is only built when a CUDA
# toolkit new enough for the target arch is available. Fall back gracefully.
try:
    importlib.import_module(f"{__package__}._qattn_sm89")
except (ImportError, OSError):
    _qattn_sm89 = None
else:
    _qattn_sm89 = torch.ops.sageattention_qattn_sm89

# Arches whose int8-QK / fp8-PV kernels we ship: Ada (sm_89) and Blackwell (sm_120).
_FP8_ARCHES = (89, 120)


def use_fp8_backend(device: torch.device) -> bool:
    """Whether to route attention through the fp8 (sv_f8) sm89/sm120 kernels.

    True on Ada / Blackwell when the fp8 extension is built, unless disabled via
    SAGEATTN_DISABLE_FP8.
    """
    if _qattn_sm89 is None or _env_flag_enabled("SAGEATTN_DISABLE_FP8"):
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return (major * 10 + minor) in _FP8_ARCHES


def _empty_lse(query: torch.Tensor, tensor_layout: int, return_lse: bool) -> torch.Tensor:
    batch_size = query.size(0)

    if tensor_layout == 0:
        num_qo_heads = query.size(2)
        qo_len = query.size(1)
    else:
        num_qo_heads = query.size(1)
        qo_len = query.size(2)

    if return_lse:
        lse = torch.empty((batch_size, num_qo_heads, qo_len), dtype=torch.float32, device=query.device)
    else:
        lse = torch.empty((0,), dtype=torch.float32, device=query.device)
    return lse


def _fake_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    tensor_layout: int,
    is_causal: bool,
    sm_scale: float,
    blk_q: int,
    blk_k: int,
    warp_q: int,
    warp_k: int,
    return_lse: bool,
) -> torch.Tensor:
    return _empty_lse(query, tensor_layout, return_lse)


torch.library.register_fake("sageattention_qattn_sm80::qk_int8_sv_f16_accum_f32_attn")(_fake_impl)
torch.library.register_fake("sageattention_qattn_sm80::qk_int8_sv_f16_accum_f16_attn")(_fake_impl)
torch.library.register_fake("sageattention_qattn_sm80::qk_int8_sv_f16_accum_f16_attn_inst_buf")(_fake_impl)


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
    is_causal: bool,
    sm_scale: float,
    blk_q: int,
    blk_k: int,
    warp_q: int,
    warp_k: int,
    return_lse: bool,
) -> torch.Tensor:
    return _empty_lse(query, tensor_layout, return_lse)


def _fake_impl_fp8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    tensor_layout: int,
    is_causal: bool,
    sm_scale: float,
    blk_q: int,
    blk_k: int,
    warp_q: int,
    warp_k: int,
    return_lse: bool,
) -> torch.Tensor:
    return _empty_lse(query, tensor_layout, return_lse)


if _qattn_sm89 is not None:
    torch.library.register_fake("sageattention_qattn_sm89::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn")(_fake_impl_fp8)
    torch.library.register_fake("sageattention_qattn_sm89::qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf")(
        _fake_impl_fp8
    )

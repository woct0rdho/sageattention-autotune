import functools

import torch
from torch._inductor.kernel.custom_op import CustomOpConfig, register_custom_op_autotuning

from . import autotune_utils
from .utils import _padded_head_dim

_AUTOTUNE_CONFIGS = (
    (128, 64, 32, 64),
    (128, 32, 32, 32),
    (64, 64, 32, 64),
    (128, 64, 16, 64),
)
_AUTOTUNE_CACHE: dict[object, tuple[int, int, int, int]] = {}


@functools.cache
def _config_is_valid(
    config: tuple[int, int, int, int],
    head_dim: int,
    is_causal: bool,
    device_index: int,
) -> bool:
    if is_causal:
        return False

    head_dim = _padded_head_dim(head_dim)
    if head_dim not in (64, 128):
        return False

    blk_q, blk_k, _, _ = config
    smem_bytes = head_dim * max(blk_q + 3 * blk_k, 2 * blk_q)
    return smem_bytes <= autotune_utils._shared_memory_limit(device_index)


@functools.cache
def _valid_configs(
    head_dim: int,
    is_causal: bool,
    device_index: int,
) -> tuple[tuple[int, int, int, int], ...]:
    return autotune_utils._valid_configs(_AUTOTUNE_CONFIGS, _config_is_valid, head_dim, is_causal, device_index)


def _eager_autotune_select(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    smooth_k: bool,
    return_lse: bool,
) -> tuple[int, int, int, int]:
    from .cutlass_attn import _sageattn_cutlass_configured

    configs = _valid_configs(q.size(-1), False, q.device.index)
    key = autotune_utils._tensor_autotune_cache_key(q, k, v, tensor_layout, False, "fp32", smooth_k, return_lse)
    return autotune_utils._eager_autotune_select(
        configs,
        _AUTOTUNE_CACHE,
        key,
        lambda config: _sageattn_cutlass_configured(q, k, v, tensor_layout, smooth_k, return_lse, config),
    )


@torch.library.custom_op("sageattention_internal::sageattn_cutlass_autotuned", mutates_args=())
def _sageattn_cutlass_autotuned(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    smooth_k: bool,
    blk_q: int = 0,
    blk_k: int = 0,
    warp_q: int = 0,
    warp_k: int = 0,
) -> torch.Tensor:
    from .cutlass_attn import _sageattn_cutlass_configured

    qk_config = (blk_q, blk_k, warp_q, warp_k)
    configs = _valid_configs(q.size(-1), False, q.device.index)
    if min(qk_config) <= 0 or qk_config not in configs:
        qk_config = configs[0]

    return _sageattn_cutlass_configured(q, k, v, tensor_layout, smooth_k, False, qk_config)


@_sageattn_cutlass_autotuned.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    smooth_k: bool,
    blk_q: int = 0,
    blk_k: int = 0,
    warp_q: int = 0,
    warp_k: int = 0,
) -> torch.Tensor:
    return torch.empty_like(q)


register_custom_op_autotuning(
    _sageattn_cutlass_autotuned,
    config_generator=lambda fake_tensors: [
        CustomOpConfig(
            blk_q=config[0],
            blk_k=config[1],
            warp_q=config[2],
            warp_k=config[3],
        )
        for config in _valid_configs(fake_tensors["q"].size(-1), False, fake_tensors["q"].device.index)
    ],
)

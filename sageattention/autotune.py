import os
from typing import Optional

import torch
import triton
from torch._inductor.kernel.custom_op import CustomOpConfig, register_custom_op_autotuning

_AUTOTUNE_CONFIGS = (
    (128, 64, 32, 64),
    (128, 32, 32, 32),
    (64, 64, 32, 64),
    (128, 64, 16, 64),
)
_AUTOTUNE_CACHE = {}


def _config_is_valid(
    config: tuple[int, int, int, int],
    head_dim: int,
    is_causal: bool,
    device: torch.device,
) -> bool:
    blk_q, blk_k, warp_q, warp_k = config
    if head_dim not in (64, 128, 256):
        return False
    if blk_q % warp_q != 0 or blk_k % warp_k != 0:
        return False
    if warp_q % 16 != 0 or warp_k % 16 != 0:
        return False
    if is_causal and blk_q // blk_k > 2:
        return False

    num_warps = (blk_q // warp_q) * (blk_k // warp_k)
    if num_warps <= 0:
        return False

    props = torch.cuda.get_device_properties(device)
    if 32 * num_warps > props.max_threads_per_block:
        return False

    qk_copy_lines = 8 if head_dim == 64 else 4
    v_copy_lines = 4
    if blk_q % (num_warps * qk_copy_lines) != 0:
        return False
    if blk_k % (num_warps * qk_copy_lines) != 0:
        return False
    if blk_q % (num_warps * v_copy_lines) != 0:
        return False
    if blk_k % (num_warps * v_copy_lines) != 0:
        return False

    smem_bytes = head_dim * max(blk_q + 3 * blk_k, 2 * blk_q)
    smem_limit = getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)
    return smem_bytes <= smem_limit


def _valid_configs(
    q: torch.Tensor,
    is_causal: bool,
) -> tuple[tuple[int, int, int, int], ...]:
    return _valid_configs_for_head_dim(q.size(-1), is_causal, q.device)


def _valid_configs_for_head_dim(
    head_dim: int,
    is_causal: bool,
    device: torch.device,
) -> tuple[tuple[int, int, int, int], ...]:
    configs = tuple(config for config in _AUTOTUNE_CONFIGS if _config_is_valid(config, head_dim, is_causal, device))
    if not configs:
        raise RuntimeError(f"No valid config for head_dim={head_dim} is_causal={is_causal}.")
    return configs


def _autotune_cache_key(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout_i: int,
    is_causal: bool,
    pv_accum_i: int,
    smooth_k: bool,
    smooth_v: bool,
    return_lse: bool,
):
    device_index = q.device.index if q.device.index is not None else torch.cuda.current_device()
    return (
        device_index,
        q.dtype,
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        tuple(q.stride()),
        tuple(k.stride()),
        tuple(v.stride()),
        layout_i,
        is_causal,
        pv_accum_i,
        smooth_k,
        smooth_v,
        return_lse,
    )


def _eager_autotune_select(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout_i: int,
    is_causal: bool,
    pv_accum_i: int,
    sm_scale: Optional[float],
    smooth_k: bool,
    smooth_v: bool,
    return_lse: bool,
) -> tuple[int, int, int, int]:
    from .core import _sageattn_configured

    configs = _valid_configs(q, is_causal)
    if len(configs) == 1:
        return configs[0]

    key = _autotune_cache_key(q, k, v, layout_i, is_causal, pv_accum_i, smooth_k, smooth_v, return_lse)
    cached = _AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached

    warmup_ms = max(1, int(os.environ.get("SAGEATTN_AUTOTUNE_WARMUP_MS", "25")))
    rep_ms = max(1, int(os.environ.get("SAGEATTN_AUTOTUNE_REP_MS", "100")))
    best_config = configs[0]
    best_ms = None

    for config in configs:
        ms = triton.testing.do_bench(
            lambda config=config: _sageattn_configured(
                q,
                k,
                v,
                layout_i,
                is_causal,
                sm_scale,
                pv_accum_i,
                smooth_k,
                smooth_v,
                return_lse,
                config,
            ),
            warmup=warmup_ms,
            rep=rep_ms,
        )
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best_config = config

    _AUTOTUNE_CACHE[key] = best_config
    return best_config


@torch.library.custom_op("sageattention_internal::sageattn_autotuned", mutates_args=())
def _sageattn_autotuned(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout_i: int,
    is_causal: bool,
    sm_scale: float,
    pv_accum_i: int,
    smooth_k: bool,
    smooth_v: bool,
    blk_q: int = 0,
    blk_k: int = 0,
    warp_q: int = 0,
    warp_k: int = 0,
) -> torch.Tensor:
    from .core import _sageattn_configured

    qk_config = (blk_q, blk_k, warp_q, warp_k)
    if min(qk_config) <= 0 or qk_config not in _valid_configs(q, is_causal):
        qk_config = _valid_configs(q, is_causal)[0]

    return _sageattn_configured(
        q,
        k,
        v,
        layout_i,
        is_causal,
        sm_scale,
        pv_accum_i,
        smooth_k,
        smooth_v,
        False,
        qk_config,
    )


@_sageattn_autotuned.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout_i: int,
    is_causal: bool,
    sm_scale: float,
    pv_accum_i: int,
    smooth_k: bool,
    smooth_v: bool,
    blk_q: int = 0,
    blk_k: int = 0,
    warp_q: int = 0,
    warp_k: int = 0,
) -> torch.Tensor:
    return torch.empty_like(q)


register_custom_op_autotuning(
    _sageattn_autotuned,
    config_generator=lambda fake_tensors: [
        CustomOpConfig(blk_q=cfg[0], blk_k=cfg[1], warp_q=cfg[2], warp_k=cfg[3])
        for cfg in _valid_configs_for_head_dim(
            fake_tensors["q"].shape[-1],
            False,
            fake_tensors["q"].device,
        )
    ],
)

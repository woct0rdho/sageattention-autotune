from typing import Optional

import torch
from torch._inductor.kernel.custom_op import CustomOpConfig, register_custom_op_autotuning

from . import autotune_utils
from .utils import _padded_head_dim

_TRITON_BLOCK_CONFIGS = (
    (256, 64),
    (128, 64),
    (128, 32),
    (64, 64),
    (64, 32),
    (32, 32),
)
_TRITON_QUANT_NUM_WARPS = (4, 8)
_TRITON_ATTN_CONFIGS = (
    # (attn_num_warps, attn_num_stages)
    (4, 2),
    (4, 3),
    (4, 4),
    (8, 2),
    (8, 3),
    (8, 4),
)
_TRITON_AUTOTUNE_CONFIGS = tuple(
    (block_m, block_n, quant_num_warps, attn_num_warps, attn_num_stages)
    for block_m, block_n in _TRITON_BLOCK_CONFIGS
    for quant_num_warps in _TRITON_QUANT_NUM_WARPS
    for attn_num_warps, attn_num_stages in _TRITON_ATTN_CONFIGS
)
_TRITON_AUTOTUNE_CACHE = {}


def _estimated_triton_smem_bytes(
    block_m: int,
    block_n: int,
    head_dim: int,
    attn_num_stages: int,
    is_causal: bool,
) -> int:
    int8_bytes = 1
    fp16_bytes = 2
    min_smem_bytes = 8 * 1024
    pipeline_prologue_stages = 1
    stage_bookkeeping_bytes = 4
    causal_mask_smem_slack_bytes = 16

    # Triton reports shared memory as a linear function of pipeline stages for this kernel.
    # Each extra live K/V pipeline stage materializes one K int8 tile and one V fp16 tile.
    # PV_ACCUM_FP32 changes registers/runtime, but not the reported shared memory.
    operand_tile_bytes = head_dim * (block_m + block_n)
    kv_pipeline_stage_bytes = head_dim * block_n * (int8_bytes + fp16_bytes)
    live_kv_pipeline_stages = max(attn_num_stages - pipeline_prologue_stages, 1)

    estimated = operand_tile_bytes + live_kv_pipeline_stages * (kv_pipeline_stage_bytes + stage_bookkeeping_bytes)
    estimated = max(estimated, min_smem_bytes)
    if is_causal:
        estimated += causal_mask_smem_slack_bytes
    return estimated


def _triton_config_is_valid(
    config: tuple[int, int, int, int, int],
    head_dim: int,
    is_causal: bool,
    device: torch.device,
) -> bool:
    block_m, block_n, _, _, attn_num_stages = config
    if is_causal and block_m % block_n != 0:
        return False

    head_dim = _padded_head_dim(head_dim)
    return _estimated_triton_smem_bytes(
        block_m, block_n, head_dim, attn_num_stages, is_causal
    ) <= autotune_utils._shared_memory_limit(device)


def _valid_triton_configs(
    q: torch.Tensor,
    is_causal: bool,
) -> tuple[tuple[int, int, int, int, int], ...]:
    return _valid_triton_configs_for_head_dim(q.size(-1), is_causal, q.device)


def _valid_triton_configs_for_head_dim(
    head_dim: int,
    is_causal: bool,
    device: torch.device,
) -> tuple[tuple[int, int, int, int, int], ...]:
    return autotune_utils._valid_configs_for_head_dim(
        _TRITON_AUTOTUNE_CONFIGS, _triton_config_is_valid, head_dim, is_causal, device
    )


def _valid_triton_block_configs_for_head_dim(
    head_dim: int,
    is_causal: bool,
    device: torch.device,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        dict.fromkeys(config[:2] for config in _valid_triton_configs_for_head_dim(head_dim, is_causal, device))
    )


def _eager_triton_autotune_select(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout_i: int,
    is_causal: bool,
    pv_accum_i: int,
    sm_scale: Optional[float],
    smooth_k: bool,
    return_lse: bool,
) -> tuple[int, int, int, int, int]:
    from .triton_attention import _sageattn_triton_configured

    configs = _valid_triton_configs(q, is_causal)
    key = autotune_utils._tensor_autotune_cache_key(q, k, v, layout_i, is_causal, pv_accum_i, smooth_k, return_lse)
    return autotune_utils._eager_autotune_select(
        configs,
        _TRITON_AUTOTUNE_CACHE,
        key,
        lambda config: _sageattn_triton_configured(
            q,
            k,
            v,
            layout_i,
            is_causal,
            sm_scale,
            pv_accum_i,
            smooth_k,
            return_lse,
            config,
        ),
    )


@torch.library.custom_op("sageattention_internal::sageattn_triton_autotuned", mutates_args=())
def _sageattn_triton_autotuned(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout_i: int,
    is_causal: bool,
    sm_scale: float,
    pv_accum_i: int,
    smooth_k: bool,
    block_m: int = 0,
    block_n: int = 0,
    quant_num_warps: int = 0,
    attn_num_warps: int = 0,
    attn_num_stages: int = 0,
) -> torch.Tensor:
    from .triton_attention import _sageattn_triton_configured

    config = (block_m, block_n, quant_num_warps, attn_num_warps, attn_num_stages)
    if min(config) <= 0 or config not in _valid_triton_configs(q, is_causal):
        config = _valid_triton_configs(q, is_causal)[0]

    return _sageattn_triton_configured(
        q,
        k,
        v,
        layout_i,
        is_causal,
        sm_scale,
        pv_accum_i,
        smooth_k,
        False,
        config,
    )


@_sageattn_triton_autotuned.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout_i: int,
    is_causal: bool,
    sm_scale: float,
    pv_accum_i: int,
    smooth_k: bool,
    block_m: int = 0,
    block_n: int = 0,
    quant_num_warps: int = 0,
    attn_num_warps: int = 0,
    attn_num_stages: int = 0,
) -> torch.Tensor:
    return torch.empty_like(q)


register_custom_op_autotuning(
    _sageattn_triton_autotuned,
    config_generator=lambda fake_tensors: [
        CustomOpConfig(
            block_m=cfg[0],
            block_n=cfg[1],
            quant_num_warps=cfg[2],
            attn_num_warps=cfg[3],
            attn_num_stages=cfg[4],
        )
        for cfg in _valid_triton_configs_for_head_dim(
            fake_tensors["q"].shape[-1],
            False,
            fake_tensors["q"].device,
        )
    ],
)

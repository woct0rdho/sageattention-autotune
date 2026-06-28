import functools

import torch
import triton

from ..autotune_utils import _shared_memory_limit
from ..utils import _padded_head_dim

_TRITON_BWD_CONFIGS = (
    # (num_warps, num_stages)
    (4, 2),
    (4, 3),
    (4, 4),
    # num_warps = 8 is currently disabled because it does not help much
    # (8, 2),
    # (8, 3),
    # (8, 4),
)

_TRITON_BWD_REUSE_CONFIGS = (
    # (num_warps, num_stages)
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
    # num_warps = 8 is currently disabled because it does not help much
    # (8, 1),
    # (8, 2),
    # (8, 3),
    # (8, 4),
)


def _estimated_bwd_dq_smem_bytes(
    block_m: int,
    block_n: int,
    head_dim: int,
    num_stages: int,
) -> int:
    fp32_bytes = 4
    stage_bookkeeping_bytes = 4
    min_smem_bytes = 8 * 1024

    head_dim = _padded_head_dim(head_dim)

    # Triton's reported dQ shared memory is stage-linear for these kernels:
    #   3 * D * BLOCK_M                          persistent dQ/dO-style operands
    # + (num_stages - 1) * (4 * D * BLOCK_N + 4) staged K/V-like pipeline storage
    # + BLOCK_M * BLOCK_N if BLOCK_M < D else 16 small tile/slack term
    # This matches compiled kernels for D in {64, 128, 256} and BLOCK configs used here.
    persistent_operand_bytes = 3 * head_dim * block_m
    pipeline_bytes = max(num_stages - 1, 0) * (fp32_bytes * head_dim * block_n + stage_bookkeeping_bytes)
    tile_slack_bytes = block_m * block_n if block_m < head_dim else 16
    return max(persistent_operand_bytes + pipeline_bytes + tile_slack_bytes, min_smem_bytes)


def _estimated_bwd_dkdv_smem_bytes(
    block_m: int,
    block_n: int,
    head_dim: int,
    num_stages: int,
) -> int:
    min_smem_bytes = 8 * 1024

    head_dim = _padded_head_dim(head_dim)

    # For sequence lengths with more than one BLOCK_M iteration, dK/dV shared
    # memory fits this expression exactly. It is conservative for single-iteration
    # sequences, where Triton can optimize away some loop-carried buffers.
    estimated = (
        block_m * block_n
        + 3 * head_dim * block_n
        + num_stages * (2 * head_dim * block_m + 8 * block_m + 4)
        - 8 * block_m
        - 4
    )
    return max(estimated, min_smem_bytes)


def _estimated_bwd_reuse_smem_bytes(
    block_m: int,
    block_n: int,
    head_dim: int,
    num_warps: int,
    num_stages: int,
) -> int:
    min_smem_bytes = 8 * 1024

    head_dim = _padded_head_dim(head_dim)

    # The reuse prototype keeps Q, dO, K, V, P/dS, dQ partials, and dK/dV
    # accumulators live in one KV-owned kernel. Compiled SM86 kernels report a
    # base footprint of MN + 4D(M+N), then a stage-linear pipeline term. The
    # small branch terms below cover Triton's observed warp/layout slack without
    # pruning valid near-limit D=64 benchmark configs.
    base_bytes = block_m * block_n + 4 * head_dim * (block_m + block_n)
    if num_stages == 1:
        estimated = base_bytes
    else:
        pipeline_stage_bytes = 2 * head_dim * block_m + 8 * block_m + 4
        estimated = base_bytes + 8 * block_m + 4 + (num_stages - 2) * pipeline_stage_bytes

    if block_m == block_n == head_dim == 64 and num_stages >= 2:
        estimated += block_m * block_n * (1 if num_warps == 4 else 2)
    elif num_warps == 8 and block_n > block_m and num_stages >= 2:
        estimated += head_dim * (block_n - block_m) * max(block_m // 16 - 1, 1)
    elif num_warps == 8 and block_m > block_n and num_stages >= 2:
        estimated += 2 * head_dim * (block_m - block_n)

    return max(estimated, min_smem_bytes)


@functools.cache
def _bwd_dq_config_is_valid(
    block_m: int,
    block_n: int,
    num_stages: int,
    head_dim: int,
    device_index: int,
) -> bool:
    return _estimated_bwd_dq_smem_bytes(block_m, block_n, head_dim, num_stages) <= _shared_memory_limit(device_index)


@functools.cache
def _bwd_dkdv_config_is_valid(
    block_m: int,
    block_n: int,
    num_stages: int,
    head_dim: int,
    device_index: int,
) -> bool:
    return _estimated_bwd_dkdv_smem_bytes(block_m, block_n, head_dim, num_stages) <= _shared_memory_limit(device_index)


@functools.cache
def _bwd_reuse_config_is_valid(
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
    head_dim: int,
    device_index: int,
) -> bool:
    return _estimated_bwd_reuse_smem_bytes(block_m, block_n, head_dim, num_warps, num_stages) <= _shared_memory_limit(
        device_index
    )


def _prune_bwd_dq_configs(
    configs: list[triton.Config], named_args: dict[str, object], **meta: object
) -> list[triton.Config]:
    q = named_args["Q"]
    assert isinstance(q, torch.Tensor)
    block_m = meta["BLOCK_M"]
    block_n = meta["BLOCK_N"]
    head_dim = meta["HEAD_DIM"]
    assert isinstance(block_m, int)
    assert isinstance(block_n, int)
    assert isinstance(head_dim, int)
    return [
        config
        for config in configs
        if _bwd_dq_config_is_valid(block_m, block_n, config.num_stages, head_dim, q.device.index)
    ]


def _prune_bwd_dkdv_configs(
    configs: list[triton.Config], named_args: dict[str, object], **meta: object
) -> list[triton.Config]:
    q = named_args["Q"]
    assert isinstance(q, torch.Tensor)
    block_m = meta["BLOCK_M"]
    block_n = meta["BLOCK_N"]
    head_dim = meta["HEAD_DIM"]
    assert isinstance(block_m, int)
    assert isinstance(block_n, int)
    assert isinstance(head_dim, int)
    return [
        config
        for config in configs
        if _bwd_dkdv_config_is_valid(block_m, block_n, config.num_stages, head_dim, q.device.index)
    ]


def _prune_bwd_reuse_configs(
    configs: list[triton.Config], named_args: dict[str, object], **meta: object
) -> list[triton.Config]:
    q = named_args["Q"]
    assert isinstance(q, torch.Tensor)
    block_m = meta["BLOCK_M"]
    block_n = meta["BLOCK_N"]
    head_dim = meta["HEAD_DIM"]
    assert isinstance(block_m, int)
    assert isinstance(block_n, int)
    assert isinstance(head_dim, int)
    return [
        config
        for config in configs
        if _bwd_reuse_config_is_valid(block_m, block_n, config.num_warps, config.num_stages, head_dim, q.device.index)
    ]


@functools.cache
def _valid_bwd_dq_configs(
    block_config: tuple[int, int],
    head_dim: int,
    device_index: int,
) -> tuple[tuple[int, int], ...]:
    block_m, block_n = block_config
    return tuple(
        bwd_config
        for bwd_config in _TRITON_BWD_CONFIGS
        if _bwd_dq_config_is_valid(block_m, block_n, bwd_config[1], head_dim, device_index)
    )


@functools.cache
def _valid_bwd_dkdv_configs(
    block_config: tuple[int, int],
    head_dim: int,
    device_index: int,
) -> tuple[tuple[int, int], ...]:
    block_m, block_n = block_config
    return tuple(
        bwd_config
        for bwd_config in _TRITON_BWD_CONFIGS
        if _bwd_dkdv_config_is_valid(block_m, block_n, bwd_config[1], head_dim, device_index)
    )


@functools.cache
def _valid_bwd_reuse_configs(
    block_config: tuple[int, int],
    head_dim: int,
    device_index: int,
) -> tuple[tuple[int, int], ...]:
    block_m, block_n = block_config
    return tuple(
        bwd_config
        for bwd_config in _TRITON_BWD_REUSE_CONFIGS
        if _bwd_reuse_config_is_valid(block_m, block_n, bwd_config[0], bwd_config[1], head_dim, device_index)
    )


@functools.cache
def _has_valid_bwd_configs(
    block_config: tuple[int, int],
    head_dim: int,
    device_index: int,
) -> bool:
    return bool(_valid_bwd_dq_configs(block_config, head_dim, device_index)) and bool(
        _valid_bwd_dkdv_configs(block_config, head_dim, device_index)
    )

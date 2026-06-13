import logging
import os
from collections.abc import Callable, Sequence
from typing import TypeVar

import torch
import triton

ConfigT = TypeVar("ConfigT", bound=tuple[int, ...])

_logger = logging.getLogger(__name__)
_AUTOTUNE_SEQ_LEN_BUCKETS = (16, 32, 64, 128, 256)


def _shared_memory_limit(device: torch.device) -> int:
    props = torch.cuda.get_device_properties(device)
    return getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)


def _autotune_seq_len_bucket(seq_len: int) -> int:
    if seq_len <= 0:
        return seq_len

    for bucket_size in _AUTOTUNE_SEQ_LEN_BUCKETS:
        if seq_len <= bucket_size:
            return bucket_size
    return triton.cdiv(seq_len, _AUTOTUNE_SEQ_LEN_BUCKETS[-1]) * _AUTOTUNE_SEQ_LEN_BUCKETS[-1]


def _tensor_shape_autotune_key(tensor: torch.Tensor, seq_dim: int) -> tuple[int, ...]:
    return tuple(_autotune_seq_len_bucket(dim) if i == seq_dim else dim for i, dim in enumerate(tensor.shape))


def _tensor_autotune_cache_key(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    *extra: object,
) -> tuple[object, ...]:
    if tensor_layout == "NHD":
        seq_dim = 1
    elif tensor_layout == "HND":
        seq_dim = 2
    else:
        raise ValueError("tensor_layout must be 'NHD' or 'HND'.")

    return (
        q.device.index,
        q.dtype,
        _tensor_shape_autotune_key(q, seq_dim),
        _tensor_shape_autotune_key(k, seq_dim),
        tensor_layout,
        *extra,
    )


def _valid_configs_for_head_dim(
    candidates: Sequence[ConfigT],
    is_valid: Callable[[ConfigT, int, bool, torch.device], bool],
    head_dim: int,
    is_causal: bool,
    device: torch.device,
) -> tuple[ConfigT, ...]:
    configs = tuple(config for config in candidates if is_valid(config, head_dim, is_causal, device))
    if not configs:
        raise RuntimeError(f"No valid config for head_dim={head_dim} is_causal={is_causal}.")
    return configs


def _eager_autotune_select(
    configs: Sequence[ConfigT],
    cache: dict[object, ConfigT],
    cache_key: object,
    benchmark: Callable[[ConfigT], object],
) -> ConfigT:
    if len(configs) == 1:
        return configs[0]

    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    warmup_ms = max(1, int(os.environ.get("SAGEATTN_AUTOTUNE_WARMUP_MS", "25")))
    rep_ms = max(1, int(os.environ.get("SAGEATTN_AUTOTUNE_REP_MS", "100")))
    best_config = configs[0]
    best_ms = None

    for config in configs:
        ms = triton.testing.do_bench(
            lambda config=config: benchmark(config),
            warmup=warmup_ms,
            rep=rep_ms,
        )
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best_config = config

    cache[cache_key] = best_config
    _logger.info("SageAttention cached autotune config %s for key %s", best_config, cache_key)
    return best_config

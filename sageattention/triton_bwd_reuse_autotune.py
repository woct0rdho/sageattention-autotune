import functools
import os

import torch
import torch._dynamo.config as dynamo_config
from torch._inductor import config
from torch._inductor.kernel.custom_op import CustomOpConfig, register_custom_op_autotuning
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.utils import do_bench_using_profiling

from . import autotune_utils
from .torch_compile_patch import register_custom_timing_target
from .triton.attn_bwd_autotune import _valid_bwd_reuse_configs
from .triton_autotune import _valid_configs as _valid_forward_configs

_TRITON_TRAINABLE_REUSE_AUTOTUNE_CACHE: dict[object, tuple[int, int]] = {}
_TRITON_TRAINABLE_REUSE_COMPILE_AUTOTUNE_NAME = "_sageattention_triton_trainable_reuse_autotuned"
_SELECTED_COMPILE_REUSE_BLOCK_CONFIGS: dict[object, tuple[int, int]] = {}


def _enable_dynamo_backward_tracing() -> None:
    setattr(dynamo_config, "trace_autograd_ops", True)


def _compile_block_config_key(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    pv_accum_dtype: str,
    smooth_k: bool,
) -> tuple[object, ...]:
    return autotune_utils._tensor_autotune_cache_key(
        q,
        k,
        v,
        tensor_layout,
        False,
        pv_accum_dtype,
        smooth_k,
        os.environ.get("SAGEATTN_REUSE_DQ_SPLITS", ""),
        "compile_reuse",
    )


@functools.cache
def _config_is_valid(
    block_config: tuple[int, int],
    head_dim: int,
    device_index: int,
) -> bool:
    return bool(_valid_bwd_reuse_configs(block_config, head_dim, device_index))


@functools.cache
def _valid_configs(
    head_dim: int,
    device_index: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        block_config
        for block_config in _valid_forward_configs(head_dim, False, device_index)
        if _config_is_valid(block_config, head_dim, device_index)
    )


def _normalize_config(
    q: torch.Tensor,
    block_m: int,
    block_n: int,
    selected_key: object | None = None,
) -> tuple[int, int]:
    block_config = (block_m, block_n)
    block_configs = _valid_configs(q.size(-1), q.device.index)
    if min(block_config) <= 0 or block_config not in block_configs:
        if selected_key is not None:
            selected_config = _SELECTED_COMPILE_REUSE_BLOCK_CONFIGS.get(selected_key)
            if selected_config is not None:
                return selected_config
        return block_configs[0]
    return block_config


def _eager_autotune_select(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    pv_accum_dtype: str,
    smooth_k: bool,
) -> tuple[int, int]:
    from .triton_bwd_reuse import _sageattn_triton_trainable_reuse_configured

    block_configs = _valid_configs(q.size(-1), q.device.index)
    key = autotune_utils._tensor_autotune_cache_key(
        q,
        k,
        v,
        tensor_layout,
        False,
        pv_accum_dtype,
        smooth_k,
        os.environ.get("SAGEATTN_REUSE_DQ_SPLITS", ""),
        "trainable_reuse",
    )
    dout = torch.randn_like(q)

    def benchmark(block_config: tuple[int, int]) -> None:
        q_bench = q.detach().requires_grad_(True)
        k_bench = k.detach().requires_grad_(True)
        v_bench = v.detach().requires_grad_(True)
        with torch.enable_grad():
            out = _sageattn_triton_trainable_reuse_configured(
                q_bench,
                k_bench,
                v_bench,
                tensor_layout,
                pv_accum_dtype,
                smooth_k,
                block_config,
            )
            out.backward(dout)

    return autotune_utils._eager_autotune_select(
        block_configs,
        _TRITON_TRAINABLE_REUSE_AUTOTUNE_CACHE,
        key,
        benchmark,
    )


def _sageattn_triton_trainable_reuse_forward_configured(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    pv_accum_dtype: str,
    smooth_k: bool,
    block_m: int = 0,
    block_n: int = 0,
) -> torch.Tensor:
    from .triton_bwd import _trainable_forward_state

    selected_key = _compile_block_config_key(q, k, v, tensor_layout, pv_accum_dtype, smooth_k)
    block_config = _normalize_config(q, block_m, block_n, selected_key)
    out, _, _, _ = _trainable_forward_state(q, k, v, tensor_layout, pv_accum_dtype, smooth_k, block_config)
    return out


@torch.library.custom_op("sageattention_internal::sageattn_triton_trainable_reuse_autotuned", mutates_args=())
def _sageattn_triton_trainable_reuse_autotuned(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    pv_accum_dtype: str,
    smooth_k: bool,
    block_m: int = 0,
    block_n: int = 0,
) -> torch.Tensor:
    return _sageattn_triton_trainable_reuse_forward_configured(
        q,
        k,
        v,
        tensor_layout,
        pv_accum_dtype,
        smooth_k,
        block_m,
        block_n,
    )


@_sageattn_triton_trainable_reuse_autotuned.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    pv_accum_dtype: str,
    smooth_k: bool,
    block_m: int = 0,
    block_n: int = 0,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op("sageattention_internal::sageattn_triton_trainable_reuse_backward", mutates_args=())
def _sageattn_triton_trainable_reuse_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    tensor_layout: str,
    pv_accum_dtype: str,
    smooth_k: bool,
    block_m: int = 0,
    block_n: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from .triton_bwd_reuse import _trainable_reuse_backward_from_inputs

    selected_key = _compile_block_config_key(q, k, v, tensor_layout, pv_accum_dtype, smooth_k)
    block_config = _normalize_config(q, block_m, block_n, selected_key)
    return _trainable_reuse_backward_from_inputs(
        q,
        k,
        v,
        dout,
        tensor_layout,
        pv_accum_dtype,
        smooth_k,
        block_config,
    )


@_sageattn_triton_trainable_reuse_backward.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    tensor_layout: str,
    pv_accum_dtype: str,
    smooth_k: bool,
    block_m: int = 0,
    block_n: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _sageattn_triton_trainable_reuse_setup_context(ctx, inputs, output) -> None:
    q, k, v, tensor_layout, pv_accum_dtype, smooth_k, block_m, block_n = inputs
    ctx.save_for_backward(q, k, v)
    ctx.tensor_layout = tensor_layout
    ctx.pv_accum_dtype = pv_accum_dtype
    ctx.smooth_k = smooth_k
    ctx.block_config = (block_m, block_n)


def _sageattn_triton_trainable_reuse_autograd(ctx, dout: torch.Tensor):
    q, k, v = ctx.saved_tensors
    dq, dk, dv = _sageattn_triton_trainable_reuse_backward(
        q,
        k,
        v,
        dout,
        ctx.tensor_layout,
        ctx.pv_accum_dtype,
        ctx.smooth_k,
        ctx.block_config[0],
        ctx.block_config[1],
    )
    return dq, dk, dv, None, None, None, None, None


torch.library.register_autograd(
    _sageattn_triton_trainable_reuse_autotuned,
    _sageattn_triton_trainable_reuse_autograd,
    setup_context=_sageattn_triton_trainable_reuse_setup_context,
)


def _record_compile_trainable_reuse_selection(choice) -> None:
    kwargs = choice.decomposition_kwargs
    q, k, v = choice.benchmark_inputs
    key = _compile_block_config_key(q, k, v, kwargs["tensor_layout"], kwargs["pv_accum_dtype"], kwargs["smooth_k"])
    _SELECTED_COMPILE_REUSE_BLOCK_CONFIGS[key] = (kwargs["block_m"], kwargs["block_n"])


def _compile_trainable_reuse_timing_target(choice, inputs: tuple[torch.Tensor, ...], out: torch.Tensor) -> float:
    kwargs = choice.decomposition_kwargs
    dout = torch.randn_like(out)

    def fn() -> None:
        q, k, v = (tensor.detach().requires_grad_(True) for tensor in inputs)
        with torch.enable_grad():
            output = _sageattn_triton_trainable_reuse_autotuned(q, k, v, **kwargs)
            output.backward(dout)

    if config.profile_bandwidth_with_do_bench_using_profiling:
        return do_bench_using_profiling(fn)
    return benchmarker.benchmark(fn, device=benchmarker.infer_device(*inputs, out))


_enable_dynamo_backward_tracing()

register_custom_timing_target(
    _TRITON_TRAINABLE_REUSE_COMPILE_AUTOTUNE_NAME,
    _compile_trainable_reuse_timing_target,
    on_select=_record_compile_trainable_reuse_selection,
)

register_custom_op_autotuning(
    _sageattn_triton_trainable_reuse_autotuned,
    config_generator=lambda fake_tensors: [
        CustomOpConfig(
            block_m=block_config[0],
            block_n=block_config[1],
        )
        for block_config in _valid_configs(
            fake_tensors["q"].size(-1),
            fake_tensors["q"].device.index,
        )
    ],
    name=_TRITON_TRAINABLE_REUSE_COMPILE_AUTOTUNE_NAME,
)

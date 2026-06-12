import os
from collections.abc import Callable

import torch
import triton
from torch._inductor.kernel.custom_op import CustomOpConfig, register_custom_op_autotuning
from torch.fx.experimental.symbolic_shapes import optimization_hint

_SM80_QK_AUTOTUNE_CONFIGS = (
    (128, 64, 32, 64),
    (128, 32, 32, 32),
    (64, 64, 32, 64),
    (128, 64, 16, 64),
)
_SM80_QK_AUTOTUNE_CACHE = {}
_PV_ACCUM_DTYPE_TO_ID = {"fp32": 0, "fp16": 1, "fp16+fp32": 2}
_PV_ACCUM_DTYPE_FROM_ID = {value: key for key, value in _PV_ACCUM_DTYPE_TO_ID.items()}


def _padded_head_dim(head_dim: int) -> int:
    if head_dim < 64:
        return 64
    if 64 < head_dim < 128:
        return 128
    if 128 < head_dim < 256:
        return 256
    if head_dim in (64, 128, 256):
        return head_dim
    raise ValueError(f"Unsupported head_dim: {head_dim}")


def _sm80_qk_config_is_valid(
    config: tuple[int, int, int, int],
    head_dim: int,
    is_causal: bool,
    qk_quant_gran: str,
    device: torch.device,
) -> bool:
    blk_q, blk_k, warp_q, warp_k = config
    if head_dim not in (64, 128, 256):
        return False
    if blk_q % warp_q != 0 or blk_k % warp_k != 0:
        return False
    if warp_q % 16 != 0 or warp_k % 16 != 0:
        return False
    if qk_quant_gran == "per_warp" and (warp_k != blk_k or blk_k not in (64, 128)):
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


def _valid_sm80_qk_configs(
    q: torch.Tensor,
    is_causal: bool,
    qk_quant_gran: str,
) -> tuple[tuple[int, int, int, int], ...]:
    return _valid_sm80_qk_configs_for_head_dim(q.size(-1), is_causal, qk_quant_gran, q.device)


def _valid_sm80_qk_configs_for_head_dim(
    head_dim: int,
    is_causal: bool,
    qk_quant_gran: str,
    device: torch.device,
) -> tuple[tuple[int, int, int, int], ...]:
    configs = tuple(
        config
        for config in _SM80_QK_AUTOTUNE_CONFIGS
        if _sm80_qk_config_is_valid(config, head_dim, is_causal, qk_quant_gran, device)
    )
    if not configs:
        raise RuntimeError(
            f"No valid sm80 QK config for head_dim={head_dim} is_causal={is_causal} qk_quant_gran={qk_quant_gran}."
        )
    return configs


def _autotune_cache_key(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    qk_quant_gran: str,
    pv_accum_dtype: str,
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
        tensor_layout,
        is_causal,
        qk_quant_gran,
        pv_accum_dtype,
        smooth_k,
        smooth_v,
        return_lse,
    )


def _select_sm80_qk_config(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    qk_quant_gran: str,
    pv_accum_dtype: str,
    smooth_k: bool,
    smooth_v: bool,
    return_lse: bool,
    run_fn: Callable[[tuple[int, int, int, int]], object],
) -> tuple[int, int, int, int]:
    configs = _valid_sm80_qk_configs(q, is_causal, qk_quant_gran)
    if torch.compiler.is_compiling():
        return configs[0]
    if len(configs) == 1 or os.environ.get("SAGEATTN_AUTOTUNE", "1").lower() in ("0", "false", "off"):
        return configs[0]

    key = _autotune_cache_key(
        q, k, v, tensor_layout, is_causal, qk_quant_gran, pv_accum_dtype, smooth_k, smooth_v, return_lse
    )
    cached = _SM80_QK_AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached

    warmup_ms = max(1, int(os.environ.get("SAGEATTN_AUTOTUNE_WARMUP_MS", "25")))
    rep_ms = max(1, int(os.environ.get("SAGEATTN_AUTOTUNE_REP_MS", "100")))
    best_config = configs[0]
    best_ms = None

    for config in configs:
        ms = triton.testing.do_bench(lambda config=config: run_fn(config), warmup=warmup_ms, rep=rep_ms)
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best_config = config

    _SM80_QK_AUTOTUNE_CACHE[key] = best_config
    print(f"sageattention autotune key={key} config={best_config} time={best_ms:.3f} ms")
    return best_config


def _tensor_layout_from_id(tensor_layout: int) -> str:
    if tensor_layout == 0:
        return "NHD"
    if tensor_layout == 1:
        return "HND"
    raise ValueError(f"Unknown tensor layout id: {tensor_layout}")


def _qk_quant_gran_from_id(qk_quant_gran: int) -> str:
    if qk_quant_gran == 3:
        return "per_thread"
    if qk_quant_gran == 2:
        return "per_warp"
    raise ValueError(f"Unknown qk quantization granularity id: {qk_quant_gran}")


def register_sm80_autotune_op(impl_fn: Callable[..., torch.Tensor]):
    @torch.library.custom_op("sageattention_internal::sageattn_sm80_autotuned", mutates_args=())
    def _sageattn_qk_int8_pv_fp16_cuda_autotuned(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tensor_layout: int,
        is_causal: bool,
        qk_quant_gran: int,
        sm_scale: float,
        pv_accum_dtype: int,
        smooth_k: bool,
        smooth_v: bool,
        blk_q: int = 0,
        blk_k: int = 0,
        warp_q: int = 0,
        warp_k: int = 0,
    ) -> torch.Tensor:
        tensor_layout_s = _tensor_layout_from_id(tensor_layout)
        qk_quant_gran_s = _qk_quant_gran_from_id(qk_quant_gran)
        pv_accum_dtype_s = _PV_ACCUM_DTYPE_FROM_ID[pv_accum_dtype]

        head_dim = _padded_head_dim(q.size(-1))
        valid_configs = _valid_sm80_qk_configs_for_head_dim(head_dim, is_causal, qk_quant_gran_s, q.device)
        qk_config = (blk_q, blk_k, warp_q, warp_k)
        if min(qk_config) <= 0 or qk_config not in valid_configs:
            qk_config = valid_configs[0]

        return impl_fn(
            q,
            k,
            v,
            tensor_layout=tensor_layout_s,
            is_causal=is_causal,
            qk_quant_gran=qk_quant_gran_s,
            sm_scale=sm_scale,
            pv_accum_dtype=pv_accum_dtype_s,
            smooth_k=smooth_k,
            smooth_v=smooth_v,
            return_lse=False,
            qk_config=qk_config,
        )

    @_sageattn_qk_int8_pv_fp16_cuda_autotuned.register_fake
    def _(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tensor_layout: int,
        is_causal: bool,
        qk_quant_gran: int,
        sm_scale: float,
        pv_accum_dtype: int,
        smooth_k: bool,
        smooth_v: bool,
        blk_q: int = 0,
        blk_k: int = 0,
        warp_q: int = 0,
        warp_k: int = 0,
    ) -> torch.Tensor:
        return torch.empty_like(q)

    def _generate_sm80_autotune_configs(fake_tensors: dict[str, torch.Tensor]):
        q = fake_tensors["q"]
        is_causal = bool(fake_tensors.get("is_causal", False))
        qk_quant_gran = int(fake_tensors.get("qk_quant_gran", 3))
        qk_quant_gran_s = _qk_quant_gran_from_id(qk_quant_gran)
        head_dim = _padded_head_dim(optimization_hint(q.shape[-1]))
        return [
            CustomOpConfig(blk_q=cfg[0], blk_k=cfg[1], warp_q=cfg[2], warp_k=cfg[3])
            for cfg in _valid_sm80_qk_configs_for_head_dim(head_dim, is_causal, qk_quant_gran_s, q.device)
        ]

    register_custom_op_autotuning(
        _sageattn_qk_int8_pv_fp16_cuda_autotuned,
        config_generator=_generate_sm80_autotune_configs,
    )
    return _sageattn_qk_int8_pv_fp16_cuda_autotuned

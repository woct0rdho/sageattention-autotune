import os

import torch
import torch.nn.functional as F

from .triton.attn_bwd_qk_int8_fused import backward_fused as _attn_backward_fused
from .triton_bwd import _from_nhd, _to_nhd, _trainable_forward_state
from .triton_bwd_fused_autotune import _eager_autotune_select, _sageattn_triton_trainable_fused_autotuned
from .utils import DEFAULT_PV_ACCUM_DTYPE


def _fixed_fused_block_config() -> tuple[int, int] | None:
    override = os.environ.get("SAGEATTN_FUSED_BLOCK")
    if override is None:
        return None

    block_parts = override.lower().replace("x", ",").split(",")
    if len(block_parts) != 2:
        raise ValueError("SAGEATTN_FUSED_BLOCK must have the form 'BLOCK_M,BLOCK_N', for example '64,128'.")

    block_m, block_n = (int(part.strip()) for part in block_parts)
    if block_m <= 0 or block_n <= 0:
        raise ValueError("SAGEATTN_FUSED_BLOCK values must be positive integers.")
    return block_m, block_n


def _trainable_fused_backward_from_state(
    dout: torch.Tensor,
    saved_tensors: tuple[torch.Tensor, ...],
    has_k_mean: bool,
    tensor_layout: str,
    head_dim: int,
    block_config: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_int8, k_int8, v_nhd, out_nhd, lse, q_scale, k_scale = saved_tensors[:7]
    k_mean = saved_tensors[7] if has_k_mean else None

    dout_nhd = _to_nhd(dout, tensor_layout)
    if dout_nhd.size(-1) != v_nhd.size(-1):
        dout_nhd = F.pad(dout_nhd, (0, v_nhd.size(-1) - dout_nhd.size(-1)))

    block_m, block_n = block_config
    dq, dk, dv = _attn_backward_fused(
        q_int8,
        k_int8,
        v_nhd,
        dout_nhd.to(torch.float16).contiguous(),
        out_nhd,
        lse,
        q_scale,
        k_scale,
        k_mean,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )

    dq = _from_nhd(dq[..., :head_dim], tensor_layout)
    dk = _from_nhd(dk[..., :head_dim], tensor_layout)
    dv = _from_nhd(dv[..., :head_dim], tensor_layout)
    return dq, dk, dv


def _trainable_fused_backward_from_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    tensor_layout: str,
    pv_accum_dtype: str,
    smooth_k: bool,
    block_config: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, saved_tensors, has_k_mean, head_dim = _trainable_forward_state(
        q,
        k,
        v,
        tensor_layout,
        pv_accum_dtype,
        smooth_k,
        block_config,
    )
    return _trainable_fused_backward_from_state(
        dout,
        tuple(saved_tensors),
        has_k_mean,
        tensor_layout,
        head_dim,
        block_config,
    )


class _SageAttnTritonTrainableFused(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tensor_layout: str,
        pv_accum_dtype: str,
        smooth_k: bool,
        block_m: int,
        block_n: int,
    ) -> torch.Tensor:
        block_config = (block_m, block_n)
        out, tensors_to_save, has_k_mean, head_dim = _trainable_forward_state(
            q,
            k,
            v,
            tensor_layout,
            pv_accum_dtype,
            smooth_k,
            block_config,
        )

        ctx.save_for_backward(*tensors_to_save)
        ctx.has_k_mean = has_k_mean
        ctx.tensor_layout = tensor_layout
        ctx.head_dim = head_dim
        ctx.block_config = block_config
        return out

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        (dout,) = grad_outputs
        dq, dk, dv = _trainable_fused_backward_from_state(
            dout,
            ctx.saved_tensors,
            ctx.has_k_mean,
            ctx.tensor_layout,
            ctx.head_dim,
            ctx.block_config,
        )
        return dq, dk, dv, None, None, None, None, None


def _sageattn_triton_trainable_fused_configured(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    pv_accum_dtype: str,
    smooth_k: bool,
    block_config: tuple[int, int],
) -> torch.Tensor:
    if pv_accum_dtype not in ("fp32", "fp16"):
        raise ValueError("pv_accum_dtype must be 'fp32' or 'fp16'.")
    if not q.is_cuda:
        raise ValueError("Input tensors must be CUDA tensors.")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise ValueError("fused SageBwd Triton trainable path supports torch.float16 only.")
    if q.device != k.device or q.device != v.device:
        raise ValueError("All tensors must be on the same device.")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("All tensors must have the same dtype.")

    return _SageAttnTritonTrainableFused.apply(
        q,
        k,
        v,
        tensor_layout,
        pv_accum_dtype,
        smooth_k,
        block_config[0],
        block_config[1],
    )


def sageattn_qk_int8_pv_fp16_triton_trainable_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
    smooth_k: bool = True,
    return_lse: bool = False,
) -> torch.Tensor:
    if is_causal:
        raise NotImplementedError("SageBwd Triton fused trainable path currently supports non-causal attention only.")
    if return_lse:
        raise NotImplementedError("return_lse is not exposed by the fused trainable SageBwd path.")
    if torch.compiler.is_compiling():
        return _sageattn_triton_trainable_fused_autotuned(
            q,
            k,
            v,
            tensor_layout,
            pv_accum_dtype,
            smooth_k,
        )

    block_config = _fixed_fused_block_config()
    if block_config is None:
        block_config = _eager_autotune_select(q, k, v, tensor_layout, pv_accum_dtype, smooth_k)

    return _sageattn_triton_trainable_fused_configured(
        q,
        k,
        v,
        tensor_layout,
        pv_accum_dtype,
        smooth_k,
        block_config,
    )

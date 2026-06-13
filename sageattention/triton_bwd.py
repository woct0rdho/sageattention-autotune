import torch
import torch.nn.functional as F

from .triton.attn_bwd_qk_int8 import backward as _attn_backward
from .triton.attn_qk_int8_per_block import forward as _attn_forward
from .triton.quant_per_block import per_block_int8
from .triton_bwd_autotune import _eager_autotune_select, _sageattn_triton_trainable_autotuned
from .utils import DEFAULT_PV_ACCUM_DTYPE, _pad_qkv


def _to_nhd(x: torch.Tensor, tensor_layout: str) -> torch.Tensor:
    if tensor_layout == "NHD":
        return x.contiguous()
    if tensor_layout == "HND":
        return x.transpose(1, 2).contiguous()
    raise ValueError("tensor_layout must be 'NHD' or 'HND'.")


def _from_nhd(x: torch.Tensor, tensor_layout: str) -> torch.Tensor:
    if tensor_layout == "NHD":
        return x
    if tensor_layout == "HND":
        return x.transpose(1, 2).contiguous()
    raise ValueError("tensor_layout must be 'NHD' or 'HND'.")


def _trainable_forward_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    pv_accum_dtype: str,
    smooth_k: bool,
    block_config: tuple[int, int],
) -> tuple[torch.Tensor, list[torch.Tensor], bool, int]:
    q_nhd = _to_nhd(q, tensor_layout)
    k_nhd = _to_nhd(k, tensor_layout)
    v_nhd = _to_nhd(v, tensor_layout)

    head_dim, q_nhd, k_nhd, v_nhd = _pad_qkv(q_nhd, k_nhd, v_nhd)
    if q_nhd.shape != k_nhd.shape or q_nhd.shape != v_nhd.shape:
        raise ValueError("initial trainable SageAttention Triton path requires q, k, and v to have the same shape.")
    if q_nhd.dtype != torch.float16:
        raise ValueError("initial trainable SageAttention Triton path supports torch.float16 only.")
    if q_nhd.stride(-1) != 1 or k_nhd.stride(-1) != 1 or v_nhd.stride(-1) != 1:
        raise ValueError("Last dimension of q, k, and v must be contiguous.")

    if smooth_k:
        k_mean_keepdim = k_nhd.mean(dim=1, keepdim=True)
        k_mean = k_mean_keepdim.squeeze(1).contiguous()
    else:
        k_mean_keepdim = None
        k_mean = None

    block_m, block_n = block_config
    q_int8, q_scale, k_int8, k_scale = per_block_int8(
        q_nhd,
        k_nhd,
        km=k_mean_keepdim,
        BLKQ=block_m,
        BLKK=block_n,
        tensor_layout="NHD",
    )
    out_nhd, lse = _attn_forward(
        q_int8,
        k_int8,
        v_nhd,
        q_scale,
        k_scale,
        tensor_layout="NHD",
        is_causal=False,
        pv_accum_dtype=pv_accum_dtype,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        output_dtype=q_nhd.dtype,
        return_lse=True,
    )

    tensors_to_save = [q_int8, k_int8, v_nhd, out_nhd, lse, q_scale, k_scale]
    if k_mean is not None:
        tensors_to_save.append(k_mean)

    out = out_nhd[..., :head_dim]
    return _from_nhd(out, tensor_layout), tensors_to_save, k_mean is not None, head_dim


def _trainable_backward_from_state(
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
    dq, dk, dv = _attn_backward(
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


def _trainable_backward_from_inputs(
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
    return _trainable_backward_from_state(
        dout,
        tuple(saved_tensors),
        has_k_mean,
        tensor_layout,
        head_dim,
        block_config,
    )


class _SageAttnTritonTrainable(torch.autograd.Function):
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
        dq, dk, dv = _trainable_backward_from_state(
            dout,
            ctx.saved_tensors,
            ctx.has_k_mean,
            ctx.tensor_layout,
            ctx.head_dim,
            ctx.block_config,
        )
        return dq, dk, dv, None, None, None, None, None


def _sageattn_triton_trainable_configured(
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
        raise ValueError("initial trainable SageAttention Triton path supports torch.float16 only.")
    if q.device != k.device or q.device != v.device:
        raise ValueError("All tensors must be on the same device.")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("All tensors must have the same dtype.")

    return _SageAttnTritonTrainable.apply(
        q,
        k,
        v,
        tensor_layout,
        pv_accum_dtype,
        smooth_k,
        block_config[0],
        block_config[1],
    )


def sageattn_qk_int8_pv_fp16_triton_trainable(
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
        raise NotImplementedError("SageBwd Triton trainable path currently supports non-causal attention only.")
    if return_lse:
        raise NotImplementedError("return_lse is not exposed by the initial trainable SageBwd path.")
    if torch.compiler.is_compiling():
        return _sageattn_triton_trainable_autotuned(
            q,
            k,
            v,
            tensor_layout,
            pv_accum_dtype,
            smooth_k,
        )

    block_config = _eager_autotune_select(q, k, v, tensor_layout, pv_accum_dtype, smooth_k)

    return _sageattn_triton_trainable_configured(
        q,
        k,
        v,
        tensor_layout,
        pv_accum_dtype,
        smooth_k,
        block_config,
    )

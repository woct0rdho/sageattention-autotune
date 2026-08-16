import pytest
import torch
import torch.nn.functional as F

from sageattention.triton.attn_bwd_autotune import _has_valid_bwd_configs
from sageattention.triton.attn_bwd_qk_int8 import backward as _attn_backward
from sageattention.triton.attn_qk_int8_per_block import forward as _attn_forward
from sageattention.triton.quant_per_block import per_block_int8
from sageattention.triton_autotune import _valid_configs as _valid_forward_configs

PreparedBackwardInputs = tuple[torch.Tensor, ...]


def _make_qkvo(
    batch_size: int = 2,
    seq_len: int = 1024,
    num_heads: int = 16,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q)
    return q, k, v, dout


def _make_valid_configs() -> tuple[tuple[int, int], ...]:
    device_index = torch.cuda.current_device()
    return tuple(
        block_config
        for block_config in _valid_forward_configs(64, False, device_index)
        if _has_valid_bwd_configs(block_config, 64, device_index)
    )


def _metric(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual = actual.float()
    expected = expected.float()
    diff = actual - expected

    # The SageBwd paper reports CosSim and Rel-L2 without a special 4D tensor convention.
    # We use the standard flattened cosine similarity and Frobenius relative error.
    cos_sim = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    fro_rel_err = (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected).clamp(min=1.0e-6)).item()
    return cos_sim, fro_rel_err


def _check(actual: torch.Tensor, expected: torch.Tensor, name: str, cos_tol: float, rtol: float) -> None:
    cos_sim, fro_rel_err = _metric(actual, expected)
    msg = f"{name}: cos_sim={cos_sim:.3g} fro_rel_err={fro_rel_err:.3g}"
    assert 1 - cos_sim < cos_tol, msg
    assert fro_rel_err < rtol, msg


def _flash_attn_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flash_attn = pytest.importorskip("flash_attn", reason="flash_attn is not installed")
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    out = flash_attn.flash_attn_func(
        q,
        k,
        v,
        dropout_p=0,
        softmax_scale=q.size(-1) ** -0.5,
        causal=False,
    )
    out.backward(dout)

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    return q.grad, k.grad, v.grad


def _prepare_sage_backward_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    block_config: tuple[int, int],
) -> PreparedBackwardInputs:
    block_m, block_n = block_config
    k_mean_keepdim = k.mean(dim=1, keepdim=True)
    k_mean = k_mean_keepdim.squeeze(1).contiguous()
    q_int8, q_scale, k_int8, k_scale = per_block_int8(
        q,
        k,
        km=k_mean_keepdim,
        BLKQ=block_m,
        BLKK=block_n,
        tensor_layout="NHD",
    )
    out = torch.empty_like(v)
    lse = torch.empty((q.size(0), q.size(2), q.size(1)), device=q.device, dtype=torch.float32)
    _attn_forward(
        q_int8,
        k_int8,
        v,
        q_scale,
        k_scale,
        out,
        lse,
        tensor_layout="NHD",
        is_causal=False,
        pv_accum_dtype="fp32",
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        return_lse=True,
    )
    return q_int8, k_int8, v, dout, out, lse, q_scale, k_scale, k_mean


def _sage_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    block_config: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_int8, k_int8, v, dout, out, lse, q_scale, k_scale, k_mean = _prepare_sage_backward_inputs(
        q, k, v, dout, block_config
    )
    dq, dk, dv = _attn_backward(
        q_int8,
        k_int8,
        v,
        dout,
        out,
        lse,
        q_scale,
        k_scale,
        k_mean,
        BLOCK_M=block_config[0],
        BLOCK_N=block_config[1],
    )

    return dq, dk, dv


def _check_backward(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    name: str,
) -> None:
    dq, dk, dv = actual
    dq_ref, dk_ref, dv_ref = expected
    _check(dq, dq_ref, f"{name} dQ", 0.003, 0.07)
    _check(dk, dk_ref, f"{name} dK", 0.003, 0.07)
    _check(dv, dv_ref, f"{name} dV", 0.002, 0.06)


@pytest.mark.parametrize("block_config", _make_valid_configs(), ids=str)
def test_qattn_triton_backward_block_config(block_config: tuple[int, int]) -> None:
    q, k, v, dout = _make_qkvo()
    actual = _sage_backward(q, k, v, dout, block_config)
    expected = _flash_attn_backward(q, k, v, dout)
    _check_backward(actual, expected, f"block_config={block_config}")

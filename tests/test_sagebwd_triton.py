import math

import pytest
import torch
import torch.nn.functional as F

from sageattention.triton_bwd import _sageattn_triton_trainable_configured
from sageattention.triton_bwd_autotune import _valid_configs


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
    return _valid_configs(head_dim=64, device_index=torch.cuda.current_device())


def _metric(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual = actual.float()
    expected = expected.float()
    diff = actual - expected

    # The SageBwd paper reports CosSim and Rel-L2 without a special 4D tensor convention.
    # We use the standard flattened cosine similarity and Frobenius relative error.
    cos_sim = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    fro_rel_err = (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected).clamp(min=1.0e-6)).item()
    return cos_sim, fro_rel_err


def _check(actual: torch.Tensor, expected: torch.Tensor, name: str, cos_threshold: float, rel_threshold: float) -> None:
    cos_sim, fro_rel_err = _metric(actual, expected)
    msg = f"{name}: cos_sim={cos_sim:.3g} fro_rel_err={fro_rel_err:.3g}"
    assert cos_sim > cos_threshold, msg
    assert fro_rel_err < rel_threshold, msg


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
        dropout_p=0.0,
        causal=False,
        softmax_scale=1 / math.sqrt(q.size(-1)),
    )
    out.backward(dout)

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    return q.grad, k.grad, v.grad


def _sage_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    block_config: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    out = _sageattn_triton_trainable_configured(
        q,
        k,
        v,
        tensor_layout="NHD",
        pv_accum_dtype="fp32",
        smooth_k=True,
        block_config=block_config,
    )
    out.backward(dout)

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    return q.grad, k.grad, v.grad


def _check_backward(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    name: str,
) -> None:
    dq, dk, dv = actual
    dq_ref, dk_ref, dv_ref = expected
    _check(dq, dq_ref, f"{name} dQ", 0.995, 0.06)
    _check(dk, dk_ref, f"{name} dK", 0.995, 0.06)
    _check(dv, dv_ref, f"{name} dV", 0.998, 0.05)


@pytest.mark.parametrize("block_config", _make_valid_configs(), ids=str)
def test_sagebwd_triton_block_config(block_config: tuple[int, int]) -> None:
    q, k, v, dout = _make_qkvo()
    actual = _sage_backward(q, k, v, dout, block_config)
    expected = _flash_attn_backward(q, k, v, dout)
    _check_backward(actual, expected, f"SageBwd Triton block_config={block_config}")

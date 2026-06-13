import torch
from test_sagebwd_triton import _check_backward, _flash_attn_backward, _make_qkvo

from sageattention import sageattn_qk_int8_pv_fp16_triton_trainable_fused


def _eager_autotuned_fused_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    out = sageattn_qk_int8_pv_fp16_triton_trainable_fused(
        q,
        k,
        v,
        tensor_layout="NHD",
        is_causal=False,
        pv_accum_dtype="fp32",
        smooth_k=True,
    )
    out.backward(dout)

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    return q.grad, k.grad, v.grad


def _compile_autotuned_fused_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)

    @torch.compile(fullgraph=True, mode="max-autotune")
    def fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dout: torch.Tensor) -> torch.Tensor:
        out = sageattn_qk_int8_pv_fp16_triton_trainable_fused(
            q,
            k,
            v,
            tensor_layout="NHD",
            is_causal=False,
            pv_accum_dtype="fp32",
            smooth_k=True,
        )
        out.backward(dout)
        return out.detach()

    out = fn(q, k, v, dout)

    assert torch.isfinite(out).all()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    return q.grad, k.grad, v.grad


def test_eager_autotuned_fused() -> None:
    q, k, v, dout = _make_qkvo()
    expected = _flash_attn_backward(q, k, v, dout)
    actual = _eager_autotuned_fused_backward(q, k, v, dout)
    _check_backward(actual, expected, "eager autotuned")


def test_compile_autotuned_fused() -> None:
    q, k, v, dout = _make_qkvo()
    expected = _flash_attn_backward(q, k, v, dout)
    actual = _compile_autotuned_fused_backward(q, k, v, dout)
    _check_backward(actual, expected, "compile autotuned")

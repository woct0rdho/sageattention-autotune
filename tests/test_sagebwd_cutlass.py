import pytest
import torch
import torch.nn.functional as F

from sageattention.cutlass_bwd import _BWD_CONFIG, _BWD_CONFIGS, _sageattn_cutlass_bwd_configured

_BWD_TEST_CANDIDATES = tuple((config, "dynamic") for config in _BWD_CONFIGS) + (
    (_BWD_CONFIG, "power2_ds"),
    (_BWD_CONFIG, "periodic_ds"),
)


def _make_qkvo(
    batch_size: int = 1,
    seq_len: int = 64,
    num_heads: int = 2,
    head_dim: int = 64,
    tensor_layout: str = "NHD",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if tensor_layout == "NHD":
        shape = (batch_size, seq_len, num_heads, head_dim)
    else:
        shape = (batch_size, num_heads, seq_len, head_dim)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q)
    return q, k, v, dout


def _to_nhd(x: torch.Tensor, tensor_layout: str) -> torch.Tensor:
    return x.transpose(1, 2).contiguous() if tensor_layout == "HND" else x


def _from_nhd(x: torch.Tensor, tensor_layout: str) -> torch.Tensor:
    return x.transpose(1, 2).contiguous() if tensor_layout == "HND" else x


def _flash_forward_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    tensor_layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flash_attn = pytest.importorskip("flash_attn", reason="flash_attn is not installed")
    q_nhd = _to_nhd(q, tensor_layout).detach().clone().requires_grad_(True)
    k_nhd = _to_nhd(k, tensor_layout).detach().clone().requires_grad_(True)
    v_nhd = _to_nhd(v, tensor_layout).detach().clone().requires_grad_(True)
    dout_nhd = _to_nhd(dout, tensor_layout)
    out_nhd, lse, _ = flash_attn.flash_attn_func(
        q_nhd,
        k_nhd,
        v_nhd,
        dropout_p=0.0,
        softmax_scale=q.size(-1) ** -0.5,
        causal=False,
        return_attn_probs=True,
    )
    out_nhd.backward(dout_nhd)

    assert q_nhd.grad is not None
    assert k_nhd.grad is not None
    assert v_nhd.grad is not None
    return (
        _from_nhd(out_nhd.detach(), tensor_layout),
        lse.detach(),
        _from_nhd(q_nhd.grad, tensor_layout),
        _from_nhd(k_nhd.grad, tensor_layout),
        _from_nhd(v_nhd.grad, tensor_layout),
    )


def _metric(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    actual = actual.float()
    expected = expected.float()
    diff = actual - expected
    cos_sim = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    fro_rel_err = (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected).clamp(min=1e-6)).item()
    max_abs_err = diff.abs().max().item()
    return cos_sim, fro_rel_err, max_abs_err


def _check_close(actual: torch.Tensor, expected: torch.Tensor, name: str, *, format_experiment: bool) -> None:
    cos_sim, fro_rel_err, max_abs_err = _metric(actual, expected)
    msg = f"{name}: cos_sim={cos_sim:.5f} fro_rel_err={fro_rel_err:.3g} max_abs_err={max_abs_err:.3g}"
    assert 1 - cos_sim < (1e-2 if format_experiment else 2e-3), msg
    assert fro_rel_err < (1.5e-1 if format_experiment else 6e-2), msg
    assert max_abs_err < (5e-1 if format_experiment else 2e-1), msg


@pytest.mark.parametrize(("config", "quantization_policy"), _BWD_TEST_CANDIDATES)
@pytest.mark.parametrize("tensor_layout", ("NHD", "HND"))
@pytest.mark.parametrize("head_dim", (64,))
@pytest.mark.parametrize("seq_len", (64, 65))
def test_sagebwd_cutlass_config_matches_flashattention(
    config: tuple[int, int, int, int],
    quantization_policy: str,
    tensor_layout: str,
    head_dim: int,
    seq_len: int,
) -> None:
    q, k, v, dout = _make_qkvo(seq_len=seq_len, head_dim=head_dim, tensor_layout=tensor_layout)
    out, lse, dq_ref, dk_ref, dv_ref = _flash_forward_backward(q, k, v, dout, tensor_layout)
    dq, dk, dv = _sageattn_cutlass_bwd_configured(q, k, v, out, dout, lse, tensor_layout, config, quantization_policy)
    format_experiment = quantization_policy != "dynamic"
    _check_close(dq, dq_ref, "dQ", format_experiment=format_experiment)
    _check_close(dk, dk_ref, "dK", format_experiment=format_experiment)
    _check_close(dv, dv_ref, "dV", format_experiment=format_experiment)

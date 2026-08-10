import pytest
import torch
import torch.nn.functional as F

from sageattention.cutlass_bwd import (
    _BWD_CONFIGS_BY_HEAD_DIM,
    _sageattn_cutlass_bwd_configured,
    sageattn_cutlass_bwd,
)
from sageattention.triton.cutlass_bwd import convert_dq

_BWD_TEST_CANDIDATES = tuple(
    (head_dim, config) for head_dim, configs in _BWD_CONFIGS_BY_HEAD_DIM.items() for config in configs
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
    generator = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(shape, device="cuda", dtype=torch.float16, generator=generator)
    k = torch.randn(shape, device="cuda", dtype=torch.float16, generator=generator)
    v = torch.randn(shape, device="cuda", dtype=torch.float16, generator=generator)
    dout = torch.randn(shape, device="cuda", dtype=torch.float16, generator=generator)
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


@pytest.mark.parametrize("tensor_layout", ("NHD", "HND"))
@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("smooth_k", (False, True))
def test_cutlass_dq_workspace_conversion(tensor_layout: str, head_dim: int, smooth_k: bool) -> None:
    batch, heads, seq_len = 2, 3, 65
    padded_seq_len = ((seq_len + 31) // 32) * 32
    dq_accum = (
        torch.arange(batch * heads * padded_seq_len * head_dim, device="cuda", dtype=torch.float32) % 1000
    ).reshape(batch, heads, padded_seq_len, head_dim)
    if tensor_layout == "NHD":
        grad_query = torch.empty((batch, seq_len, heads, head_dim), device="cuda", dtype=torch.float16)
    else:
        grad_query = torch.empty((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)

    rows = torch.arange(seq_len, device="cuda")[:, None]
    cols = torch.arange(head_dim, device="cuda")[None, :]
    lane = (rows & 7) * 4 + ((cols & 7) >> 1)
    value = (cols & 1) | (((rows >> 3) & 1) << 1) | (((cols >> 3) & 1) << 2)
    physical = (rows >> 4) * (16 * head_dim) + (cols >> 4) * 256 + lane + 32 * value
    expected_nhd = (
        dq_accum.reshape(batch, heads, -1)
        .index_select(2, physical.reshape(-1))
        .reshape(batch, heads, seq_len, head_dim)
    )
    if smooth_k:
        ds_sum = 0.001 * torch.arange(batch * heads * padded_seq_len, device="cuda", dtype=torch.float32).reshape(
            batch, heads, padded_seq_len
        )
        if tensor_layout == "NHD":
            k_mean = 0.01 * torch.arange(batch * heads * head_dim, device="cuda", dtype=torch.float16).reshape(
                batch, 1, heads, head_dim
            )
            k_mean_nhd = k_mean[:, 0]
        else:
            k_mean = 0.01 * torch.arange(batch * heads * head_dim, device="cuda", dtype=torch.float16).reshape(
                batch, heads, 1, head_dim
            )
            k_mean_nhd = k_mean[:, :, 0]
        expected_nhd += ds_sum[:, :, :seq_len, None] * k_mean_nhd[:, :, None, :]
    else:
        ds_sum = None
        k_mean = None
    expected = expected_nhd.transpose(1, 2) if tensor_layout == "NHD" else expected_nhd

    convert_dq(dq_accum, grad_query, tensor_layout, ds_sum, k_mean)
    torch.testing.assert_close(grad_query, expected.to(torch.float16), rtol=0, atol=0)


@pytest.mark.parametrize(("head_dim", "config"), _BWD_TEST_CANDIDATES)
@pytest.mark.parametrize("tensor_layout", ("NHD", "HND"))
@pytest.mark.parametrize("seq_len", (64, 65))
@pytest.mark.parametrize("smooth_k", (False, True))
def test_sagebwd_cutlass_config_matches_flashattention(
    config: tuple[int, int, int, int],
    tensor_layout: str,
    head_dim: int,
    seq_len: int,
    smooth_k: bool,
) -> None:
    q, k, v, dout = _make_qkvo(seq_len=seq_len, head_dim=head_dim, tensor_layout=tensor_layout)
    out, lse, dq_ref, dk_ref, dv_ref = _flash_forward_backward(q, k, v, dout, tensor_layout)
    dq, dk, dv = _sageattn_cutlass_bwd_configured(q, k, v, out, dout, lse, tensor_layout, smooth_k, config)
    _check_close(dq, dq_ref, "dQ", format_experiment=True)
    _check_close(dk, dk_ref, "dK", format_experiment=True)
    _check_close(dv, dv_ref, "dV", format_experiment=True)


@pytest.mark.parametrize("tensor_layout", ("NHD", "HND"))
@pytest.mark.parametrize("head_dim", (64, 128))
def test_sagebwd_cutlass_public_default_selects_head_dim_config(tensor_layout: str, head_dim: int) -> None:
    q, k, v, dout = _make_qkvo(seq_len=65, head_dim=head_dim, tensor_layout=tensor_layout)
    out, lse, *_ = _flash_forward_backward(q, k, v, dout, tensor_layout)
    expected = _sageattn_cutlass_bwd_configured(
        q,
        k,
        v,
        out,
        dout,
        lse,
        tensor_layout,
        True,
        _BWD_CONFIGS_BY_HEAD_DIM[head_dim][0],
    )
    actual = sageattn_cutlass_bwd(q, k, v, out, dout, lse, tensor_layout)
    for index, (actual_grad, expected_grad) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=1e-6 if index == 0 else 0)


@pytest.mark.parametrize("tensor_layout", ("NHD", "HND"))
@pytest.mark.parametrize("seq_len", (256, 257, 512, 513))
@pytest.mark.parametrize("smooth_k", (False, True))
def test_sagebwd_cutlass_n256_matches_n128(tensor_layout: str, seq_len: int, smooth_k: bool) -> None:
    q, k, v, dout = _make_qkvo(seq_len=seq_len, tensor_layout=tensor_layout)
    out, lse, *_ = _flash_forward_backward(q, k, v, dout, tensor_layout)
    n128 = _sageattn_cutlass_bwd_configured(q, k, v, out, dout, lse, tensor_layout, smooth_k, (32, 64, 64, 128))
    n256 = _sageattn_cutlass_bwd_configured(q, k, v, out, dout, lse, tensor_layout, smooth_k, (32, 64, 64, 256))

    torch.testing.assert_close(n256[0], n128[0], rtol=0, atol=2e-4)
    torch.testing.assert_close(n256[1], n128[1], rtol=0, atol=0)
    torch.testing.assert_close(n256[2], n128[2], rtol=0, atol=0)

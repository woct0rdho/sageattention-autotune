import pytest
import torch
from test_sagebwd_triton import _check_backward, _flash_attn_backward, _make_qkvo

from sageattention.triton_bwd_reuse import _sageattn_triton_trainable_reuse_configured
from sageattention.triton_bwd_reuse_autotune import _valid_configs


def _make_valid_configs() -> tuple[tuple[int, int], ...]:
    return _valid_configs(head_dim=64, device_index=torch.cuda.current_device())


def _sage_reuse_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    block_config: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    out = _sageattn_triton_trainable_reuse_configured(
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


@pytest.mark.parametrize("block_config", _make_valid_configs(), ids=str)
def test_sagebwd_triton_reuse_block_config(block_config: tuple[int, int]) -> None:
    q, k, v, dout = _make_qkvo()
    actual = _sage_reuse_backward(q, k, v, dout, block_config)
    expected = _flash_attn_backward(q, k, v, dout)
    _check_backward(actual, expected, f"block_config={block_config}")


def test_sagebwd_triton_reuse_flashattn_tile() -> None:
    block_config = (64, 128)
    q, k, v, dout = _make_qkvo()
    actual = _sage_reuse_backward(q, k, v, dout, block_config)
    expected = _flash_attn_backward(q, k, v, dout)
    _check_backward(actual, expected, f"block_config={block_config}")


def test_sagebwd_triton_reuse_split_dq_accum(monkeypatch: pytest.MonkeyPatch) -> None:
    block_config = _make_valid_configs()[0]
    monkeypatch.setenv("SAGEATTN_REUSE_DQ_SPLITS", "1024")
    q, k, v, dout = _make_qkvo()
    actual = _sage_reuse_backward(q, k, v, dout, block_config)
    expected = _flash_attn_backward(q, k, v, dout)
    _check_backward(actual, expected, f"block_config={block_config} split_dq_accum")

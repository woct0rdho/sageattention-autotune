import pytest
import torch
from test_sagebwd_triton import _check_backward, _flash_attn_backward, _make_qkvo, _prepare_sage_backward_inputs

from sageattention.triton.attn_bwd_autotune import _valid_bwd_fused_configs
from sageattention.triton.attn_bwd_qk_int8_fused import backward_fused as _attn_backward_fused
from sageattention.triton_autotune import _valid_configs as _valid_forward_configs


def _make_valid_configs() -> tuple[tuple[int, int], ...]:
    device_index = torch.cuda.current_device()
    return tuple(
        block_config
        for block_config in _valid_forward_configs(64, False, device_index)
        if _valid_bwd_fused_configs(block_config, 64, device_index)
    )


def _sage_fused_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    block_config: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_int8, k_int8, v, dout, out, lse, q_scale, k_scale, k_mean = _prepare_sage_backward_inputs(
        q, k, v, dout, block_config
    )
    return _attn_backward_fused(
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


@pytest.mark.parametrize("block_config", _make_valid_configs(), ids=str)
def test_qattn_triton_fused_backward_block_config(block_config: tuple[int, int]) -> None:
    q, k, v, dout = _make_qkvo()
    actual = _sage_fused_backward(q, k, v, dout, block_config)
    expected = _flash_attn_backward(q, k, v, dout)
    _check_backward(actual, expected, f"block_config={block_config}")


def test_qattn_triton_fused_backward_flashattn_tile() -> None:
    block_config = (64, 128)
    q, k, v, dout = _make_qkvo()
    actual = _sage_fused_backward(q, k, v, dout, block_config)
    expected = _flash_attn_backward(q, k, v, dout)
    _check_backward(actual, expected, f"block_config={block_config}")


def test_qattn_triton_fused_backward_split_dq_accum(monkeypatch: pytest.MonkeyPatch) -> None:
    block_config = _make_valid_configs()[0]
    monkeypatch.setenv("SAGEATTN_FUSED_DQ_SPLITS", "1024")
    q, k, v, dout = _make_qkvo()
    actual = _sage_fused_backward(q, k, v, dout, block_config)
    expected = _flash_attn_backward(q, k, v, dout)
    _check_backward(actual, expected, f"block_config={block_config} split_dq_accum")

from itertools import product

import pytest
import torch
from test_sageattn import _error_report, _expected, _make_qkv

from sageattention.triton_attn import _sageattn_triton_configured
from sageattention.triton_autotune import _valid_triton_block_configs

_MODES = tuple(
    product(
        (64, 128, 256),
        (torch.float16, torch.bfloat16),
        ("HND", "NHD"),
        (False, True),
        ("fp32", "fp16"),
        (False, True),
    )
)


def _valid_block_configs() -> tuple[tuple[int, int], ...]:
    device_index = torch.cuda.current_device()
    block_configs = []
    for head_dim, _, _, is_causal, _, _ in _MODES:
        for block_config in _valid_triton_block_configs(head_dim, is_causal, device_index):
            if block_config not in block_configs:
                block_configs.append(block_config)
    return tuple(block_configs)


def _run_case(
    block_config: tuple[int, int],
    *,
    head_dim: int,
    dtype: torch.dtype,
    tensor_layout: str,
    is_causal: bool,
    pv_accum_dtype: str,
    smooth_k: bool,
) -> tuple[bool, str]:
    q, k, v = _make_qkv(head_dim=head_dim, tensor_layout=tensor_layout, dtype=dtype)
    expected = _expected(q, k, v, tensor_layout, is_causal)

    actual = _sageattn_triton_configured(
        q,
        k,
        v,
        tensor_layout,
        is_causal,
        pv_accum_dtype,
        smooth_k,
        False,
        block_config,
    )

    return _error_report(actual, expected)


@pytest.mark.parametrize("block_config", _valid_block_configs(), ids=str)
def test_sageattn_triton_block_config(block_config: tuple[int, int]) -> None:
    config_errors = []
    config_tested = 0
    device_index = torch.cuda.current_device()

    for head_dim, dtype, tensor_layout, is_causal, pv_accum_dtype, smooth_k in _MODES:
        if block_config not in _valid_triton_block_configs(head_dim, is_causal, device_index):
            continue

        config_tested += 1
        name = (
            f"head_dim={head_dim} dtype={dtype} layout={tensor_layout} is_causal={is_causal} "
            f"pv_accum_dtype={pv_accum_dtype} smooth_k={smooth_k}"
        )
        try:
            passed, msg = _run_case(
                block_config,
                head_dim=head_dim,
                dtype=dtype,
                tensor_layout=tensor_layout,
                is_causal=is_causal,
                pv_accum_dtype=pv_accum_dtype,
                smooth_k=smooth_k,
            )
        except Exception as e:
            passed = False
            msg = f"error={e}"

        if not passed:
            config_errors.append(f"{name}: {msg}")

    assert config_tested > 0
    assert not config_errors, "\n".join(config_errors)

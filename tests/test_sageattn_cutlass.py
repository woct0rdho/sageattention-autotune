from itertools import product

import pytest
import torch
from test_sageattn import _attention_report, _expected, _make_qkv, _mode_id

from sageattention.cutlass_autotune import _AUTOTUNE_CONFIGS, _valid_configs

_MODES = tuple(
    product(
        (64, 128),
        (torch.float16,),
        ("HND", "NHD"),
        (False,),
        ("fp32",),
        (False, True),
        (False, True),
    )
)

_TEST_CASES = (
    pytest.param((2, 16, 1024), (0.013, 0.08, 0.0003, 0.05), id="aligned"),
    pytest.param((1, 2, 129), (0.04, 0.20, 0.003, 0.20), id="tail"),
)


def _valid_cases():
    device_index = torch.cuda.current_device()
    cases = []
    for config in _AUTOTUNE_CONFIGS:
        for mode in _MODES:
            head_dim, _, _, is_causal, _, _, _ = mode
            if config in _valid_configs(head_dim, is_causal, device_index):
                cases.append(pytest.param(config, mode, id=f"config={config}-{_mode_id(mode)}"))
    return tuple(cases)


def _run_case(
    config: tuple[int, int, int, int],
    head_dim: int,
    dtype: torch.dtype,
    tensor_layout: str,
    is_causal: bool,
    pv_accum_dtype: str,
    smooth_k: bool,
    return_lse: bool,
    shape: tuple[int, int, int],
    tolerances: tuple[float, float, float, float],
) -> tuple[bool, str]:
    cutlass_attn = pytest.importorskip(
        "sageattention.cutlass_attn", reason="sageattention CUTLASS qattn kernel is not installed"
    )
    batch_size, num_heads, seq_len = shape
    q, k, v = _make_qkv(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        tensor_layout=tensor_layout,
        dtype=dtype,
    )
    expected = _expected(q, k, v, tensor_layout, is_causal, return_lse)

    actual = cutlass_attn._sageattn_cutlass_configured(
        q,
        k,
        v,
        tensor_layout,
        smooth_k,
        return_lse,
        config,
    )
    rtol, atol, lse_rtol, lse_atol = tolerances
    return _attention_report(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        lse_rtol=lse_rtol,
        lse_atol=lse_atol,
    )


@pytest.mark.parametrize(("config", "mode"), _valid_cases())
@pytest.mark.parametrize(("shape", "tolerances"), _TEST_CASES)
def test_sageattn_cutlass_config_matches_flashattention(
    config: tuple[int, int, int, int],
    mode: tuple[int, torch.dtype, str, bool, str, bool, bool],
    shape: tuple[int, int, int],
    tolerances: tuple[float, float, float, float],
) -> None:
    passed, msg = _run_case(config, *mode, shape, tolerances)
    assert passed, msg

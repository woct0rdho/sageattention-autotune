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
) -> tuple[bool, str]:
    cutlass_attn = pytest.importorskip(
        "sageattention.cutlass_attn", reason="sageattention CUTLASS qattn kernel is not installed"
    )
    q, k, v = _make_qkv(head_dim=head_dim, tensor_layout=tensor_layout, dtype=dtype)
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
    return _attention_report(actual, expected)


@pytest.mark.parametrize(("config", "mode"), _valid_cases())
def test_sageattn_cutlass_autotune_config(
    config: tuple[int, int, int, int],
    mode: tuple[int, torch.dtype, str, bool, str, bool, bool],
) -> None:
    passed, msg = _run_case(config, *mode)
    assert passed, msg

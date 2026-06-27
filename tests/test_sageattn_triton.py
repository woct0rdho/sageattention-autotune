from itertools import product

import torch
from test_sageattn import _error_report, _expected, _make_qkv

from sageattention.triton_attn import _sageattn_triton_configured
from sageattention.triton_autotune import _valid_triton_configs_for_head_dim

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


def _run_case(
    config: tuple[int, int, int, int],
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
        config,
    )

    return _error_report(actual, expected)


def _representative_config(
    block_config: tuple[int, int],
    *,
    head_dim: int,
    is_causal: bool,
    device: torch.device,
) -> tuple[int, int, int, int] | None:
    for config in _valid_triton_configs_for_head_dim(head_dim, is_causal, device):
        if config[:2] == block_config:
            return config
    return None


def _valid_block_configs(
    modes: tuple[tuple[int, torch.dtype, str, bool, str, bool], ...], device: torch.device
) -> tuple[tuple[int, int], ...]:
    block_configs = []
    for head_dim, _, _, is_causal, _, _ in modes:
        for config in _valid_triton_configs_for_head_dim(head_dim, is_causal, device):
            block_config = config[:2]
            if block_config not in block_configs:
                block_configs.append(block_config)
    return tuple(block_configs)


def test_sageattn_triton_block_configs() -> None:
    failed_configs = []
    device = torch.device("cuda")

    for block_config in _valid_block_configs(_MODES, device):
        config_errors = []
        config_tested = 0

        for head_dim, dtype, tensor_layout, is_causal, pv_accum_dtype, smooth_k in _MODES:
            config = _representative_config(block_config, head_dim=head_dim, is_causal=is_causal, device=device)
            if config is None:
                continue

            config_tested += 1
            name = (
                f"head_dim={head_dim} dtype={dtype} layout={tensor_layout} is_causal={is_causal} "
                f"pv_accum_dtype={pv_accum_dtype} smooth_k={smooth_k} full_config={config}"
            )
            try:
                passed, msg = _run_case(
                    config,
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
                config_errors.append(f"  {name}: {msg}")

        if config_tested == 0 or config_errors:
            errors = "\n".join(config_errors)
            failed_configs.append(f"{block_config} tested_cases={config_tested}\n{errors}")

    assert not failed_configs, "\n".join(failed_configs)

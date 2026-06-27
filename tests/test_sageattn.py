from itertools import product

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from sageattention.cuda_autotune import _AUTOTUNE_CONFIGS, _valid_configs

_MODES = tuple(
    product(
        (64, 128, 256),
        (torch.float16, torch.bfloat16),
        ("HND", "NHD"),
        (False, True),
        ("fp32", "fp16", "fp16+fp32"),
        (False, True),
    )
)


def _make_qkv(
    batch_size: int = 2,
    num_heads: int = 16,
    seq_len: int = 1024,
    head_dim: int = 64,
    tensor_layout: str = "HND",
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tensor_layout == "HND":
        shape = (batch_size, num_heads, seq_len, head_dim)
    else:
        shape = (batch_size, seq_len, num_heads, head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v


def _expected(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tensor_layout: str, is_causal: bool) -> torch.Tensor:
    if tensor_layout == "NHD":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    return expected.transpose(1, 2) if tensor_layout == "NHD" else expected


def _error_report(actual: torch.Tensor, expected: torch.Tensor) -> tuple[bool, str]:
    actual = actual.float()
    expected = expected.float()
    diff = actual - expected
    fro_rel_err = (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected).clamp(min=1e-6)).item()
    max_abs_err = diff.abs().max().item()

    passed = fro_rel_err <= 0.02 and max_abs_err <= 0.1
    msg = f"fro_rel_err={fro_rel_err:.3g} max_abs_err={max_abs_err:.3g}"
    return passed, msg


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
    cuda_attn = pytest.importorskip("sageattention.cuda_attn", reason="sageattention CUDA kernel is not installed")
    q, k, v = _make_qkv(head_dim=head_dim, tensor_layout=tensor_layout, dtype=dtype)
    expected = _expected(q, k, v, tensor_layout, is_causal)

    actual = cuda_attn._sageattn_configured(
        q,
        k,
        v,
        tensor_layout,
        is_causal,
        pv_accum_dtype,
        smooth_k,
        False,
        False,
        config,
    )

    return _error_report(actual, expected)


@pytest.mark.parametrize("config", _AUTOTUNE_CONFIGS, ids=str)
def test_sageattn_cuda_autotune_config(config: tuple[int, int, int, int]) -> None:
    config_errors = []
    config_tested = 0

    for head_dim, dtype, tensor_layout, is_causal, pv_accum_dtype, smooth_k in _MODES:
        q, _, _ = _make_qkv(head_dim=head_dim, tensor_layout=tensor_layout, dtype=dtype)
        if config not in _valid_configs(q.size(-1), is_causal, q.device.index):
            continue

        config_tested += 1
        name = (
            f"head_dim={head_dim} dtype={dtype} layout={tensor_layout} is_causal={is_causal} "
            f"pv_accum_dtype={pv_accum_dtype} smooth_k={smooth_k}"
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
            config_errors.append(f"{name}: {msg}")

    assert config_tested > 0
    assert not config_errors, "\n".join(config_errors)

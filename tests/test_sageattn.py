from itertools import product

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from sageattention.core import _sageattn_configured
from sageattention.core_autotune import _AUTOTUNE_CONFIGS, _valid_configs


def _make_qkv(batch_size=2, num_heads=16, seq_len=1024, head_dim=64, tensor_layout="HND", dtype=torch.float16):
    if tensor_layout == "HND":
        shape = (batch_size, num_heads, seq_len, head_dim)
    else:
        shape = (batch_size, seq_len, num_heads, head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v


def _expected(q, k, v, tensor_layout, is_causal):
    if tensor_layout == "NHD":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    return expected.transpose(1, 2) if tensor_layout == "NHD" else expected


def _error_report(actual, expected):
    actual = actual.float()
    expected = expected.float()
    diff = actual - expected
    fro_rel_err = (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected).clamp(min=1e-6)).item()
    max_abs_err = diff.abs().max().item()

    passed = fro_rel_err <= 0.02 and max_abs_err <= 0.1
    msg = f"fro_rel_err={fro_rel_err:.3g} max_abs_err={max_abs_err:.3g}"
    return passed, msg


def _run_case(config, *, head_dim, dtype, tensor_layout, is_causal, pv_accum_dtype, smooth_k):
    q, k, v = _make_qkv(head_dim=head_dim, tensor_layout=tensor_layout, dtype=dtype)
    expected = _expected(q, k, v, tensor_layout, is_causal)

    layout_i = {"NHD": 0, "HND": 1}[tensor_layout]
    pv_accum_i = {"fp32": 0, "fp16": 1, "fp16+fp32": 2}[pv_accum_dtype]
    actual = _sageattn_configured(
        q,
        k,
        v,
        layout_i,
        is_causal,
        None,
        pv_accum_i,
        smooth_k,
        False,
        False,
        config,
    )

    return _error_report(actual, expected)


def main():
    print(f"Testing SageAttention autotune configs ({len(_AUTOTUNE_CONFIGS)} compiled configs)\n")
    print("Config format: (blk_q, blk_k, warp_q, warp_k)")
    print("=" * 80)

    failed_configs = []
    passed_configs = []

    modes = list(
        product(
            (64, 128, 256),
            (torch.float16, torch.bfloat16),
            ("HND", "NHD"),
            (False, True),
            ("fp32", "fp16", "fp16+fp32"),
            (False, True),
        )
    )

    for config in _AUTOTUNE_CONFIGS:
        config_passed = True
        config_errors = []
        config_tested = 0

        for head_dim, dtype, tensor_layout, is_causal, pv_accum_dtype, smooth_k in modes:
            q, _, _ = _make_qkv(head_dim=head_dim, tensor_layout=tensor_layout, dtype=dtype)
            if config not in _valid_configs(q, is_causal):
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
                msg = f"error={e!r}"

            if not passed:
                config_passed = False
                config_errors.append(f"  {name}: {msg}")

        status = "PASS" if config_passed and config_tested > 0 else "FAIL"
        print(f"[{status}] {config} tested_cases={config_tested}")
        if config_passed and config_tested > 0:
            passed_configs.append(config)
        else:
            failed_configs.append(config)
            for error in config_errors:
                print(error)

    print("=" * 80)
    print(f"Summary: {len(passed_configs)}/{len(_AUTOTUNE_CONFIGS)} configs passed")

    if failed_configs:
        print(f"\nFailed configs ({len(failed_configs)}):")
        for config in failed_configs:
            print(f"  {config}")

    assert not failed_configs


if __name__ == "__main__":
    main()

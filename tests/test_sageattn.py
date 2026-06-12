from itertools import product

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from sageattention import sageattn_qk_int8_pv_fp16_cuda
from sageattention.autotune import _SM80_QK_AUTOTUNE_CONFIGS, _valid_sm80_qk_configs


def _error_report(actual, expected):
    actual = actual.float()
    expected = expected.float()
    diff = actual - expected
    fro_rel_err = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected).clamp(min=1e-6)
    max_abs_err = diff.abs().max()
    return fro_rel_err.item(), max_abs_err.item()


def _make_qkv(batch_size=2, num_heads=16, seq_len=1024, head_dim=64, tensor_layout="HND", dtype=torch.float16):
    if tensor_layout == "HND":
        shape = (batch_size, num_heads, seq_len, head_dim)
    else:
        shape = (batch_size, seq_len, num_heads, head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v


def _sdpa_expected(q, k, v, tensor_layout, is_causal):
    if tensor_layout == "NHD":
        q_ref = q.transpose(1, 2)
        k_ref = k.transpose(1, 2)
        v_ref = v.transpose(1, 2)
    else:
        q_ref, k_ref, v_ref = q, k, v

    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=is_causal)

    return expected.transpose(1, 2) if tensor_layout == "NHD" else expected


def _run_case(config, *, head_dim, dtype, tensor_layout, is_causal, qk_quant_gran, pv_accum_dtype):
    q, k, v = _make_qkv(head_dim=head_dim, tensor_layout=tensor_layout, dtype=dtype)
    expected = _sdpa_expected(q, k, v, tensor_layout, is_causal)
    actual = sageattn_qk_int8_pv_fp16_cuda(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        pv_accum_dtype=pv_accum_dtype,
        qk_config=config,
    )

    fro_rel_err, max_abs_err = _error_report(actual, expected)
    msg = f"fro_rel_err={fro_rel_err:.3g} max_abs_err={max_abs_err:.3g}"
    fro_rel_tol = 0.02 if dtype == torch.bfloat16 else 0.02
    max_abs_tol = 0.1 if dtype == torch.bfloat16 else 0.1
    passed = fro_rel_err <= fro_rel_tol and max_abs_err <= max_abs_tol
    return passed, msg


def main():
    torch.manual_seed(0)
    print(f"Testing SageAttention sm80 autotune configs ({len(_SM80_QK_AUTOTUNE_CONFIGS)} compiled configs)\n")
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
            ("per_thread", "per_warp"),
            ("fp32", "fp16", "fp16+fp32"),
        )
    )

    for config in _SM80_QK_AUTOTUNE_CONFIGS:
        config_passed = True
        config_errors = []
        config_tested = 0

        for head_dim, dtype, tensor_layout, is_causal, qk_quant_gran, pv_accum_dtype in modes:
            q, _, _ = _make_qkv(head_dim=head_dim, tensor_layout=tensor_layout, dtype=dtype)
            if config not in _valid_sm80_qk_configs(q, is_causal, qk_quant_gran):
                continue

            config_tested += 1
            name = (
                f"head_dim={head_dim} layout={tensor_layout} is_causal={is_causal} "
                f"dtype={dtype} qk_quant_gran={qk_quant_gran} pv_accum_dtype={pv_accum_dtype}"
            )
            try:
                passed, msg = _run_case(
                    config,
                    head_dim=head_dim,
                    dtype=dtype,
                    tensor_layout=tensor_layout,
                    is_causal=is_causal,
                    qk_quant_gran=qk_quant_gran,
                    pv_accum_dtype=pv_accum_dtype,
                )
            except Exception as exc:
                passed = False
                msg = f"error={exc!r}"

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
    print(f"Summary: {len(passed_configs)}/{len(_SM80_QK_AUTOTUNE_CONFIGS)} configs passed")

    if failed_configs:
        print(f"\nFailed configs ({len(failed_configs)}):")
        for config in failed_configs:
            print(f"  {config}")

    if passed_configs:
        print(f"\nPassed configs ({len(passed_configs)}):")
        for config in passed_configs:
            print(f"  {config}")

    assert not failed_configs


if __name__ == "__main__":
    main()

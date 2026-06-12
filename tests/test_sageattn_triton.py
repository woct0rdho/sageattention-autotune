from itertools import product

import torch
from test_sageattn import _error_report, _expected, _make_qkv

from sageattention.triton_attention import _sageattn_triton_configured
from sageattention.triton_autotune import _valid_triton_configs_for_head_dim


def _run_case(config, *, head_dim, dtype, tensor_layout, is_causal, pv_accum_dtype, smooth_k):
    q, k, v = _make_qkv(head_dim=head_dim, tensor_layout=tensor_layout, dtype=dtype)
    expected = _expected(q, k, v, tensor_layout, is_causal)

    layout_i = {"NHD": 0, "HND": 1}[tensor_layout]
    pv_accum_i = {"fp32": 0, "fp16": 1}[pv_accum_dtype]
    actual = _sageattn_triton_configured(
        q,
        k,
        v,
        layout_i,
        is_causal,
        None,
        pv_accum_i,
        smooth_k,
        False,
        config,
    )

    return _error_report(actual, expected)


def _representative_config(block_config, *, head_dim, is_causal, device):
    for config in _valid_triton_configs_for_head_dim(head_dim, is_causal, device):
        if config[:2] == block_config:
            return config
    return None


def _valid_block_configs(modes, device):
    block_configs = []
    for head_dim, _, _, is_causal, _, _ in modes:
        for config in _valid_triton_configs_for_head_dim(head_dim, is_causal, device):
            block_config = config[:2]
            if block_config not in block_configs:
                block_configs.append(block_config)
    return tuple(block_configs)


def main():
    print("Testing SageAttention Triton block configs")
    print("Config format: (BLOCK_M, BLOCK_N)")
    print("=" * 80)

    modes = list(
        product(
            (64, 128, 256),
            (torch.float16, torch.bfloat16),
            ("HND", "NHD"),
            (False, True),
            ("fp32", "fp16"),
            (False, True),
        )
    )

    failed_configs = []
    passed_configs = []
    device = torch.device("cuda")
    block_configs = _valid_block_configs(modes, device)

    for block_config in block_configs:
        config_passed = True
        config_errors = []
        config_tested = 0

        for head_dim, dtype, tensor_layout, is_causal, pv_accum_dtype, smooth_k in modes:
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
                msg = f"error={e!r}"

            if not passed:
                config_passed = False
                config_errors.append(f"  {name}: {msg}")

        status = "PASS" if config_passed and config_tested > 0 else "FAIL"
        print(f"[{status}] {block_config} tested_cases={config_tested}")
        if config_passed and config_tested > 0:
            passed_configs.append(block_config)
        else:
            failed_configs.append(block_config)
            for error in config_errors:
                print(error)

    print("=" * 80)
    print(f"Summary: {len(passed_configs)}/{len(block_configs)} block configs passed")

    if failed_configs:
        print(f"\nFailed block configs ({len(failed_configs)}):")
        for config in failed_configs:
            print(f"  {config}")

    assert not failed_configs


if __name__ == "__main__":
    main()

from itertools import product

import torch
from test_sageattn import _error_report, _expected, _make_qkv

from sageattention.triton_attention import sageattn_qk_int8_pv_fp16_triton


def _run_case(*, head_dim, dtype, tensor_layout, is_causal, pv_accum_dtype, smooth_k):
    q, k, v = _make_qkv(head_dim=head_dim, tensor_layout=tensor_layout, dtype=dtype)
    expected = _expected(q, k, v, tensor_layout, is_causal)

    actual = sageattn_qk_int8_pv_fp16_triton(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
    )

    return _error_report(actual, expected)


def main():
    print("Testing SageAttention Triton kernel")
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

    failures = []
    tested = 0
    for head_dim, dtype, tensor_layout, is_causal, pv_accum_dtype, smooth_k in modes:
        tested += 1
        name = (
            f"head_dim={head_dim} dtype={dtype} layout={tensor_layout} is_causal={is_causal} "
            f"pv_accum_dtype={pv_accum_dtype} smooth_k={smooth_k}"
        )
        try:
            passed, msg = _run_case(
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
            failures.append(f"  {name}: {msg}")

    if failures:
        print(f"[FAIL] {len(failures)}/{tested} cases failed")
        for failure in failures:
            print(failure)
    else:
        print(f"[PASS] {tested}/{tested} cases passed")

    assert not failures


if __name__ == "__main__":
    main()

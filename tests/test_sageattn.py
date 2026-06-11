from itertools import product

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from sageattention import sageattn


def _error_report(actual, expected):
    actual = actual.float()
    expected = expected.float()
    diff = actual - expected
    fro_rel_err = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected).clamp(min=1e-6)
    max_abs_err = diff.abs().max()
    return fro_rel_err.item(), max_abs_err.item()


def _run_case(**kwargs):
    q = torch.randn(4, 32, 64, 128, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(q, k, v, is_causal=kwargs.get("is_causal", False))

    actual = sageattn(q, k, v, **kwargs)
    fro_rel_err, max_abs_err = _error_report(actual, expected)
    msg = f"fro_rel_err={fro_rel_err:.3g} max_abs_err={max_abs_err:.3g}"
    passed = fro_rel_err <= 0.02 and max_abs_err <= 0.1
    return passed, msg


def main():
    cases = []
    for is_causal, qk_quant_gran, pv_accum_dtype in product(
        (False, True),
        ("per_thread", "per_warp"),
        ("fp32", "fp16", "fp16+fp32"),
    ):
        kwargs = {
            "is_causal": is_causal,
            "qk_quant_gran": qk_quant_gran,
            "pv_accum_dtype": pv_accum_dtype,
        }
        name = f"is_causal={is_causal} qk_quant_gran={qk_quant_gran} pv_accum_dtype={pv_accum_dtype}"
        cases.append((name, kwargs))

    all_passed = True
    failures = []
    for name, kwargs in cases:
        try:
            passed, msg = _run_case(**kwargs)
        except Exception as exc:
            passed = False
            msg = f"error={exc!r}"

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {msg}")
        all_passed = all_passed and passed
        if not passed:
            failures.append(f"{name}: {msg}")

    print(f"All test cases pass: {all_passed}")
    assert all_passed, "\n".join(failures)


if __name__ == "__main__":
    main()

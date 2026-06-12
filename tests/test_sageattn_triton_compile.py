import torch
from test_sageattn import _error_report, _expected, _make_qkv

from sageattention import sageattn_qk_int8_pv_fp16_triton


def _check(actual, expected, label):
    passed, msg = _error_report(actual, expected)
    msg = f"{label}: {msg}"
    assert passed, msg
    print(msg)


def test_eager_autotuned():
    q, k, v = _make_qkv()
    expected = _expected(q, k, v, "HND", False)
    actual = sageattn_qk_int8_pv_fp16_triton(q, k, v, tensor_layout="HND", is_causal=False)
    _check(actual, expected, "eager Triton autotuned")


def test_compile_autotuned():
    q, k, v = _make_qkv()
    expected = _expected(q, k, v, "HND", False)

    @torch.compile(fullgraph=True, mode="max-autotune")
    def fn(q, k, v):
        return sageattn_qk_int8_pv_fp16_triton(q, k, v, tensor_layout="HND", is_causal=False)

    actual = fn(q, k, v)
    _check(actual, expected, "compile Triton autotuned")


def main():
    test_eager_autotuned()
    test_compile_autotuned()
    print("All Triton compile tests passed")


if __name__ == "__main__":
    main()

import pytest
import torch
from test_sageattn import _error_report, _expected, _make_qkv


def _check(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    passed, msg = _error_report(actual, expected)
    msg = f"{label}: {msg}"
    assert passed, msg
    print(msg)


def test_eager_autotuned() -> None:
    cuda_attn = pytest.importorskip("sageattention.cuda_attn", reason="sageattention CUDA kernel is not installed")
    q, k, v = _make_qkv()
    expected = _expected(q, k, v, "HND", False)
    actual = cuda_attn.sageattn_qk_int8_pv_fp16_cuda(q, k, v, tensor_layout="HND", is_causal=False)
    _check(actual, expected, "eager autotuned")


def test_compile_autotuned() -> None:
    cuda_attn = pytest.importorskip("sageattention.cuda_attn", reason="sageattention CUDA kernel is not installed")
    q, k, v = _make_qkv()
    expected = _expected(q, k, v, "HND", False)

    @torch.compile(fullgraph=True, mode="max-autotune")
    def fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return cuda_attn.sageattn_qk_int8_pv_fp16_cuda(q, k, v, tensor_layout="HND", is_causal=False)

    actual = fn(q, k, v)
    _check(actual, expected, "compile autotuned")

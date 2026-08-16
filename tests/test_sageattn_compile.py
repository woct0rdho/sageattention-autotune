from collections.abc import Callable

import pytest
import torch
from test_sageattn import _attention_report, _expected, _make_qkv
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake


def _check(
    actual: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    expected: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    label: str,
) -> None:
    passed, msg = _attention_report(actual, expected)
    msg = f"{label}: {msg}"
    assert passed, msg
    print(msg)


def _check_fake(
    fn: Callable[..., torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    return_lse: bool,
) -> None:
    with FakeTensorMode():
        q = torch.empty((1, 4, 65, 80), device="cuda", dtype=torch.float16)
        k = torch.empty_like(q)
        v = torch.empty_like(q)
        result = fn(q, k, v, tensor_layout="HND", return_lse=return_lse)

        tensors = result if isinstance(result, tuple) else (result,)
        assert all(is_fake(tensor) for tensor in tensors)
        assert tensors[0].shape == q.shape
        assert tensors[0].dtype == q.dtype
        if return_lse:
            assert len(tensors) == 2
            assert tensors[1].shape == (1, 4, 65)
            assert tensors[1].dtype == torch.float32
        else:
            assert len(tensors) == 1


def _check_functionalize(
    fn: Callable[..., torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    return_lse: bool,
) -> None:
    q, k, v = _make_qkv()
    expected = _expected(q, k, v, "HND", False, return_lse)
    functionalized = torch.func.functionalize(lambda q, k, v: fn(q, k, v, tensor_layout="HND", return_lse=return_lse))
    actual = functionalized(q, k, v)
    _check(actual, expected, f"functionalize return_lse={return_lse}")


@pytest.mark.parametrize("return_lse", (False, True))
def test_eager_autotuned(return_lse: bool) -> None:
    cuda_attn = pytest.importorskip("sageattention.cuda_attn", reason="sageattention CUDA kernel is not installed")
    q, k, v = _make_qkv()
    expected = _expected(q, k, v, "HND", False, return_lse)

    actual = cuda_attn.sageattn_qk_int8_pv_fp16_cuda(
        q, k, v, tensor_layout="HND", is_causal=False, return_lse=return_lse
    )
    _check(actual, expected, f"eager autotuned return_lse={return_lse}")


@pytest.mark.parametrize("return_lse", (False, True))
def test_compile_autotuned(return_lse: bool) -> None:
    cuda_attn = pytest.importorskip("sageattention.cuda_attn", reason="sageattention CUDA kernel is not installed")
    q, k, v = _make_qkv()
    expected = _expected(q, k, v, "HND", False, return_lse)

    fn = torch.compile(cuda_attn.sageattn_qk_int8_pv_fp16_cuda, fullgraph=True, mode="max-autotune")
    actual = fn(q, k, v, tensor_layout="HND", is_causal=False, return_lse=return_lse)
    _check(actual, expected, f"compile autotuned return_lse={return_lse}")


@pytest.mark.parametrize("return_lse", (False, True))
def test_fake_autotuned(return_lse: bool) -> None:
    cuda_attn = pytest.importorskip("sageattention.cuda_attn", reason="sageattention CUDA kernel is not installed")
    _check_fake(cuda_attn.sageattn_qk_int8_pv_fp16_cuda, return_lse)


@pytest.mark.parametrize("return_lse", (False, True))
def test_functionalize_autotuned(return_lse: bool) -> None:
    cuda_attn = pytest.importorskip("sageattention.cuda_attn", reason="sageattention CUDA kernel is not installed")
    _check_functionalize(cuda_attn.sageattn_qk_int8_pv_fp16_cuda, return_lse)

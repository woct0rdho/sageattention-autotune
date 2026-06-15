"""Correctness tests for the int8-QK / fp8-PV (sm89/sm120) kernels.

Skipped automatically unless the current GPU uses the fp8 backend (Ada / Blackwell
with the _qattn_sm89 extension built).
"""

from itertools import product

import pytest
import torch
from test_sageattn import _error_report, _expected, _make_qkv

from sageattention.cuda_attn import _sageattn_configured
from sageattention.cuda_compile import use_fp8_backend

_FP8 = torch.cuda.is_available() and use_fp8_backend(torch.device("cuda"))
pytestmark = pytest.mark.skipif(not _FP8, reason="fp8 (sm89/sm120) backend not active on this GPU")

_SM89_CONFIGS = (
    (128, 64, 32, 64),
    (64, 64, 32, 64),
    (128, 64, 16, 64),
)


@pytest.mark.parametrize("config", _SM89_CONFIGS)
@pytest.mark.parametrize("pv_accum_dtype", ["fp32", "fp16+fp32"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp8_configured(config, pv_accum_dtype, head_dim, tensor_layout, is_causal, dtype):
    q, k, v = _make_qkv(head_dim=head_dim, tensor_layout=tensor_layout, dtype=dtype)
    expected = _expected(q, k, v, tensor_layout, is_causal)
    actual = _sageattn_configured(
        q, k, v, tensor_layout, is_causal, pv_accum_dtype, True, False, False, config
    )
    passed, msg = _error_report(actual, expected)
    assert passed, msg


@pytest.mark.parametrize("hq,hkv", [(16, 16), (16, 4), (16, 1)])
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
def test_fp8_gqa(hq, hkv, tensor_layout):
    from sageattention import sageattn

    b, s, d = 2, 512, 128
    shape_q = (b, hq, s, d) if tensor_layout == "HND" else (b, s, hq, d)
    shape_kv = (b, hkv, s, d) if tensor_layout == "HND" else (b, s, hkv, d)
    q = torch.randn(shape_q, device="cuda", dtype=torch.float16)
    k = torch.randn(shape_kv, device="cuda", dtype=torch.float16)
    v = torch.randn(shape_kv, device="cuda", dtype=torch.float16)

    hd_dim = 2 if tensor_layout == "HND" else 2  # head dim index after transpose
    qT = q if tensor_layout == "HND" else q.transpose(1, 2)
    kT = (k if tensor_layout == "HND" else k.transpose(1, 2)).repeat_interleave(hq // hkv, 1)
    vT = (v if tensor_layout == "HND" else v.transpose(1, 2)).repeat_interleave(hq // hkv, 1)
    ref = torch.nn.functional.scaled_dot_product_attention(qT, kT, vT)
    if tensor_layout == "NHD":
        ref = ref.transpose(1, 2)

    actual = sageattn(q, k, v, tensor_layout=tensor_layout, is_causal=False)
    passed, msg = _error_report(actual, ref)
    assert passed, msg

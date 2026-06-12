import importlib
import os

from .triton_attn import sageattn_qk_int8_pv_fp16_triton

try:
    importlib.import_module(f"{__package__}._qattn_sm80")
except (ImportError, OSError):
    sageattn_qk_int8_pv_fp16_cuda = None
else:
    from .cuda_attn import sageattn_qk_int8_pv_fp16_cuda

if os.getenv("SAGEATTN_TRITON_BACKEND", "0") == "1" or sageattn_qk_int8_pv_fp16_cuda is None:
    sageattn = sageattn_qk_int8_pv_fp16_triton
else:
    sageattn = sageattn_qk_int8_pv_fp16_cuda

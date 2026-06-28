import importlib
import os

from .triton_attn import sageattn_qk_int8_pv_fp16_triton
from .triton_bwd import sageattn_qk_int8_pv_fp16_triton_trainable
from .triton_bwd_reuse import sageattn_qk_int8_pv_fp16_triton_trainable_reuse

try:
    importlib.import_module(f"{__package__}._qattn_sm80")
except (ImportError, OSError):
    sageattn_qk_int8_pv_fp16_cuda = None
else:
    from .cuda_attn import sageattn_qk_int8_pv_fp16_cuda

if os.getenv("SAGEATTN_BACKEND", "").lower() == "triton":
    sageattn = sageattn_qk_int8_pv_fp16_triton
elif os.getenv("SAGEATTN_BACKEND", "").lower() == "triton_trainable":
    sageattn = sageattn_qk_int8_pv_fp16_triton_trainable
elif os.getenv("SAGEATTN_BACKEND", "").lower() == "triton_trainable_reuse":
    sageattn = sageattn_qk_int8_pv_fp16_triton_trainable_reuse
else:
    sageattn = sageattn_qk_int8_pv_fp16_cuda

# Placeholders for compatibility with libraries such as diffusers. Not implemented yet.
sageattn_qk_int8_pv_fp8_cuda = None
sageattn_qk_int8_pv_fp8_cuda_sm90 = None
sageattn_varlen = None

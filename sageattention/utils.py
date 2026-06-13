import os

import torch
import torch.nn.functional as F

LOG2_E = 1.44269504

DEFAULT_PV_ACCUM_DTYPE = os.getenv("SAGEATTN_DEFAULT_PV_ACCUM_DTYPE", "fp32").lower()
if DEFAULT_PV_ACCUM_DTYPE not in ("fp32", "fp16", "fp16+fp32"):
    DEFAULT_PV_ACCUM_DTYPE = "fp32"


def _env_flag_enabled(name):
    return os.getenv(name, "0").lower() in ("1", "true", "yes", "on")


def _padded_head_dim(head_dim: int) -> int:
    if head_dim < 64:
        return 64
    if 64 < head_dim < 128:
        return 128
    if 128 < head_dim < 256:
        return 256
    if head_dim in (64, 128, 256):
        return head_dim
    raise ValueError(f"Unsupported head_dim: {head_dim}")


def _pad_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    head_dim = q.size(-1)
    pad_to = _padded_head_dim(head_dim)
    if pad_to == head_dim:
        return head_dim, q, k, v

    padding = (0, pad_to - head_dim)
    return head_dim, F.pad(q, padding), F.pad(k, padding), F.pad(v, padding)

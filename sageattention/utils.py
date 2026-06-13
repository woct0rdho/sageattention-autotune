import os

import torch
import torch.nn.functional as F

LOG2_E = 1.44269504088896340736

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


def _lse_correction(q: torch.Tensor, km: torch.Tensor, tensor_layout: str, head_dim_index: int) -> torch.Tensor:
    num_qo_heads = q.size(head_dim_index)
    num_kv_heads = km.size(head_dim_index)
    q_per_kv_heads = num_qo_heads // num_kv_heads
    km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=head_dim_index) if q_per_kv_heads > 1 else km

    if tensor_layout == "NHD":
        correction = torch.matmul(q.transpose(1, 2), km_broadcast.permute(0, 2, 3, 1)).squeeze(-1)
    else:
        correction = torch.matmul(q, km_broadcast.transpose(2, 3)).squeeze(-1)
    return correction.to(torch.float32)

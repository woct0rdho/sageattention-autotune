import os
from typing import Literal, overload

import torch
import torch.nn.functional as F
from torch._guards import detect_fake_mode

LOG2_E = 1.44269504088896340736

DEFAULT_PV_ACCUM_DTYPE = os.getenv("SAGEATTN_DEFAULT_PV_ACCUM_DTYPE", "fp32").lower()
if DEFAULT_PV_ACCUM_DTYPE not in ("fp32", "fp16", "fp16+fp32"):
    DEFAULT_PV_ACCUM_DTYPE = "fp32"


def _env_flag_enabled(name: str) -> bool:
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


def _pad_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    head_dim = q.size(-1)
    pad_to = _padded_head_dim(head_dim)
    if pad_to == head_dim:
        return head_dim, q, k, v

    padding = (0, pad_to - head_dim)
    return head_dim, F.pad(q, padding), F.pad(k, padding), F.pad(v, padding)


def _is_compiling_or_fake(tensor: torch.Tensor) -> bool:
    return (
        torch.compiler.is_compiling() or detect_fake_mode((tensor,)) is not None or torch._is_functional_tensor(tensor)
    )


def _allocate_forward_outputs(
    query: torch.Tensor,
    tensor_layout: str,
    return_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    padded_head_dim = _padded_head_dim(query.size(-1))
    output = query.new_empty((*query.shape[:-1], padded_head_dim))

    if tensor_layout == "NHD":
        num_qo_heads = query.size(2)
        qo_len = query.size(1)
    elif tensor_layout == "HND":
        num_qo_heads = query.size(1)
        qo_len = query.size(2)
    else:
        raise ValueError("tensor_layout must be 'NHD' or 'HND'.")

    lse = query.new_empty((query.size(0), num_qo_heads, qo_len), dtype=torch.float32) if return_lse else None
    return output, lse


@overload
def _format_forward_outputs(
    output: torch.Tensor,
    lse: torch.Tensor | None,
    head_dim: int,
    return_lse: Literal[False],
) -> torch.Tensor: ...


@overload
def _format_forward_outputs(
    output: torch.Tensor,
    lse: torch.Tensor | None,
    head_dim: int,
    return_lse: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def _format_forward_outputs(
    output: torch.Tensor,
    lse: torch.Tensor | None,
    head_dim: int,
    return_lse: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...


def _format_forward_outputs(
    output: torch.Tensor,
    lse: torch.Tensor | None,
    head_dim: int,
    return_lse: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    output = output[..., :head_dim]
    if return_lse:
        assert lse is not None
        return output, lse
    return output


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

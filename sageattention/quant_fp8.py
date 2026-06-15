"""Per-channel fp8 (e4m3) quantization of V for the sm89/sm120 fp8-PV kernels.

The fp8 attention kernel consumes V transposed to
``[batch, num_kv_heads, head_dim, kv_len_padded]`` (e4m3 stored as int8) together
with a per-channel scale ``[batch, num_kv_heads, head_dim]``. The kernel computes
``softmax(QK^T) @ V_fp8`` and the per-channel scale is applied to the output via
the ``fuse_v_scale`` path, so ``out = scale * (softmax @ V_fp8)`` recovers the real
attention output.
"""

import torch
import torch.nn.functional as F

# Largest finite magnitude representable by float8_e4m3fn.
_E4M3_MAX = 448.0

# Within each group of 16 kv positions, the fp8 kernel's ldmatrix reads V in this
# permuted order (matches upstream transpose_pad_permute_cuda). Caches the gather
# index per (seq_pad, device).
_PERM16 = (0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15)
_PERM_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


def _seq_permute_index(seq_pad: int, device: torch.device) -> torch.Tensor:
    key = (seq_pad, device)
    idx = _PERM_CACHE.get(key)
    if idx is None:
        perm = torch.tensor(_PERM16, device=device)
        idx = (torch.arange(seq_pad // 16, device=device)[:, None] * 16 + perm[None, :]).reshape(-1)
        _PERM_CACHE[key] = idx
    return idx


def per_channel_fp8_v(
    v: torch.Tensor,
    tensor_layout: str = "HND",
    pad_to: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize V to per-channel e4m3 fp8, transposed/padded/permuted for the kernel.

    Mirrors upstream SageAttention's ``per_channel_fp8``: V is transposed so
    head_dim is the row and kv_len the (contiguous) column, padded to a multiple of
    ``pad_to``, and the kv columns are permuted within each 16-block by ``_PERM16``
    so the fp8 ``ldmatrix`` loads land in the right lanes. Scale is per-channel.

    Args:
        v: value tensor, ``[b, h_kv, kv_len, d]`` (HND) or ``[b, kv_len, h_kv, d]`` (NHD).
        tensor_layout: "HND" or "NHD".
        pad_to: kv_len is padded to a multiple of this (must cover the largest CTA_K).

    Returns:
        v_fp8: int8 tensor ``[b, h_kv, d, kv_len_padded]`` holding e4m3 bytes, contiguous.
        v_scale: float32 tensor ``[b, h_kv, d]`` (per-channel dequant scale).
    """
    if tensor_layout == "NHD":
        v = v.transpose(1, 2)  # -> [b, h_kv, kv_len, d]
    elif tensor_layout != "HND":
        raise ValueError("tensor_layout must be 'NHD' or 'HND'.")

    b, h_kv, kv_len, d = v.shape

    # per-channel scale over the real sequence (padding/permutation do not change amax)
    v_t = v.transpose(2, 3)  # [b, h_kv, d, kv_len] (view)
    scale = v_t.abs().amax(dim=-1, keepdim=True).float() / _E4M3_MAX + 1e-8  # [b,h,d,1]

    kv_len_pad = ((kv_len + pad_to - 1) // pad_to) * pad_to
    v_pad = F.pad(v_t, (0, kv_len_pad - kv_len)) if kv_len_pad != kv_len else v_t

    # permute kv within each 16-block to match the kernel's fp8 ldmatrix layout
    idx = _seq_permute_index(kv_len_pad, v.device)
    v_perm = v_pad.index_select(-1, idx)

    v_fp8 = (v_perm / scale).to(torch.float8_e4m3fn).view(torch.int8).contiguous()
    v_scale = scale.squeeze(-1).contiguous()  # [b, h_kv, d]
    return v_fp8, v_scale

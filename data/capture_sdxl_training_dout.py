import gc
import logging
import sys
from pathlib import Path
from typing import Any

COMFY_ROOT = Path(r"C:\ComfyUI")
CHECKPOINT = COMFY_ROOT / "models" / "checkpoints" / "sdxl_anime" / "WAI-Illustrious-v17.safetensors"
OUTPUT = Path(__file__).resolve().parent / "sdxl_periodic_ds_inputs_training_dout_latent128_seed2.pt"
LATENT_SIZE = 128
IMAGE_SIZE = 1024
SEED = 20260809
LOSS_SCALE = 65536.0
PROMPT = "an anime illustration of a moonlit mountain village, lanterns, detailed architecture, rich atmospheric colors"
SIGMA_INDICES = (300, 700, 900)
TARGET_SEQ = (LATENT_SIZE // 2) * (LATENT_SIZE // 2)

sys.path.insert(0, str(COMFY_ROOT))
sys.argv = [sys.argv[0]]

import comfy.ldm.modules.attention as comfy_attention  # ty: ignore[unresolved-import]
import comfy.sd  # ty: ignore[unresolved-import]
import torch
import torch.nn.functional as F
from comfy import model_management  # ty: ignore[unresolved-import]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def to_nhd(tensor: torch.Tensor, heads: int, skip_reshape: bool) -> torch.Tensor:
    if skip_reshape:
        if tensor.ndim != 4 or tensor.shape[1] != heads:
            raise ValueError(f"unexpected skip-reshape tensor shape: {tuple(tensor.shape)}, heads={heads}")
        return tensor.transpose(1, 2).contiguous()
    if tensor.ndim != 3 or tensor.shape[-1] % heads != 0:
        raise ValueError(f"unexpected packed tensor shape: {tuple(tensor.shape)}, heads={heads}")
    return tensor.reshape(tensor.shape[0], tensor.shape[1], heads, tensor.shape[-1] // heads).contiguous()


def output_to_nhd(gradient: torch.Tensor, heads: int, skip_output_reshape: bool, dim_head: int) -> torch.Tensor:
    if skip_output_reshape:
        if gradient.ndim != 4:
            raise ValueError(f"unexpected attention output gradient shape: {tuple(gradient.shape)}")
        return gradient.transpose(1, 2).contiguous()
    if gradient.ndim != 3 or gradient.shape[-1] != heads * dim_head:
        raise ValueError(f"unexpected packed attention output gradient shape: {tuple(gradient.shape)}")
    return gradient.reshape(gradient.shape[0], gradient.shape[1], heads, dim_head).contiguous()


def unload(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    model_management.unload_all_models()
    torch.cuda.empty_cache()


def load_text_condition(device: torch.device, cpu: torch.device):
    print("loading SDXL CLIP")
    _, clip, _, _ = comfy.sd.load_checkpoint_guess_config(
        str(CHECKPOINT),
        output_vae=False,
        output_clip=True,
        output_clipvision=False,
        output_model=False,
        model_options={"load_device": device, "offload_device": cpu, "dtype": torch.float16},
        disable_dynamic=True,
    )
    tokens = clip.tokenize(PROMPT)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    cond = cond.detach().to(device=cpu, dtype=torch.float16)
    pooled = pooled.detach().to(device=cpu, dtype=torch.float32)
    print(f"prompt condition={tuple(cond.shape)} pooled={tuple(pooled.shape)}")
    unload(clip, clip.patcher)
    return cond, pooled


def load_random_image_latent(device: torch.device, cpu: torch.device) -> torch.Tensor:
    print("loading SDXL VAE")
    vae_patcher = comfy.sd.load_checkpoint_guess_config(
        str(CHECKPOINT),
        output_vae=True,
        output_clip=False,
        output_clipvision=False,
        output_model=False,
        model_options={"load_device": device, "offload_device": cpu, "dtype": torch.float16},
        disable_dynamic=True,
    )[2]
    vae = vae_patcher
    image = torch.rand((1, IMAGE_SIZE, IMAGE_SIZE, 3), device=device, dtype=torch.float16) * 2.0 - 1.0
    with torch.inference_mode():
        latent = vae.encode(image)
    latent = latent.detach().to(device=cpu, dtype=torch.float32)
    print(f"encoded random image latent={tuple(latent.shape)} dtype={latent.dtype}")
    unload(vae, getattr(vae, "patcher", None), image)
    return latent


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    cpu = torch.device("cpu")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    context_cpu, pooled_cpu = load_text_condition(device, cpu)
    clean_latent_cpu = load_random_image_latent(device, cpu)

    print(f"loading UNet {CHECKPOINT}")
    patcher = comfy.sd.load_checkpoint_guess_config_model_only(
        str(CHECKPOINT),
        model_options={"load_device": device, "offload_device": cpu, "dtype": torch.float16},
        disable_dynamic=True,
    )
    model = patcher.patch_model(device)
    model.eval()
    model.requires_grad_(False)
    dtype = model.get_dtype_inference()
    context = context_cpu.to(device=device, dtype=dtype)
    pooled = pooled_cpu.to(device=device, dtype=torch.float32)
    y = model.encode_adm(
        pooled_output=pooled,
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
        crop_w=0,
        crop_h=0,
        target_width=IMAGE_SIZE,
        target_height=IMAGE_SIZE,
    ).to(device=device, dtype=dtype)
    clean_latent = clean_latent_cpu.to(device=device, dtype=dtype)
    print(
        f"model={type(model).__name__} dtype={dtype} adm={tuple(y.shape)} "
        f"loaded_gib={torch.cuda.memory_allocated() / 2**30:.3f}"
    )

    records: list[dict[str, Any]] = []
    original_attention = comfy_attention.optimized_attention
    original_attention_masked = comfy_attention.optimized_attention_masked

    def capture_attention(
        q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs
    ):
        nonlocal records
        q_nhd = to_nhd(q, heads, skip_reshape)
        k_nhd = to_nhd(k, heads, skip_reshape)
        v_nhd = to_nhd(v, heads, skip_reshape)
        should_capture = (
            len(records) < len(SIGMA_INDICES)
            and mask is None
            and q_nhd.shape[1] == k_nhd.shape[1] == v_nhd.shape[1] == TARGET_SEQ
            and q_nhd.shape[-1] == 64
            and (not records or "dout" in records[-1])
        )
        out = original_attention(
            q,
            k,
            v,
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )
        if should_capture:
            record = {
                "index": len(records),
                "seq_len": int(q_nhd.shape[1]),
                "heads": int(heads),
                "head_dim": int(q_nhd.shape[-1]),
                "q": q_nhd.detach().to(device=cpu, dtype=torch.float16),
                "k": k_nhd.detach().to(device=cpu, dtype=torch.float16),
                "v": v_nhd.detach().to(device=cpu, dtype=torch.float16),
            }
            records.append(record)
            if not out.requires_grad:
                raise RuntimeError("captured attention output does not require grad")

            def save_dout(gradient: torch.Tensor, record=record) -> torch.Tensor:
                record["dout"] = output_to_nhd(gradient.detach(), heads, skip_output_reshape, q_nhd.shape[-1]).to(
                    device=cpu, dtype=torch.float16
                )
                return gradient

            out.register_hook(save_dout)
            print(f"captured forward index={record['index']} seq={record['seq_len']} heads={record['heads']}")
        return out

    def capture_attention_masked(
        q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs
    ):
        if original_attention_masked is original_attention:
            return capture_attention(
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )
        return original_attention_masked(
            q,
            k,
            v,
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )

    comfy_attention.optimized_attention = capture_attention
    comfy_attention.optimized_attention_masked = capture_attention_masked

    losses = []
    try:
        for capture_index, sigma_index in enumerate(SIGMA_INDICES):
            sigma_value = float(model.model_sampling.sigmas[sigma_index])
            sigma = torch.tensor([sigma_value], device=device, dtype=torch.float32)
            generator = torch.Generator(device=device)
            generator.manual_seed(SEED + 1000 + sigma_index)
            noise = torch.randn(clean_latent.shape, device=device, dtype=dtype, generator=generator)
            noisy_latent = model.model_sampling.noise_scaling(sigma, noise, clean_latent).detach().requires_grad_(True)
            with torch.enable_grad():
                output = model.apply_model(
                    noisy_latent,
                    sigma,
                    c_crossattn=context,
                    y=y,
                    transformer_options={},
                )
                # For an EPS model, x0 MSE is epsilon MSE multiplied by sigma^2.
                # The global factor does not change the periodic quantization ratios.
                loss = F.mse_loss(output.float(), clean_latent.float(), reduction="mean") * LOSS_SCALE
                loss.backward()
            torch.cuda.synchronize()
            if len(records) != capture_index + 1 or "dout" not in records[-1]:
                raise RuntimeError(f"dO was not captured for sigma index {sigma_index}")
            records[-1]["sigma_index"] = sigma_index
            records[-1]["sigma"] = sigma_value
            records[-1]["loss"] = float(loss.detach().cpu())
            losses.append(float(loss.detach().cpu()))
            print(f"step sigma_index={sigma_index} sigma={sigma_value:.6f} loss={loss.item():.6e}")
            del output, loss, noisy_latent, noise, sigma
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        comfy_attention.optimized_attention = original_attention
        comfy_attention.optimized_attention_masked = original_attention_masked

    required = ("q", "k", "v", "dout")
    for record in records:
        missing = [name for name in required if name not in record]
        if missing:
            raise RuntimeError(f"record {record['index']} missing {missing}")

    payload = {
        "checkpoint": str(CHECKPOINT),
        "seed": SEED,
        "prompt": PROMPT,
        "image_size": IMAGE_SIZE,
        "latent_size": LATENT_SIZE,
        "sigma_indices": SIGMA_INDICES,
        "target_seq": TARGET_SEQ,
        "losses": losses,
        "loss_scale": LOSS_SCALE,
        "input_kind": "random image encoded by the SDXL VAE, real SDXL CLIP prompt condition, DDPM-style noisy latent, and one-step x0 MSE training loss",
        "loss_relation": "For the EPS model, x0 MSE is sigma^2 times epsilon MSE at fixed sigma; dO differs only by a global factor.",
        "records": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, OUTPUT)
    print(f"saved {len(records)} records to {OUTPUT}")
    print(f"record_shapes={[tuple(record['q'].shape) for record in records]}")
    print(f"peak_cuda_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}")

    del payload, records, model, patcher, context, pooled, y, clean_latent, context_cpu, pooled_cpu, clean_latent_cpu
    unload()


if __name__ == "__main__":
    main()

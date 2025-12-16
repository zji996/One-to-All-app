from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from py_core.one_to_all_animation.df11_loader import df11_available, load_df11_into_model
from py_core.one_to_all_animation.fp8_loader import has_fp8_scaled_weights, load_fp8_scaled_into_module
from py_core.one_to_all_animation.torch_utils import torch_default_dtype


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    v = v.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class BuiltPipeline:
    pipe: Any
    model_dtype: Any
    prompt_embeds: Any | None
    negative_prompt_embeds: Any | None
    negative_prompt: list[str]


def build_pipeline(
    *,
    video_generation_dir: Path,
    checkpoint_dir: Path,
    wan_dir: Path,
    device: str,
    use_df11_initial: bool,
    df11_force_on: bool,
    df11_force_off: bool,
    prompt: str,
) -> BuiltPipeline:
    import torch
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    from transformers import AutoConfig, AutoTokenizer, UMT5EncoderModel
    from diffusers.models import AutoencoderKLWan

    from opensora.sample.pipeline_wanx_vhuman_tokenreplace import WanPipeline
    from opensora.model_variants.wanx_diffusers_src import (
        WanTransformer3DModel_Refextractor_2D_Controlnet_prefix,
    )

    model_dtype = torch.bfloat16
    scheduler = FlowMatchEulerDiscreteScheduler(
        shift=7.0, num_train_timesteps=1000, use_dynamic_shifting=False
    )

    use_df11 = bool(use_df11_initial)

    def _load_text_encoder_and_tokenizer():
        nonlocal use_df11
        tokenizer = AutoTokenizer.from_pretrained(str(wan_dir), subfolder="tokenizer")
        te_dir = wan_dir / "text_encoder"
        te_df11_dir = wan_dir / "text_encoder_df11"

        if not df11_force_on and not df11_force_off:
            if not use_df11 and te_df11_dir.is_dir() and df11_available():
                use_df11 = True

        if use_df11 and te_df11_dir.is_dir() and _env_bool("ONE_TO_ALL_DF11_TEXT_ENCODER", True):
            te_config = AutoConfig.from_pretrained(str(wan_dir), subfolder="text_encoder")
            text_encoder = UMT5EncoderModel(te_config)
            load_df11_into_model(text_encoder, te_df11_dir)
            text_encoder = text_encoder.to(dtype=model_dtype)
        elif has_fp8_scaled_weights(te_dir):
            te_config = AutoConfig.from_pretrained(str(wan_dir), subfolder="text_encoder")
            text_encoder = UMT5EncoderModel(te_config)
            missing, unexpected = load_fp8_scaled_into_module(
                text_encoder, te_dir, target_dtype=model_dtype, strict=False
            )
            if missing:
                raise KeyError(
                    f"FP8 text_encoder missing keys (count={len(missing)}), first={missing[:5]}"
                )
            if unexpected:
                raise KeyError(
                    f"FP8 text_encoder unexpected keys (count={len(unexpected)}), first={unexpected[:5]}"
                )
            text_encoder = text_encoder.to(dtype=model_dtype)
        else:
            text_encoder = UMT5EncoderModel.from_pretrained(
                str(wan_dir), torch_dtype=model_dtype, subfolder="text_encoder"
            )
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        return text_encoder, tokenizer

    def _load_vae():
        vae_dir = wan_dir / "vae"
        if has_fp8_scaled_weights(vae_dir):
            vae_config = AutoencoderKLWan.load_config(str(wan_dir), subfolder="vae")
            vae = AutoencoderKLWan.from_config(vae_config).to(dtype=model_dtype)
            missing, unexpected = load_fp8_scaled_into_module(
                vae, vae_dir, target_dtype=model_dtype, strict=False
            )
            if missing:
                raise KeyError(f"FP8 VAE missing keys (count={len(missing)}), first={missing[:5]}")
            if unexpected:
                raise KeyError(
                    f"FP8 VAE unexpected keys (count={len(unexpected)}), first={unexpected[:5]}"
                )
        else:
            vae = AutoencoderKLWan.from_pretrained(
                str(wan_dir), torch_dtype=model_dtype, subfolder="vae"
            )
        vae.requires_grad_(False)
        vae.eval()
        return vae

    text_encoder, tokenizer = _load_text_encoder_and_tokenizer()
    vae = _load_vae()

    config_path = video_generation_dir / "configs" / "wan2.1_t2v_14b.json"
    with torch_default_dtype(model_dtype):
        model = WanTransformer3DModel_Refextractor_2D_Controlnet_prefix.from_config(
            str(config_path)
        ).to(model_dtype)
        model.set_up_controlnet(
            str(video_generation_dir / "configs" / "wan2.1_t2v_14b_controlnet_1.json"),
            model_dtype,
        )
        model.set_up_refextractor(
            str(video_generation_dir / "configs" / "wan2.1_t2v_14b_refextractor_2d_withmask2.json"),
            model_dtype,
        )
    model.eval()
    model.requires_grad_(False)

    transformer_df11_dir = checkpoint_dir / "transformer_df11"
    if not df11_force_on and not df11_force_off:
        if not use_df11 and transformer_df11_dir.is_dir() and df11_available():
            use_df11 = True

    use_df11_transformer = bool(use_df11 and transformer_df11_dir.is_dir())
    if not use_df11_transformer:
        load_fp8_scaled_into_module(model, checkpoint_dir, target_dtype=model_dtype, strict=True)

    pipe = WanPipeline(
        transformer=model,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )

    negative_prompt = [
        "black background, Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
    ]

    precompute_prompt_embeds = _env_bool(
        "ONE_TO_ALL_PRECOMPUTE_PROMPT_EMBEDS",
        default=bool(use_df11 and device.startswith("cuda")),
    )
    drop_text_encoder_after_prompt = _env_bool(
        "ONE_TO_ALL_DROP_TEXT_ENCODER_AFTER_PROMPT",
        default=bool(precompute_prompt_embeds and device.startswith("cuda")),
    )

    prompt_embeds = None
    negative_prompt_embeds = None
    if precompute_prompt_embeds:
        if pipe.text_encoder is not None and device.startswith("cuda"):
            pipe.text_encoder.to(device, dtype=model_dtype)
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            device=torch.device(device) if device else None,
            dtype=model_dtype,
            max_sequence_length=512,
        )
        if drop_text_encoder_after_prompt:
            try:
                if pipe.text_encoder is not None:
                    pipe.text_encoder.to("cpu")
                pipe.text_encoder = None
                pipe.tokenizer = None
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                print("[info] precomputed prompt embeds; dropped text_encoder/tokenizer to save VRAM")
            except Exception as exc:
                print(f"[warn] failed to drop text_encoder/tokenizer: {exc}")

    if use_df11_transformer:
        load_df11_into_model(pipe.transformer, transformer_df11_dir)

    def _should_enable_cpu_offload() -> bool:
        if use_df11 and os.environ.get("ONE_TO_ALL_DF11_PREFER_BLOCK_OFFLOAD", "").strip() not in {
            "0",
            "false",
            "no",
            "n",
            "off",
        }:
            return False
        if _env_flag("ONE_TO_ALL_DISABLE_CPU_OFFLOAD"):
            return False
        return device.startswith("cuda")

    if _should_enable_cpu_offload():
        gpu_id = 0
        if device.startswith("cuda") and ":" in device:
            gpu_id = int(device.split(":", 1)[1])

        if hasattr(pipe, "enable_sequential_cpu_offload") and not _env_flag(
            "ONE_TO_ALL_DISABLE_SEQUENTIAL_CPU_OFFLOAD"
        ):
            try:
                pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
                print("[info] enabled sequential CPU offload")
            except Exception as exc:
                print(f"[warn] failed to enable sequential CPU offload: {exc}")
                pipe.to(device, dtype=model_dtype)
        elif hasattr(pipe, "enable_model_cpu_offload"):
            try:
                pipe.enable_model_cpu_offload(gpu_id=gpu_id)
                print("[info] enabled model CPU offload")
            except Exception as exc:
                print(f"[warn] failed to enable model CPU offload: {exc}")
                pipe.to(device, dtype=model_dtype)
        else:
            pipe.to(device, dtype=model_dtype)
    else:
        pipe.to(device, dtype=model_dtype)

    return BuiltPipeline(
        pipe=pipe,
        model_dtype=model_dtype,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt=negative_prompt,
    )


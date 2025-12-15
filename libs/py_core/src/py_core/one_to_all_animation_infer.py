from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from py_core.one_to_all_model import resolve_one_to_all_checkpoint_dir
from py_core.settings import settings


def _find_repo_root() -> Path:
    candidates = [Path.cwd(), Path(__file__).resolve()]
    for start in candidates:
        for parent in [start, *start.parents]:
            if (parent / "apps").is_dir() and (parent / "libs").is_dir():
                return parent
    return Path.cwd()


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return _find_repo_root() / path


@contextmanager
def _third_party_video_generation_context() -> Iterator[Path]:
    third_party_root = _resolve_repo_path(settings.one_to_all_animation_dir)
    video_generation_dir = third_party_root / "video-generation"
    if not video_generation_dir.is_dir():
        raise FileNotFoundError(
            f"One-to-All-Animation video-generation dir not found: {video_generation_dir}"
        )

    runtime_root = _resolve_repo_path(settings.one_to_all_animation_runtime_dir)
    runtime_video_generation_dir = runtime_root / "video-generation"
    runtime_video_generation_dir.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    os.chdir(runtime_video_generation_dir)
    sys.path.insert(0, str(video_generation_dir))
    try:
        yield video_generation_dir
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path


def _infer_default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


@dataclass(frozen=True)
class OneToAllAnimationRunConfig:
    reference_image_path: Path
    motion_video_path: Path
    model_name: str | None = None
    prompt: str = ""
    align_mode: str = "ref"  # "ref" or "pose"
    frame_interval: int = 1
    do_align: bool = True
    face_change: bool = True
    head_change: bool = False
    without_face: bool = False
    image_guidance_scale: float = 2.0
    pose_guidance_scale: float = 1.5
    num_inference_steps: int = 30
    seed: int = 42
    black_image_cfg: bool = True
    black_pose_cfg: bool = True
    controlnet_conditioning_scale: float = 1.0
    device: str = ""


def _load_fp8_scaled_state_dict(checkpoint_dir: Path, *, target_dtype) -> dict[str, Any]:
    """
    Loads One-to-All weights from a checkpoint directory.

    Supports fp8_scaled-style weights where:
    - `<name>.weight` is FP8 (e5m2/e4m3fn)
    - `<name>.scale_weight` is FP16 scale per output row
    """

    from safetensors import safe_open
    import torch

    shard_paths: list[Path]
    if checkpoint_dir.is_file():
        shard_paths = [checkpoint_dir]
    else:
        shard_paths = sorted(checkpoint_dir.glob("*.safetensors"))
        if not shard_paths:
            raise FileNotFoundError(f"No .safetensors shards found in: {checkpoint_dir}")

    scales: dict[str, torch.Tensor] = {}
    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith(".scale_weight"):
                    scales[key] = f.get_tensor(key)

    def _apply_fp8_scale(fp8_weight: torch.Tensor, scale_weight: torch.Tensor) -> torch.Tensor:
        if fp8_weight.ndim < 2:
            raise ValueError(f"Expected fp8_weight.ndim>=2, got {fp8_weight.ndim} ({fp8_weight.shape})")
        if scale_weight.ndim != 1:
            raise ValueError(
                f"Expected scale_weight.ndim==1, got {scale_weight.ndim} ({scale_weight.shape})"
            )
        if int(scale_weight.shape[0]) != int(fp8_weight.shape[0]):
            raise ValueError(
                "scale_weight shape mismatch: "
                f"scale={tuple(scale_weight.shape)} weight={tuple(fp8_weight.shape)}"
            )
        view_shape = (int(scale_weight.shape[0]),) + (1,) * (fp8_weight.ndim - 1)
        return fp8_weight.to(torch.float16) * scale_weight.to(torch.float16).view(view_shape)

    state_dict: dict[str, torch.Tensor] = {}
    fp8_dtypes = {torch.float8_e5m2, torch.float8_e4m3fn}
    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith(".scale_weight"):
                    continue
                tensor = f.get_tensor(key)
                if tensor.dtype in fp8_dtypes and key.endswith(".weight"):
                    scale_key = key[: -len(".weight")] + ".scale_weight"
                    scale = scales.get(scale_key)
                    if scale is None:
                        raise KeyError(
                            f"Missing scale tensor for FP8 weight: {key} (expected {scale_key})"
                        )
                    tensor = _apply_fp8_scale(tensor, scale)
                    tensor = tensor.to(target_dtype)
                state_dict[key] = tensor
    return state_dict


def run_one_to_all_animation_inference(cfg: OneToAllAnimationRunConfig) -> dict[str, Any]:
    """
    Run One-to-All-Animation inference for a single (reference_image, motion_video) pair.

    Returns a dict with local output paths.
    """

    device = cfg.device or os.environ.get("ONE_TO_ALL_DEVICE") or _infer_default_device()

    checkpoint_dir = resolve_one_to_all_checkpoint_dir(cfg.model_name)
    checkpoint_dir = _resolve_repo_path(str(checkpoint_dir))
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"One-to-All checkpoint dir not found: {checkpoint_dir}")

    data_dir = _resolve_repo_path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(int(time.time() * 1000))
    out_dir = data_dir / "one_to_all" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = out_dir / "result.mp4"

    with _third_party_video_generation_context() as video_generation_dir:
        pretrained_root_candidates = [
            checkpoint_dir / "pretrained_models",
            _resolve_repo_path(settings.one_to_all_animation_pretrained_dir),
        ]
        pretrained_root = next((p for p in pretrained_root_candidates if p.is_dir()), None)
        if pretrained_root is None:
            raise FileNotFoundError(
                "Missing pretrained_models; download pretrained assets:\n"
                "  uv run --project apps/api scripts/download_one_to_all_animation_pretrained.py --with-wan-14b\n"
                f"tried={pretrained_root_candidates}"
            )

        model_path_candidates = [
            pretrained_root / "Wan2.1-T2V-14B-Diffusers",
            _resolve_repo_path(settings.one_to_all_wan_t2v_14b_diffusers_dir),
        ]
        model_path = next((p for p in model_path_candidates if p.is_dir()), None)
        if model_path is None:
            raise FileNotFoundError(
                "Missing Wan2.1 Diffusers base model dir; download pretrained assets:\n"
                "  uv run --project apps/api scripts/download_one_to_all_animation_pretrained.py --with-wan-14b\n"
                f"tried={model_path_candidates}"
            )
        if not model_path.is_dir():
            raise FileNotFoundError(
                "Missing Wan2.1 Diffusers base model dir; download pretrained assets:\n"
                "  uv run --project apps/api scripts/download_one_to_all_animation_pretrained.py --with-wan-14b\n"
                f"expected_dir={model_path}"
            )

        pose2d_ckpt = pretrained_root / "process_checkpoint" / "pose2d" / "vitpose_h_wholebody.onnx"
        det_ckpt = pretrained_root / "process_checkpoint" / "det" / "yolov10m.onnx"
        dwpose_dir = pretrained_root / "DWPose"

        def _onnx_checkpoint_exists(path: Path) -> bool:
            if path.is_file():
                return True
            if path.is_dir():
                # Some downloads store the checkpoint as a directory containing `end2end.onnx`.
                return (path / "end2end.onnx").is_file()
            return False

        if not _onnx_checkpoint_exists(pose2d_ckpt) or not _onnx_checkpoint_exists(det_ckpt):
            raise FileNotFoundError(
                "Missing pose preprocess checkpoints; download pretrained assets:\n"
                "  uv run --project apps/api scripts/download_one_to_all_animation_pretrained.py\n"
                f"expected_files={[str(pose2d_ckpt), str(det_ckpt)]}"
            )
        if not dwpose_dir.is_dir():
            raise FileNotFoundError(
                "Missing DWPose assets; download pretrained assets:\n"
                "  uv run --project apps/api scripts/download_one_to_all_animation_pretrained.py\n"
                f"expected_dir={dwpose_dir}"
            )

        import numpy as np
        import torch
        from PIL import Image

        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from opensora.sample.pipeline_wanx_vhuman_tokenreplace import WanPipeline
        from opensora.model_variants.wanx_diffusers_src import (
            WanTransformer3DModel_Refextractor_2D_Controlnet_prefix,
        )

        from infer_utils import load_poses_whole_video, resizecrop

        # Build model/pipeline
        model_dtype = torch.bfloat16
        config_path = video_generation_dir / "configs" / "wan2.1_t2v_14b.json"

        scheduler = FlowMatchEulerDiscreteScheduler(
            shift=7.0, num_train_timesteps=1000, use_dynamic_shifting=False
        )
        from transformers import AutoTokenizer, AutoConfig, UMT5EncoderModel
        from diffusers.models import AutoencoderKLWan

        def _has_fp8_scaled_weights(path: Path) -> bool:
            from safetensors import safe_open

            st_paths = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
            for p in st_paths:
                if not p.is_file():
                    continue
                with safe_open(str(p), framework="pt", device="cpu") as f:
                    for k in f.keys():
                        if k.endswith(".scale_weight"):
                            return True
            return False

        def _load_text_encoder_and_tokenizer(wan_dir: Path):
            tokenizer = AutoTokenizer.from_pretrained(str(wan_dir), subfolder="tokenizer")
            te_dir = wan_dir / "text_encoder"
            if _has_fp8_scaled_weights(te_dir):
                te_config = AutoConfig.from_pretrained(str(wan_dir), subfolder="text_encoder")
                text_encoder = UMT5EncoderModel(te_config)
                te_state = _load_fp8_scaled_state_dict(te_dir, target_dtype=model_dtype)
                missing, unexpected = text_encoder.load_state_dict(te_state, strict=False)
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

        def _load_vae(wan_dir: Path):
            vae_dir = wan_dir / "vae"
            if _has_fp8_scaled_weights(vae_dir):
                vae_config = AutoencoderKLWan.load_config(str(wan_dir), subfolder="vae")
                vae = AutoencoderKLWan.from_config(vae_config).to(dtype=model_dtype)
                vae_state = _load_fp8_scaled_state_dict(vae_dir, target_dtype=model_dtype)
                missing, unexpected = vae.load_state_dict(vae_state, strict=False)
                if missing:
                    raise KeyError(f"FP8 VAE missing keys (count={len(missing)}), first={missing[:5]}")
                if unexpected:
                    raise KeyError(
                        f"FP8 VAE unexpected keys (count={len(unexpected)}), first={unexpected[:5]}"
                    )
            else:
                vae = AutoencoderKLWan.from_pretrained(str(wan_dir), torch_dtype=model_dtype, subfolder="vae")
            vae.requires_grad_(False)
            vae.eval()
            return vae

        text_encoder, tokenizer = _load_text_encoder_and_tokenizer(model_path)
        vae = _load_vae(model_path)

        model = (
            WanTransformer3DModel_Refextractor_2D_Controlnet_prefix.from_config(
                str(config_path)
            ).to(model_dtype)
        )
        model.set_up_controlnet(
            str(video_generation_dir / "configs" / "wan2.1_t2v_14b_controlnet_1.json"),
            model_dtype,
        )
        model.set_up_refextractor(
            str(
                video_generation_dir
                / "configs"
                / "wan2.1_t2v_14b_refextractor_2d_withmask2.json"
            ),
            model_dtype,
        )
        model.eval()
        model.requires_grad_(False)

        state_dict = _load_fp8_scaled_state_dict(checkpoint_dir, target_dtype=model_dtype)
        model.load_state_dict(state_dict, strict=True)

        pipe = WanPipeline(
            transformer=model,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )

        def _env_flag(name: str) -> bool:
            v = os.environ.get(name, "").strip().lower()
            return v in {"1", "true", "yes", "y", "on"}

        def _should_enable_cpu_offload() -> bool:
            if _env_flag("ONE_TO_ALL_DISABLE_CPU_OFFLOAD"):
                return False
            if _env_flag("ONE_TO_ALL_ENABLE_CPU_OFFLOAD"):
                return True
            try:
                if device.startswith("cuda"):
                    gpu_id = 0
                    if ":" in device:
                        gpu_id = int(device.split(":", 1)[1])
                    total = int(torch.cuda.get_device_properties(gpu_id).total_memory)
                    return total <= 36 * 1024**3
            except Exception:
                pass
            return False

        if _should_enable_cpu_offload() and hasattr(pipe, "enable_model_cpu_offload"):
            try:
                gpu_id = 0
                if device.startswith("cuda") and ":" in device:
                    gpu_id = int(device.split(":", 1)[1])
                pipe.enable_model_cpu_offload(gpu_id=gpu_id)
                print("[info] enabled model CPU offload")
            except Exception as exc:
                print(f"[warn] failed to enable model CPU offload: {exc}")
                pipe.to(device, dtype=model_dtype)
        else:
            pipe.to(device, dtype=model_dtype)

        # Preprocess inputs (pose extraction etc.)
        ref_img_tmp = Image.open(cfg.reference_image_path).convert("RGB")
        w_ref, h_ref = ref_img_tmp.size
        h, w = h_ref, w_ref
        max_short = 768
        if min(h, w) > max_short:
            if h < w:
                scale = max_short / h
                h, w = max_short, int(w * scale)
            else:
                scale = max_short / w
                w, h = max_short, int(h * scale)
        new_h = (h // 16) * 16
        new_w = (w // 16) * 16

        def transform(pil_img: Image.Image) -> Image.Image:
            return resizecrop(pil_img, th=new_h, tw=new_w)

        pose_tensor_u8, image_input, pose_input, mask_input = load_poses_whole_video(
            video_path=str(cfg.motion_video_path),
            reference=str(cfg.reference_image_path),
            frame_interval=max(1, int(cfg.frame_interval)),
            transform=transform,
            do_align=cfg.do_align,
            alignmode=cfg.align_mode,
            face_change=cfg.face_change,
            head_change=cfg.head_change,
            without_face=cfg.without_face,
            anchor_idx=0,
        )

        try:
            import decord

            vr = decord.VideoReader(str(cfg.motion_video_path))
            fps = float(vr.get_avg_fps() if vr.get_avg_fps() > 0 else 30)
        except Exception:
            fps = 30.0
        output_fps = fps / max(1, int(cfg.frame_interval))

        # Convert pose control video tensor to [-1, 1] range, shape [1,C,T,H,W]
        pose_tensor = (
            pose_tensor_u8.float().permute(1, 0, 2, 3).unsqueeze(0) / 255.0 * 2 - 1
        )

        # Prepare mask tensor: [1,1,1,H,W]
        mask_l = mask_input.convert("L")
        mask_np = np.array(mask_l, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).unsqueeze(2)

        # Prepare source pose image tensor: [1,C,1,H,W]
        src_pose_tensor = (
            torch.from_numpy(np.array(pose_input))
            .unsqueeze(0)
            .float()
            .permute(0, 3, 1, 2)
            / 255.0
            * 2
            - 1
        ).unsqueeze(2)

        # Split plan (mirrors upstream defaults)
        main_chunk = 81
        overlap_frames = 5
        final_chunk_candidates = [65, 69, 73, 77, 81]

        def build_split_plan(total_len: int) -> list[tuple[int, int]]:
            ranges: list[tuple[int, int]] = []
            start = 0
            while True:
                current_chunk_end = start + main_chunk
                next_chunk_start = start + (main_chunk - overlap_frames)
                if next_chunk_start + main_chunk >= total_len:
                    ranges.append((start, current_chunk_end))
                    final_chunk_start = -1
                    for length in final_chunk_candidates:
                        potential_start = total_len - length
                        if potential_start < current_chunk_end - overlap_frames:
                            final_chunk_start = potential_start
                            break
                    if final_chunk_start == -1:
                        final_chunk_start = next_chunk_start
                    ranges.append((final_chunk_start, total_len))
                    break
                ranges.append((start, current_chunk_end))
                start = next_chunk_start
            return ranges

        negative_prompt = [
            "black background, Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
        ]

        split_plan = build_split_plan(int(pose_tensor.shape[2]))
        all_frames: dict[int, np.ndarray] = {}

        for start, end in split_plan:
            sub_video = pose_tensor[:, :, start:end].to(device)  # [1,C,T,H,W]
            prev_frames = None
            if start > 0:
                needed_idx = range(start, start + overlap_frames)
                if all(i in all_frames for i in needed_idx):
                    prev_frames = [
                        Image.fromarray(all_frames[i]) for i in needed_idx  # type: ignore[arg-type]
                    ]

            output = pipe(
                image=image_input,
                image_mask=mask_tensor.to(device),
                control_video=sub_video,
                prompt=cfg.prompt,
                negative_prompt=negative_prompt,
                height=new_h,
                width=new_w,
                num_frames=end - start,
                image_guidance_scale=float(cfg.image_guidance_scale),
                pose_guidance_scale=float(cfg.pose_guidance_scale),
                num_inference_steps=int(cfg.num_inference_steps),
                generator=torch.Generator(device=device).manual_seed(int(cfg.seed)),
                black_image_cfg=cfg.black_image_cfg,
                black_pose_cfg=cfg.black_pose_cfg,
                controlnet_conditioning_scale=float(cfg.controlnet_conditioning_scale),
                return_tensor=True,
                case1=False,
                token_replace=(prev_frames is not None),
                prev_frames=prev_frames,
                image_pose=src_pose_tensor.to(device),
            ).frames

            chunk_np = (
                (output[0].detach().float().cpu() / 2 + 0.5)
                .clamp(0, 1)
                .permute(1, 2, 3, 0)
                .numpy()
            )
            chunk_np = (chunk_np * 255).astype("uint8")
            for j in range(end - start):
                all_frames[start + j] = chunk_np[j]

        sorted_idx = sorted(all_frames.keys())
        frames = [all_frames[i] for i in sorted_idx]
        import imageio

        imageio.mimwrite(str(output_video_path), frames, fps=output_fps, quality=5)

        return {
            "run_id": run_id,
            "local_output_dir": str(out_dir),
            "local_video_path": str(output_video_path),
            "model_checkpoint_dir": str(checkpoint_dir),
            "device": device,
            "output_fps": output_fps,
            "num_frames": len(frames),
        }

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from py_core.one_to_all_animation.onnx import patch_onnxruntime_for_pose
from py_core.one_to_all_animation.paths import resolve_repo_path
from py_core.one_to_all_animation.pipeline import build_pipeline
from py_core.one_to_all_animation.pose import load_poses_whole_video, resizecrop
from py_core.one_to_all_animation.split_plan import build_split_plan
from py_core.one_to_all_animation.third_party import video_generation_context
from py_core.one_to_all_animation.torch_utils import infer_default_device
from py_core.one_to_all_model import resolve_one_to_all_checkpoint_dir
from py_core.settings import settings


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


def _onnx_checkpoint_exists(path: Path) -> bool:
    if path.is_file():
        return True
    if path.is_dir():
        return (path / "end2end.onnx").is_file()
    return False


def _env_int_or_none(name: str) -> int | None:
    v = os.environ.get(name)
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    return int(v)


def run_one_to_all_animation_inference(cfg: OneToAllAnimationRunConfig) -> dict[str, Any]:
    """
    Run One-to-All-Animation inference for a single (reference_image, motion_video) pair.

    Returns a dict with local output paths.
    """

    device = cfg.device or os.environ.get("ONE_TO_ALL_DEVICE") or infer_default_device()

    checkpoint_dir = resolve_one_to_all_checkpoint_dir(cfg.model_name)
    checkpoint_dir = resolve_repo_path(str(checkpoint_dir))
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"One-to-All checkpoint dir not found: {checkpoint_dir}")

    df11_env = os.environ.get("ONE_TO_ALL_USE_DF11", "").strip().lower()
    df11_force_on = df11_env in {"1", "true", "yes", "y", "on"}
    df11_force_off = df11_env in {"0", "false", "no", "n", "off"}
    use_df11 = bool(df11_force_on)
    if df11_force_off or not device.startswith("cuda"):
        use_df11 = False

    data_dir = resolve_repo_path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(int(time.time() * 1000))
    out_dir = data_dir / "one_to_all" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = out_dir / "result.mp4"

    with video_generation_context() as (video_generation_dir, runtime_root):
        pretrained_root_candidates = [
            checkpoint_dir / "pretrained_models",
            resolve_repo_path(settings.one_to_all_animation_pretrained_dir),
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
            resolve_repo_path(settings.one_to_all_wan_t2v_14b_diffusers_dir),
        ]
        wan_dir = next((p for p in model_path_candidates if p.is_dir()), None)
        if wan_dir is None:
            raise FileNotFoundError(
                "Missing Wan2.1 Diffusers base model dir; download pretrained assets:\n"
                "  uv run --project apps/api scripts/download_one_to_all_animation_pretrained.py --with-wan-14b\n"
                f"tried={model_path_candidates}"
            )

        pose2d_ckpt = (
            pretrained_root
            / "process_checkpoint"
            / "pose2d"
            / "vitpose_h_wholebody.onnx"
        )
        det_ckpt = pretrained_root / "process_checkpoint" / "det" / "yolov10m.onnx"
        if not _onnx_checkpoint_exists(pose2d_ckpt) or not _onnx_checkpoint_exists(det_ckpt):
            raise FileNotFoundError(
                "Missing pose preprocess checkpoints; download pretrained assets:\n"
                "  uv run --project apps/api scripts/download_one_to_all_animation_pretrained.py\n"
                f"expected_files={[str(pose2d_ckpt), str(det_ckpt)]}"
            )

        patch_onnxruntime_for_pose(runtime_root)

        built = build_pipeline(
            video_generation_dir=video_generation_dir,
            checkpoint_dir=checkpoint_dir,
            wan_dir=wan_dir,
            device=device,
            use_df11_initial=use_df11,
            df11_force_on=df11_force_on,
            df11_force_off=df11_force_off,
            prompt=cfg.prompt,
        )
        pipe = built.pipe

        import numpy as np
        import torch
        from PIL import Image

        ref_img_tmp = Image.open(cfg.reference_image_path).convert("RGB")
        w_ref, h_ref = ref_img_tmp.size
        h, w = h_ref, w_ref

        max_short_env = os.environ.get("ONE_TO_ALL_MAX_SHORT")
        max_short = int(max_short_env) if max_short_env else 576

        max_long = _env_int_or_none("ONE_TO_ALL_MAX_LONG")
        if max_long is None:
            try:
                if device.startswith("cuda"):
                    gpu_id = 0
                    if ":" in device:
                        gpu_id = int(device.split(":", 1)[1])
                    total = int(torch.cuda.get_device_properties(gpu_id).total_memory)
                    if total <= 36 * 1024**3:
                        max_long = 1024
            except Exception:
                max_long = None

        if min(h, w) > max_short:
            if h < w:
                scale = max_short / h
                h, w = max_short, int(w * scale)
            else:
                scale = max_short / w
                w, h = max_short, int(h * scale)
        if max_long is not None and max(h, w) > max_long:
            if h > w:
                scale = max_long / h
                h, w = max_long, int(w * scale)
            else:
                scale = max_long / w
                w, h = max_long, int(h * scale)
        new_h = (h // 16) * 16
        new_w = (w // 16) * 16

        max_long_str = str(max_long) if max_long is not None else "unset"
        print(
            "[info] resize ref "
            f"{w_ref}x{h_ref} -> {new_w}x{new_h} "
            f"(max_short={max_short}, max_long={max_long_str})"
        )
        if new_h != h or new_w != w:
            print(f"[info] rounded to multiples of 16: {w}x{h} -> {new_w}x{new_h}")

        def transform(pil_img: Image.Image) -> Image.Image:
            return resizecrop(pil_img, th=new_h, tw=new_w)

        pose_tensor_u8, image_input, pose_input, mask_input = load_poses_whole_video(
            video_path=str(cfg.motion_video_path),
            reference=str(cfg.reference_image_path),
            pose2d_checkpoint_path=pose2d_ckpt,
            det_checkpoint_path=det_ckpt,
            device=device,
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

        pose_tensor = (
            pose_tensor_u8.float().permute(1, 0, 2, 3).unsqueeze(0) / 255.0 * 2 - 1
        )

        mask_l = mask_input.convert("L")
        mask_np = np.array(mask_l, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).unsqueeze(2)

        src_pose_tensor = (
            torch.from_numpy(np.array(pose_input))
            .unsqueeze(0)
            .float()
            .permute(0, 3, 1, 2)
            / 255.0
            * 2
            - 1
        ).unsqueeze(2)

        total_len = int(pose_tensor.shape[2])
        debug_max_frames = os.environ.get("ONE_TO_ALL_DEBUG_MAX_FRAMES", "").strip()
        if debug_max_frames:
            max_frames = max(0, int(debug_max_frames))
            if 0 < max_frames < total_len:
                pose_tensor = pose_tensor[:, :, :max_frames]
                total_len = int(pose_tensor.shape[2])

        split = build_split_plan(total_frames=total_len, device=device)
        split_plan = list(split.plan)

        debug_max_chunks = os.environ.get("ONE_TO_ALL_DEBUG_MAX_CHUNKS", "").strip()
        if debug_max_chunks:
            max_chunks = max(0, int(debug_max_chunks))
            if max_chunks > 0:
                split_plan = split_plan[:max_chunks]

        all_frames: dict[int, np.ndarray] = {}
        print(
            f"[info] split plan: total_frames={total_len}, main_chunk={split.main_chunk}, "
            f"overlap_frames={split.overlap_frames}, chunks={len(split_plan)}"
        )
        if debug_max_frames:
            print(f"[info] debug ONE_TO_ALL_DEBUG_MAX_FRAMES={debug_max_frames}")
        if debug_max_chunks:
            print(f"[info] debug ONE_TO_ALL_DEBUG_MAX_CHUNKS={debug_max_chunks}")

        for start, end in split_plan:
            end = min(int(end), total_len)
            start = max(int(start), 0)
            if end <= start:
                continue

            sub_video = pose_tensor[:, :, start:end]
            prev_frames = None
            if start > 0:
                needed_idx = range(start, start + split.overlap_frames)
                if all(i in all_frames for i in needed_idx):
                    prev_frames = [Image.fromarray(all_frames[i]) for i in needed_idx]

            output = pipe(
                image=image_input,
                image_mask=mask_tensor,
                control_video=sub_video,
                prompt=None if built.prompt_embeds is not None else cfg.prompt,
                negative_prompt=None
                if built.negative_prompt_embeds is not None
                else built.negative_prompt,
                prompt_embeds=built.prompt_embeds,
                negative_prompt_embeds=built.negative_prompt_embeds,
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
                image_pose=src_pose_tensor,
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


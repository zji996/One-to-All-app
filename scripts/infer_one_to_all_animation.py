from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from py_core.one_to_all_animation_infer import OneToAllAnimationRunConfig, run_one_to_all_animation_inference


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_dir_from_repo_root(repo_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run One-to-All-Animation inference for a single pair of inputs.\n\n"
            "Run via worker env:\n"
            "  uv sync --project apps/worker --extra one_to_all_animation\n"
            "  uv run --project apps/worker scripts/infer_one_to_all_animation.py \\\n"
            "    --reference-image third_party/One-to-All-Animation/examples/img.png \\\n"
            "    --motion-video third_party/One-to-All-Animation/examples/vid.mp4\n"
        )
    )
    parser.add_argument("--reference-image", required=True, help="Local image path.")
    parser.add_argument("--motion-video", required=True, help="Local video path.")
    parser.add_argument(
        "--model-name",
        default=os.environ.get("ONE_TO_ALL_MODEL_NAME") or None,
        help="Optional alias: 14b_fp8 / 14b / 1.3b_1 / 1.3b_2 (default: env/Settings).",
    )
    parser.add_argument("--prompt", default="", help="Optional text prompt.")
    parser.add_argument("--align-mode", default="ref", help="ref | pose (default: ref).")
    parser.add_argument("--frame-interval", type=int, default=1, help="Sample every N frames (>=1).")
    parser.add_argument("--no-align", action="store_true", help="Disable pose retargeting.")
    parser.add_argument("--image-guidance-scale", type=float, default=2.0)
    parser.add_argument("--pose-guidance-scale", type=float, default=1.5)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=os.environ.get("ONE_TO_ALL_DEVICE") or "", help="e.g. cuda:0 / cpu")
    args = parser.parse_args()

    repo_root = _repo_root()
    cfg = OneToAllAnimationRunConfig(
        reference_image_path=_resolve_dir_from_repo_root(repo_root, args.reference_image),
        motion_video_path=_resolve_dir_from_repo_root(repo_root, args.motion_video),
        model_name=args.model_name,
        prompt=args.prompt,
        align_mode=args.align_mode,
        frame_interval=max(1, int(args.frame_interval)),
        do_align=not args.no_align,
        image_guidance_scale=float(args.image_guidance_scale),
        pose_guidance_scale=float(args.pose_guidance_scale),
        num_inference_steps=int(args.num_inference_steps),
        seed=int(args.seed),
        device=str(args.device),
    )

    result = run_one_to_all_animation_inference(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


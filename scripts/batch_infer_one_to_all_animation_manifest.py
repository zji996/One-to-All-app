from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from py_core.one_to_all_animation_infer import OneToAllAnimationRunConfig, run_one_to_all_animation_inference


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_from_repo_root(repo_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def _default_manifest() -> str:
    data_dir = os.environ.get("DATA_DIR") or "data"
    return str(Path(data_dir) / "testsets" / "one_to_all_animation" / "new_examples" / "manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run One-to-All-Animation inference on a manifest produced by "
            "`scripts/prepare_one_to_all_animation_new_examples.py`.\n\n"
            "Run via worker env:\n"
            "  uv sync --project apps/worker --extra one_to_all_animation\n"
            "  uv run --project apps/worker scripts/batch_infer_one_to_all_animation_manifest.py\n"
        )
    )
    parser.add_argument("--manifest", default=_default_manifest(), help="Path to manifest.json.")
    parser.add_argument(
        "--model-name",
        default=os.environ.get("ONE_TO_ALL_MODEL_NAME") or None,
        help="Optional alias: 14b_fp8 / 14b / 1.3b_1 / 1.3b_2 (default: env/Settings).",
    )
    parser.add_argument("--prompt", default="", help="Optional text prompt.")
    parser.add_argument("--align-mode", default="ref", help="ref | pose (default: ref).")
    parser.add_argument("--frame-interval", type=int, default=2, help="Sample every N frames (>=1).")
    parser.add_argument("--no-align", action="store_true", help="Disable pose retargeting.")
    parser.add_argument("--image-guidance-scale", type=float, default=2.0)
    parser.add_argument("--pose-guidance-scale", type=float, default=1.5)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=os.environ.get("ONE_TO_ALL_DEVICE") or "", help="e.g. cuda:0 / cpu")
    parser.add_argument("--limit", type=int, default=0, help="If >0, only run first N cases.")
    args = parser.parse_args()

    repo_root = _repo_root()
    manifest_path = _resolve_from_repo_root(repo_root, str(args.manifest))
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent

    cases = manifest.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"invalid manifest (missing cases): {manifest_path}")

    limit = int(args.limit or 0)
    if limit > 0:
        cases = cases[:limit]

    started_ms = int(time.time() * 1000)
    results: list[dict[str, Any]] = []
    for i, c in enumerate(cases, start=1):
        if not isinstance(c, dict):
            raise ValueError(f"invalid case type: {type(c)}")
        ref_rel = c.get("reference_image")
        motion_rel = c.get("motion_video")
        if not isinstance(ref_rel, str) or not ref_rel:
            raise ValueError(f"invalid case reference_image: {c}")
        if not isinstance(motion_rel, str) or not motion_rel:
            raise ValueError(f"invalid case motion_video: {c}")

        cfg = OneToAllAnimationRunConfig(
            reference_image_path=base_dir / ref_rel,
            motion_video_path=base_dir / motion_rel,
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

        print(f"[run] {i}/{len(cases)} id={c.get('id')}")
        r = run_one_to_all_animation_inference(cfg)
        r["case"] = c
        results.append(r)

    out = {
        "manifest": str(manifest_path),
        "started_at_ms": started_ms,
        "finished_at_ms": int(time.time() * 1000),
        "results": results,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


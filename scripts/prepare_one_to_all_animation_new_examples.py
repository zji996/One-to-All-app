from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_from_repo_root(repo_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def _default_dst_dir() -> str:
    data_dir = os.environ.get("DATA_DIR") or "data"
    return str(Path(data_dir) / "testsets" / "one_to_all_animation" / "new_examples")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy `third_party/One-to-All-Animation/examples/new_examples` into DATA_DIR for local smoke tests.\n"
            "This keeps `third_party/` read-only.\n\n"
            "Run via worker env:\n"
            "  uv run --project apps/worker scripts/prepare_one_to_all_animation_new_examples.py\n"
        )
    )
    parser.add_argument(
        "--src",
        default="third_party/One-to-All-Animation/examples/new_examples",
        help="Source examples dir (default: third_party/One-to-All-Animation/examples/new_examples).",
    )
    parser.add_argument(
        "--dst",
        default=_default_dst_dir(),
        help="Destination dir under DATA_DIR (default: data/testsets/one_to_all_animation/new_examples).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    parser.add_argument(
        "--motion",
        default="",
        help="Motion video filename within src (default: first *.mp4 found).",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    src_dir = _resolve_from_repo_root(repo_root, str(args.src))
    dst_dir = _resolve_from_repo_root(repo_root, str(args.dst))

    if not src_dir.is_dir():
        raise FileNotFoundError(f"source dir not found: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    )
    mp4s = sorted([p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"])
    if not images:
        raise FileNotFoundError(f"no images found under: {src_dir}")
    if not mp4s:
        raise FileNotFoundError(f"no .mp4 found under: {src_dir}")

    if args.motion:
        motion_src = src_dir / str(args.motion)
        if not motion_src.is_file():
            raise FileNotFoundError(f"--motion not found: {motion_src}")
    else:
        motion_src = mp4s[0]

    def _copy_one(p: Path) -> None:
        target = dst_dir / p.name
        if target.exists() and not args.overwrite:
            return
        shutil.copy2(p, target)

    for p in images:
        _copy_one(p)
    _copy_one(motion_src)

    cases: list[dict[str, str]] = []
    for p in images:
        cases.append(
            {
                "id": p.stem,
                "reference_image": p.name,
                "motion_video": motion_src.name,
            }
        )

    manifest = {
        "name": "one_to_all_animation_new_examples",
        "created_at_ms": int(time.time() * 1000),
        "source_dir": str(src_dir),
        "dest_dir": str(dst_dir),
        "cases": cases,
    }
    (dst_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_from_repo_root(repo_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def _copy_file(src: Path, dst: Path, *, strategy: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if strategy == "symlink":
        dst.symlink_to(src)
        return

    if strategy == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass

    shutil.copy2(src, dst)


def _sync_tree(
    src_root: Path,
    dst_root: Path,
    *,
    strategy: str,
    dry_run: bool,
    exclude_wan_dirs: set[str] | None = None,
) -> int:
    if not src_root.is_dir():
        raise FileNotFoundError(f"source dir not found: {src_root}")
    dst_root.mkdir(parents=True, exist_ok=True)

    exclude_wan_dirs = exclude_wan_dirs or set()

    copied = 0
    for dirpath, dirnames, filenames in os.walk(src_root):
        dirpath = Path(dirpath)
        rel = dirpath.relative_to(src_root)

        # Skip temporary model download artifacts.
        dirnames[:] = [d for d in dirnames if d not in {"._____temp", ".git"}]
        if rel.parts and any(p in {"._____temp", ".git"} for p in rel.parts):
            continue

        # If we are copying `pretrained_models/`, keep Wan Diffusers slim by default.
        # Example: `Wan2.1-T2V-14B-Diffusers/transformer` is redundant for One-to-All finetuned checkpoints.
        if len(rel.parts) >= 2 and rel.parts[0].startswith("Wan") and rel.parts[1] in exclude_wan_dirs:
            dirnames[:] = []
            continue

        target_dir = dst_root / rel
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)

        for name in filenames:
            if name in {".DS_Store"}:
                continue
            src = dirpath / name
            dst = target_dir / name
            if dst.exists():
                continue
            copied += 1
            if dry_run:
                continue
            _copy_file(src, dst, strategy=strategy)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a complete local One-to-All-14b model repository by copying required pretrained assets\n"
            "(Wan2.1 Diffusers base model + pose preprocess checkpoints) into `models/One-to-All-14b/pretrained_models`.\n\n"
            "Run via API env (recommended):\n"
            "  uv run --project apps/api scripts/prepare_one_to_all_14b_model_repo.py\n"
        )
    )
    parser.add_argument(
        "--source-pretrained-dir",
        default=os.environ.get("ONE_TO_ALL_ANIMATION_PRETRAINED_DIR")
        or "models/One-to-All-Animation/pretrained_models",
        help="Source directory containing pretrained assets (default: models/One-to-All-Animation/pretrained_models).",
    )
    parser.add_argument(
        "--target-pretrained-dir",
        default=os.environ.get("ONE_TO_ALL_PRETRAINED_DIR")
        or os.environ.get("ONE_TO_ALL_ANIMATION_PRETRAINED_DIR")
        or "models/One-to-All-14b/pretrained_models",
        help="Target directory to populate (default: models/One-to-All-14b/pretrained_models).",
    )
    parser.add_argument(
        "--strategy",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="How to place files: copy | hardlink | symlink (default: copy).",
    )
    parser.add_argument(
        "--include-wan-transformer",
        action="store_true",
        help="Include Wan Diffusers `transformer/` weights when copying (usually unnecessary for One-to-All-14b).",
    )
    parser.add_argument(
        "--include-wan-assets",
        action="store_true",
        help="Include Wan Diffusers `assets/` and `examples/` when copying (usually unnecessary).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be done.")
    args = parser.parse_args()

    repo_root = _repo_root()
    src_root = _resolve_from_repo_root(repo_root, str(args.source_pretrained_dir))
    dst_root = _resolve_from_repo_root(repo_root, str(args.target_pretrained_dir))

    def _pretty_path(path: Path) -> str:
        try:
            return str(path.relative_to(repo_root))
        except Exception:
            return str(path)

    if args.dry_run:
        print(f"[dry-run] source={src_root}")
        print(f"[dry-run] target={dst_root}")

    if not src_root.is_dir():
        print(f"[skip] source not found: {src_root}")
        print(
            "[hint] If you already downloaded pretrained assets elsewhere, pass --source-pretrained-dir.\n"
            "       Otherwise, download directly into the target with:\n"
            "         uv run --project apps/api scripts/download_one_to_all_animation_pretrained.py --with-wan-14b"
        )
        return 0

    exclude_wan_dirs: set[str] = set()
    if not args.include_wan_transformer:
        exclude_wan_dirs.add("transformer")
    if not args.include_wan_assets:
        exclude_wan_dirs.update({"assets", "examples"})

    copied = _sync_tree(
        src_root,
        dst_root,
        strategy=str(args.strategy),
        dry_run=bool(args.dry_run),
        exclude_wan_dirs=exclude_wan_dirs,
    )

    print("[done] ok")
    print(f"[done] copied_files={copied}")
    print(f"[done] target_pretrained_dir={dst_root}")
    print(
        "[env] Suggested variables:\n"
        f"  ONE_TO_ALL_PRETRAINED_DIR={_pretty_path(dst_root)}\n"
        f"  WAN_T2V_14B_DIFFUSERS_DIR={_pretty_path(dst_root / 'Wan2.1-T2V-14B-Diffusers')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

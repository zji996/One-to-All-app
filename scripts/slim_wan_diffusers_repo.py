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


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            fp = Path(dirpath) / name
            try:
                total += fp.stat().st_size
            except FileNotFoundError:
                pass
    return total


def _fmt_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TiB"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Slim a local Wan Diffusers repo by removing unneeded folders like `transformer/`.\n\n"
            "Run via API env:\n"
            "  uv run --project apps/api scripts/slim_wan_diffusers_repo.py --apply\n"
        )
    )
    parser.add_argument(
        "--wan-dir",
        default=os.environ.get("WAN_T2V_14B_DIFFUSERS_DIR")
        or "models/One-to-All-14b/pretrained_models/Wan2.1-T2V-14B-Diffusers",
        help="Path to Wan Diffusers folder (default: WAN_T2V_14B_DIFFUSERS_DIR or models/One-to-All-14b/... ).",
    )
    parser.add_argument("--apply", action="store_true", help="Actually delete; otherwise dry-run.")
    parser.add_argument(
        "--remove",
        action="append",
        default=["transformer", "assets", "examples", "._____temp"],
        help="Folder name to remove (repeatable). Default: transformer, assets, examples, ._____temp",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    wan_dir = _resolve_from_repo_root(repo_root, str(args.wan_dir))
    if not wan_dir.is_dir():
        raise SystemExit(f"Wan dir not found: {wan_dir}")

    required = ["vae", "text_encoder", "tokenizer", "scheduler"]
    missing = [name for name in required if not (wan_dir / name).exists()]
    if missing:
        print(f"[warn] missing expected folders in {wan_dir}: {missing}")

    candidates: list[Path] = []
    for name in args.remove:
        p = wan_dir / name
        if p.exists():
            candidates.append(p)

    if not candidates:
        print("[skip] nothing to remove")
        return 0

    for p in candidates:
        size = _dir_size_bytes(p) if p.is_dir() else p.stat().st_size
        print(f"[plan] remove={p} size~={_fmt_bytes(size)}")

    if not args.apply:
        print("[dry-run] pass --apply to delete")
        return 0

    for p in candidates:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink(missing_ok=True)
        print(f"[done] removed={p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

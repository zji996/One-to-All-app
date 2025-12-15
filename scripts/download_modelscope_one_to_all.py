from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

from py_core.one_to_all_model import (
    resolve_one_to_all_model_dir_name,
    resolve_one_to_all_repo_id,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_dir_from_repo_root(repo_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def _dir_has_any_content(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return True
    return any(path.iterdir())


def _is_empty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and not any(path.iterdir())


def _call_snapshot_download(repo_id: str, *, target_dir: Path, revision: str | None) -> str:
    from modelscope.hub.snapshot_download import snapshot_download

    sig = inspect.signature(snapshot_download)
    kwargs: dict[str, object] = {}

    if "revision" in sig.parameters and revision:
        kwargs["revision"] = revision

    if "local_dir" in sig.parameters:
        kwargs["local_dir"] = str(target_dir)
        if "local_dir_use_symlinks" in sig.parameters:
            kwargs["local_dir_use_symlinks"] = False
        return str(snapshot_download(repo_id, **kwargs))

    cache_dir = target_dir.parent / ".modelscope_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if "cache_dir" in sig.parameters:
        kwargs["cache_dir"] = str(cache_dir)

    downloaded = Path(snapshot_download(repo_id, **kwargs))
    if target_dir.exists():
        if target_dir.is_symlink():
            return str(downloaded)
        if _is_empty_dir(target_dir):
            target_dir.rmdir()
        else:
            raise RuntimeError(
                f"Target dir already exists ({target_dir}), but ModelScope downloaded to {downloaded}; "
                "please remove the target dir or set ONE_TO_ALL_MODEL_DIR to the returned path."
            )

    try:
        target_dir.symlink_to(downloaded, target_is_directory=True)
    except Exception:
        target_dir.mkdir(parents=True, exist_ok=True)
        raise RuntimeError(
            f"ModelScope downloaded to {downloaded}, but could not create symlink at {target_dir}; "
            "please set ONE_TO_ALL_MODEL_DIR to the returned path."
        ) from None

    return str(downloaded)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download One-to-All checkpoints from ModelScope into MODELS_DIR.\n\n"
            "Run via API env:\n"
            "  uv sync --project apps/api --extra model_download\n"
            "  uv run --project apps/api scripts/download_modelscope_one_to_all.py --model 14b\n"
        )
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ONE_TO_ALL_MODEL_NAME") or "14b",
        help="Model alias: 14b / 1.3b_1 / 1.3b_2 (default: 14b).",
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("ONE_TO_ALL_MODEL_REPO_ID") or None,
        help="Override ModelScope repo id (e.g. MochunniaN1/One-to-All-14b).",
    )
    parser.add_argument(
        "--target-subdir",
        default=os.environ.get("ONE_TO_ALL_MODEL_DIR") or None,
        help="Override target dir under MODELS_DIR (default derived from --model).",
    )
    parser.add_argument(
        "--revision",
        default=os.environ.get("ONE_TO_ALL_MODEL_REVISION") or None,
        help="Optional revision/tag.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check if target dir has content; no downloads.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow download even if target dir has content.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    models_dir = _resolve_dir_from_repo_root(repo_root, os.environ.get("MODELS_DIR", "models"))
    target_subdir = args.target_subdir or resolve_one_to_all_model_dir_name(args.model)
    target_dir = models_dir / target_subdir

    if args.check_only:
        exists = _dir_has_any_content(target_dir)
        print(f"[check] target_dir={target_dir} exists={exists}")
        return 0 if exists else 1

    if _dir_has_any_content(target_dir) and not args.force:
        print(f"[skip] already exists: {target_dir}")
        return 0

    repo_id = args.repo_id or resolve_one_to_all_repo_id(args.model)
    print(f"[download] repo_id={repo_id}")
    print(f"[download] target_dir={target_dir}")

    try:
        downloaded = _call_snapshot_download(repo_id, target_dir=target_dir, revision=args.revision)
    except ModuleNotFoundError as exc:
        if exc.name != "modelscope":
            raise
        print(
            "Missing dependency: modelscope\n"
            "Install it in the API env:\n"
            "  uv sync --project apps/api --extra model_download",
            file=sys.stderr,
        )
        return 2

    print(f"[done] downloaded_to={downloaded}")
    print(f"[done] usable_path={target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

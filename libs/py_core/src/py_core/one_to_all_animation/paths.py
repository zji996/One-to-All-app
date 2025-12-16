from __future__ import annotations

from pathlib import Path


def find_repo_root() -> Path:
    candidates = [Path.cwd(), Path(__file__).resolve()]
    for start in candidates:
        for parent in [start, *start.parents]:
            if (parent / "apps").is_dir() and (parent / "libs").is_dir():
                return parent
    return Path.cwd()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return find_repo_root() / path


from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from py_core.one_to_all_animation.paths import resolve_repo_path
from py_core.config.settings import settings


@contextmanager
def video_generation_context() -> Iterator[tuple[Path, Path]]:
    third_party_root = resolve_repo_path(settings.one_to_all_animation_dir)
    video_generation_dir = third_party_root / "video-generation"
    if not video_generation_dir.is_dir():
        raise FileNotFoundError(
            f"One-to-All-Animation video-generation dir not found: {video_generation_dir}"
        )

    runtime_root = resolve_repo_path(settings.one_to_all_animation_runtime_dir)
    runtime_video_generation_dir = runtime_root / "video-generation"
    runtime_video_generation_dir.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    os.chdir(runtime_video_generation_dir)
    sys.path.insert(0, str(video_generation_dir))
    try:
        yield video_generation_dir, runtime_root
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path

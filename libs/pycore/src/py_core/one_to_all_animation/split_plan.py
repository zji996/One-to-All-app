from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SplitPlan:
    plan: list[tuple[int, int]]
    total_frames: int
    main_chunk: int
    overlap_frames: int


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or not v.strip():
        return int(default)
    return int(v)


def choose_main_chunk(*, device: str) -> int:
    if os.environ.get("ONE_TO_ALL_MAIN_CHUNK"):
        return _env_int("ONE_TO_ALL_MAIN_CHUNK", 81)

    main_chunk = 81
    try:
        if device.startswith("cuda"):
            import torch

            gpu_id = 0
            if ":" in device:
                gpu_id = int(device.split(":", 1)[1])
            total = int(torch.cuda.get_device_properties(gpu_id).total_memory)
            if total <= 36 * 1024**3:
                main_chunk = 33
    except Exception:
        pass
    return main_chunk


def build_split_plan(*, total_frames: int, device: str) -> SplitPlan:
    total_len = int(total_frames)
    main_chunk = int(choose_main_chunk(device=device))
    overlap_frames = _env_int("ONE_TO_ALL_OVERLAP_FRAMES", 5 if main_chunk >= 81 else 3)
    final_chunk_candidates = [25, 29, 33] if main_chunk <= 33 else [65, 69, 73, 77, 81]

    def _plan() -> list[tuple[int, int]]:
        if total_len <= 0:
            return []
        if total_len <= main_chunk:
            return [(0, total_len)]
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

    return SplitPlan(
        plan=_plan(),
        total_frames=total_len,
        main_chunk=main_chunk,
        overlap_frames=overlap_frames,
    )


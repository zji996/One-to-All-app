from __future__ import annotations

import os
from pathlib import Path


def df11_available() -> bool:
    try:
        import dfloat11  # noqa: F401
    except Exception:
        return False
    return True


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    v = v.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def load_df11_into_model(model, df11_dir: Path) -> None:
    try:
        from dfloat11 import DFloat11Model
    except Exception as exc:
        raise ImportError(
            "DF11 weights found, but `dfloat11` is not installed.\n"
            "Install via worker env:\n"
            "  uv sync --project apps/worker --extra df11\n"
            "  # or: uv run --project apps/worker pip install 'dfloat11[cuda12]'\n"
        ) from exc

    prefer_block_offload = _env_flag("ONE_TO_ALL_DF11_PREFER_BLOCK_OFFLOAD", True)
    cpu_offload = _env_flag("ONE_TO_ALL_DF11_CPU_OFFLOAD", prefer_block_offload)
    pin_memory = _env_flag("ONE_TO_ALL_DF11_PIN_MEMORY", prefer_block_offload)
    cpu_offload_blocks = os.environ.get("ONE_TO_ALL_DF11_CPU_OFFLOAD_BLOCKS", "").strip()
    cpu_offload_blocks_val = int(cpu_offload_blocks) if cpu_offload_blocks else None

    DFloat11Model.from_pretrained(
        dfloat11_model_name_or_path=str(df11_dir),
        device="cpu",
        bfloat16_model=model,
        cpu_offload=cpu_offload,
        cpu_offload_blocks=cpu_offload_blocks_val,
        pin_memory=pin_memory,
    )


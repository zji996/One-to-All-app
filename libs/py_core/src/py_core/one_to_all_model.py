from __future__ import annotations

from pathlib import Path

from py_core.settings import settings


_MODEL_NAME_TO_DIR: dict[str, str] = {
    "14b": "One-to-All-14b",
    "14b_fp8": "One-to-All-14b-FP8",
    "1.3b_1": "One-to-All-1.3b_1",
    "1.3b_2": "One-to-All-1.3b_2",
}

_MODEL_NAME_TO_REPO_ID: dict[str, str] = {
    "14b": "MochunniaN1/One-to-All-14b",
    "14b_fp8": "MochunniaN1/One-to-All-14b",
    "1.3b_1": "MochunniaN1/One-to-All-1.3b_1",
    "1.3b_2": "MochunniaN1/One-to-All-1.3b_2",
}


def resolve_one_to_all_repo_id(model_name: str | None) -> str:
    if not model_name:
        return settings.one_to_all_model_repo_id
    return _MODEL_NAME_TO_REPO_ID.get(model_name, model_name)


def resolve_one_to_all_model_dir_name(model_name: str | None) -> str:
    if not model_name:
        return settings.one_to_all_model_dir
    return _MODEL_NAME_TO_DIR.get(model_name, model_name)


def resolve_one_to_all_checkpoint_dir(model_name: str | None) -> Path:
    return Path(settings.models_dir) / resolve_one_to_all_model_dir_name(model_name)

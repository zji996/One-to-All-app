from __future__ import annotations

import argparse
import os
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_from_repo_root(repo_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def _any_file_under(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return True
    for _, _, files in os.walk(path):
        if files:
            return True
    return False


def _onnx_checkpoint_exists(path: Path) -> bool:
    if path.is_file():
        return True
    if path.is_dir():
        return (path / "end2end.onnx").is_file()
    return False


def _default_model_dir() -> str:
    env = os.environ.get("ONE_TO_ALL_14B_DIR")
    if env:
        return env
    try:
        from py_core.settings import settings

        return str(Path(settings.models_dir) / settings.one_to_all_model_dir)
    except Exception:
        return "models/One-to-All-14b-FP8"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that `models/<ONE_TO_ALL_MODEL_DIR>/` contains everything needed for One-to-All-Animation inference.\n\n"
            "Run via worker env:\n"
            "  uv run --project apps/worker scripts/verify_one_to_all_14b_model_repo.py\n"
        )
    )
    parser.add_argument(
        "--model-dir",
        default=_default_model_dir(),
        help="Path to One-to-All model repo dir (default: MODELS_DIR/ONE_TO_ALL_MODEL_DIR).",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    model_dir = _resolve_from_repo_root(repo_root, str(args.model_dir))

    errors: list[str] = []
    checks: list[tuple[str, bool]] = []

    # 1) One-to-All finetuned checkpoint weights
    transformer_df11_dir = model_dir / "transformer_df11"
    has_transformer_df11 = (
        transformer_df11_dir.is_dir()
        and (transformer_df11_dir / "config.json").is_file()
        and any(transformer_df11_dir.glob("*.safetensors"))
    )
    checks.append(("one_to_all_dir_exists", model_dir.is_dir()))
    checks.append(
        (
            "one_to_all_has_index_or_df11",
            (model_dir / "model.safetensors.index.json").is_file() or has_transformer_df11,
        )
    )
    checks.append(("one_to_all_has_config", (model_dir / "configuration.json").is_file()))
    checks.append(("one_to_all_has_bf16_or_df11_weights", any(model_dir.glob("*.safetensors")) or has_transformer_df11))

    # 2) Pretrained assets needed by upstream preprocessing + Wan base components
    pretrained_root = model_dir / "pretrained_models"
    wan_dir = pretrained_root / "Wan2.1-T2V-14B-Diffusers"
    checks.append(("pretrained_root_exists", pretrained_root.is_dir()))
    checks.append(("dwpose_exists", (pretrained_root / "DWPose").is_dir()))
    checks.append(("dwpose_non_empty", _any_file_under(pretrained_root / "DWPose")))

    pose2d_ckpt = (
        pretrained_root / "process_checkpoint" / "pose2d" / "vitpose_h_wholebody.onnx"
    )
    det_ckpt = pretrained_root / "process_checkpoint" / "det" / "yolov10m.onnx"
    checks.append(("pose2d_ckpt_exists", _onnx_checkpoint_exists(pose2d_ckpt)))
    checks.append(("det_ckpt_exists", _onnx_checkpoint_exists(det_ckpt)))

    text_encoder_df11_dir = wan_dir / "text_encoder_df11"
    has_text_encoder_df11 = (
        text_encoder_df11_dir.is_dir()
        and (text_encoder_df11_dir / "config.json").is_file()
        and any(text_encoder_df11_dir.glob("*.safetensors"))
    )

    for name in ["vae", "text_encoder", "tokenizer", "scheduler"]:
        p = wan_dir / name
        checks.append((f"wan_{name}_dir_exists", p.is_dir()))
        checks.append((f"wan_{name}_non_empty", _any_file_under(p)))

    text_encoder_dir = wan_dir / "text_encoder"
    checks.append(
        (
            "wan_text_encoder_has_bf16_or_df11_weights",
            has_text_encoder_df11 or any(text_encoder_dir.glob("*.safetensors")),
        )
    )

    # If transformer exists, warn (it is huge + redundant for One-to-All-14b).
    transformer_dir = wan_dir / "transformer"
    if transformer_dir.exists():
        checks.append(("wan_transformer_present_warn", True))

    ok = True
    for name, passed in checks:
        if name.endswith("_warn"):
            print(f"[warn] {name}=true")
            continue
        print(f"[check] {name}={str(passed).lower()}")
        if not passed:
            ok = False
            errors.append(name)

    if not ok:
        print("[result] FAIL")
        print(f"[result] missing_or_invalid={errors}")
        return 1

    print("[result] OK")
    print(f"[result] model_dir={model_dir}")
    print(
        "[note] You can now delete the legacy pretrained stash:\n"
        "  rm -rf models/One-to-All-Animation"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_dir_from_repo_root(repo_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def _download_with_modelscope(
    repo_id: str,
    *,
    target_dir: Path,
    allow_patterns: str | list[str] | None = None,
    revision: str | None = None,
) -> None:
    from modelscope.hub.snapshot_download import snapshot_download

    snapshot_download(
        model_id=repo_id,
        revision=revision,
        allow_patterns=allow_patterns,
        local_dir=str(target_dir),
        repo_type="model",
    )


def _download_with_huggingface(
    repo_id: str,
    *,
    target_dir: Path,
    allow_patterns: str | list[str] | None = None,
    revision: str | None = None,
) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=allow_patterns,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        repo_type="model",
        resume_download=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download One-to-All-Animation pretrained assets (pose preprocess checkpoints and Wan base model).\n\n"
            "Recommended (API env, ModelScope):\n"
            "  uv sync --project apps/api --extra model_download\n"
            "  uv run --project apps/api scripts/download_one_to_all_animation_pretrained.py --source modelscope --with-wan-14b\n"
        )
    )
    parser.add_argument(
        "--source",
        choices=["modelscope", "hf"],
        default=os.environ.get("ONE_TO_ALL_PRETRAINED_SOURCE") or "modelscope",
        help="Download source: modelscope | hf (default: modelscope).",
    )
    parser.add_argument(
        "--target-dir",
        default=os.environ.get("ONE_TO_ALL_PRETRAINED_DIR")
        or os.environ.get("ONE_TO_ALL_ANIMATION_PRETRAINED_DIR")
        or "models/One-to-All-14b/pretrained_models",
        help="Download destination directory (default: models/One-to-All-14b/pretrained_models).",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com",
        help="HuggingFace endpoint/mirror (default: https://hf-mirror.com).",
    )
    parser.add_argument(
        "--revision",
        default=os.environ.get("ONE_TO_ALL_PRETRAINED_REVISION") or None,
        help="Optional revision/tag.",
    )
    parser.add_argument(
        "--with-wan-14b",
        action="store_true",
        help=(
            "Also download Wan-AI/Wan2.1-T2V-14B-Diffusers components needed for One-to-All "
            "(defaults to vae/text_encoder/tokenizer/scheduler; pass --with-wan-transformer to include transformer weights)."
        ),
    )
    parser.add_argument(
        "--with-wan-1-3b",
        action="store_true",
        help=(
            "Also download Wan-AI/Wan2.1-T2V-1.3B-Diffusers components "
            "(defaults to vae/text_encoder/tokenizer/scheduler; pass --with-wan-transformer to include transformer weights)."
        ),
    )
    parser.add_argument(
        "--with-wan-transformer",
        action="store_true",
        help=(
            "Also download the base Wan Diffusers `transformer/` weights. "
            "One-to-All-14b already provides finetuned transformer weights, so this is usually not needed."
        ),
    )
    args = parser.parse_args()

    download_fn = None
    if args.source == "modelscope":
        try:
            import modelscope  # noqa: F401
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Missing dependency: modelscope\n"
                "Install it in the API env:\n"
                "  uv sync --project apps/api --extra model_download\n"
                "Or switch to HuggingFace:\n"
                "  uv sync --project apps/worker --extra one_to_all_animation\n"
                "  uv run --project apps/worker scripts/download_one_to_all_animation_pretrained.py --source hf --with-wan-14b\n"
            ) from exc
        download_fn = _download_with_modelscope
    else:
        os.environ.setdefault("HF_ENDPOINT", str(args.hf_endpoint))
        print(f"[env] HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
        try:
            import huggingface_hub  # noqa: F401
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Missing dependency: huggingface-hub\n"
                "Install it in the worker env:\n"
                "  uv sync --project apps/worker --extra one_to_all_animation\n"
                "Or switch to ModelScope:\n"
                "  uv sync --project apps/api --extra model_download\n"
                "  uv run --project apps/api scripts/download_one_to_all_animation_pretrained.py --source modelscope --with-wan-14b\n"
            ) from exc
        download_fn = _download_with_huggingface

    repo_root = _repo_root()
    target_root = _resolve_dir_from_repo_root(repo_root, str(args.target_dir))
    target_root.mkdir(parents=True, exist_ok=True)
    print(f"[target] {target_root}")

    # DWPose (StableAnimator)
    download_fn(
        "FrancisRing/StableAnimator",
        target_dir=target_root,
        allow_patterns="DWPose/*",
        revision=args.revision,
    )

    # Wan pose preprocess checkpoints
    download_fn(
        "Wan-AI/Wan2.2-Animate-14B",
        target_dir=target_root,
        allow_patterns="process_checkpoint/det/*",
        revision=args.revision,
    )
    download_fn(
        "Wan-AI/Wan2.2-Animate-14B",
        target_dir=target_root,
        allow_patterns="process_checkpoint/pose2d/*",
        revision=args.revision,
    )

    wan_keep_patterns = [
        "README*",
        "*.md",
        "*.json",
        ".mv",
        ".msc",
        "scheduler/*",
        "tokenizer/*",
        "text_encoder/*",
        "vae/*",
        "null_text_embedding.pt",
    ]

    if args.with_wan_1_3b:
        download_fn(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            target_dir=(target_root / "Wan2.1-T2V-1.3B-Diffusers"),
            allow_patterns=None if args.with_wan_transformer else wan_keep_patterns,
            revision=args.revision,
        )

    if args.with_wan_14b:
        download_fn(
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            target_dir=(target_root / "Wan2.1-T2V-14B-Diffusers"),
            allow_patterns=None if args.with_wan_transformer else wan_keep_patterns,
            revision=args.revision,
        )

    print("[done] ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

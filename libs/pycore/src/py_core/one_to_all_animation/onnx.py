from __future__ import annotations

import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    v = v.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def patch_onnxruntime_for_pose(runtime_root: Path) -> None:
    if not _env_bool("ONE_TO_ALL_POSE_USE_ONNX", True):
        return
    try:
        import onnxruntime as ort
    except Exception:
        return

    if getattr(ort, "_one_to_all_patched", False):
        return

    cache_root = runtime_root / "onnx_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    original = ort.InferenceSession

    def _session_with_fallback(*args, **kwargs):  # type: ignore[no-untyped-def]
        sess_options = kwargs.get("sess_options")
        if sess_options is None:
            sess_options = ort.SessionOptions()
            kwargs["sess_options"] = sess_options

        if _env_bool("ONE_TO_ALL_ONNX_SAVE_OPTIMIZED", True):
            model_path = kwargs.get("path_or_bytes")
            if model_path is None and args:
                model_path = args[0]
            if isinstance(model_path, (str, os.PathLike)):
                import hashlib

                p = Path(model_path)
                token = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:10]
                name = p.name or "model.onnx"
                sess_options.optimized_model_filepath = str(
                    cache_root / f"{name}.{token}.optimized.onnx"
                )

        try:
            return original(*args, **kwargs)
        except Exception:
            providers = kwargs.get("providers")
            if providers and any("CUDAExecutionProvider" in str(p) for p in providers):
                kwargs["providers"] = ["CPUExecutionProvider"]
                return original(*args, **kwargs)
            raise

    ort.InferenceSession = _session_with_fallback  # type: ignore[assignment]
    ort._one_to_all_patched = True


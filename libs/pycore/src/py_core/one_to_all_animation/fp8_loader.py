from __future__ import annotations

from pathlib import Path


def has_fp8_scaled_weights(path: Path) -> bool:
    from safetensors import safe_open

    st_paths = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
    for p in st_paths:
        if not p.is_file():
            continue
        with safe_open(str(p), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith(".scale_weight"):
                    return True
    return False


def load_fp8_scaled_into_module(
    module,
    checkpoint_dir: Path,
    *,
    target_dtype,
    strict: bool,
) -> tuple[list[str], list[str]]:
    """
    Load fp8_scaled-style weights directly into an existing module to reduce peak RAM.

    Returns (missing_keys, unexpected_keys) similar to `load_state_dict`.
    """

    from safetensors import safe_open
    import torch

    shard_paths: list[Path]
    if checkpoint_dir.is_file():
        shard_paths = [checkpoint_dir]
    else:
        shard_paths = sorted(checkpoint_dir.glob("*.safetensors"))
        if not shard_paths:
            raise FileNotFoundError(f"No .safetensors shards found in: {checkpoint_dir}")

    fp8_dtypes = {torch.float8_e5m2, torch.float8_e4m3fn}

    scales: dict[str, torch.Tensor] = {}
    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith(".scale_weight"):
                    scales[key] = f.get_tensor(key)

    def _apply_fp8_scale(fp8_weight: torch.Tensor, scale_weight: torch.Tensor) -> torch.Tensor:
        if fp8_weight.ndim < 2:
            raise ValueError(
                f"Expected fp8_weight.ndim>=2, got {fp8_weight.ndim} ({fp8_weight.shape})"
            )
        if scale_weight.ndim != 1:
            raise ValueError(
                f"Expected scale_weight.ndim==1, got {scale_weight.ndim} ({scale_weight.shape})"
            )
        if int(scale_weight.shape[0]) != int(fp8_weight.shape[0]):
            raise ValueError(
                "scale_weight shape mismatch: "
                f"scale={tuple(scale_weight.shape)} weight={tuple(fp8_weight.shape)}"
            )
        view_shape = (int(scale_weight.shape[0]),) + (1,) * (fp8_weight.ndim - 1)
        return fp8_weight.to(torch.float16) * scale_weight.to(torch.float16).view(view_shape)

    param_tensors = module.state_dict()
    missing_keys = set(param_tensors.keys())
    unexpected_keys: list[str] = []

    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith(".scale_weight"):
                    continue
                if key not in param_tensors:
                    unexpected_keys.append(key)
                    continue

                tensor = f.get_tensor(key)
                if tensor.dtype in fp8_dtypes and key.endswith(".weight"):
                    scale_key = key[: -len(".weight")] + ".scale_weight"
                    scale = scales.get(scale_key)
                    if scale is None:
                        raise KeyError(
                            f"Missing scale tensor for FP8 weight: {key} (expected {scale_key})"
                        )
                    tensor = _apply_fp8_scale(tensor, scale)

                tensor = tensor.to(target_dtype)
                dest = param_tensors[key]
                if dest.shape != tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for {key}: checkpoint={tuple(tensor.shape)} module={tuple(dest.shape)}"
                    )
                dest.copy_(tensor)
                missing_keys.discard(key)

    missing = sorted(missing_keys)
    if strict and (missing or unexpected_keys):
        raise KeyError(
            "State dict mismatch: "
            f"missing={len(missing)} unexpected={len(unexpected_keys)} "
            f"(missing_first={missing[:5]} unexpected_first={unexpected_keys[:5]})"
        )
    return missing, unexpected_keys


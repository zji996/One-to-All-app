from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable


_WAN_DIFFUSERS_DIRNAME = "Wan2.1-T2V-14B-Diffusers"
_WAN_TEXT_ENCODER_SUBDIR = "text_encoder"

_PATCH_EMBEDDING_WEIGHT_RE = re.compile(r"(?:^|\\.)patch_embedding\\.weight$")
_EMBEDDING_WEIGHT_RE = re.compile(r"(?:^|\\.)embeddings?\\.weight$")
_TOKEN_EMBEDDING_WEIGHT_RE = re.compile(r"(?:^|\\.)token_embeddings?\\.weight$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_from_repo_root(repo_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def _to_device(tensor, *, device: str):
    if tensor.device.type == device:
        return tensor
    return tensor.to(device)


def _tensor_nbytes(tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _scale_key_from_weight_key(weight_key: str) -> str:
    if not weight_key.endswith(".weight"):
        raise ValueError(f"Expected a .weight key, got: {weight_key}")
    return weight_key[: -len(".weight")] + ".scale_weight"


def _is_embedding_like_weight_key(key: str) -> bool:
    if "embedder" in key:
        return False
    if _PATCH_EMBEDDING_WEIGHT_RE.search(key):
        return True
    if _EMBEDDING_WEIGHT_RE.search(key):
        return True
    if _TOKEN_EMBEDDING_WEIGHT_RE.search(key):
        return True
    return False


def _fp8_dtype():
    import torch

    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        raise RuntimeError("Missing torch.float8_e4m3fn; please upgrade PyTorch.")
    return dtype


def _try_fp8_cpu_transfer(fp8_dtype) -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    x = torch.tensor([1.0], device="cuda", dtype=torch.float16).to(fp8_dtype)
    try:
        _ = x.cpu()
        return True
    except Exception:
        return False


def _quantize_weight_fp8_per_out_dim(
    weight,
    *,
    fp8_dtype,
    save_device: str,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch

    if weight.ndim < 2:
        raise ValueError(f"Expected weight.ndim>=2, got shape={tuple(weight.shape)}")
    if not weight.is_floating_point():
        raise ValueError(f"Expected floating weight, got dtype={weight.dtype}")

    weight_f32 = weight.to(torch.float32)
    fp8_max = torch.finfo(fp8_dtype).max

    reduce_dims = tuple(range(1, weight_f32.ndim))
    max_abs_per_out = weight_f32.abs().amax(dim=reduce_dims)
    scale = (max_abs_per_out / fp8_max).clamp_min(1e-8)
    scale = torch.where(max_abs_per_out == 0, torch.ones_like(scale), scale)

    view_shape = (int(scale.shape[0]),) + (1,) * (weight_f32.ndim - 1)
    q = (weight_f32 / scale.view(view_shape)).clamp(min=-fp8_max, max=fp8_max).to(fp8_dtype)
    scale_weight = scale.to(torch.float16)

    q = _to_device(q, device=save_device)
    scale_weight = _to_device(scale_weight, device=save_device)
    return q, scale_weight


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_index(path: Path, *, weight_map: dict[str, str], total_size: int, metadata: dict) -> None:
    payload = {
        "metadata": {"total_size": int(total_size), **metadata},
        "weight_map": weight_map,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _copy_top_level_non_weight_files(*, src_dir: Path, dst_dir: Path, index_name: str) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.iterdir():
        if src.is_dir():
            continue
        if src.name.endswith(".safetensors"):
            continue
        if src.name == index_name:
            continue
        shutil.copy2(src, dst_dir / src.name)


def _quantize_indexed_checkpoint_dir(
    input_dir: Path,
    output_dir: Path,
    *,
    index_name: str,
    should_quantize: Callable[[str, "torch.Tensor"], bool],
    fp8_dtype,
    device: str,
    save_device: str,
    max_shard_bytes: int,
    force: bool,
    dry_run: bool,
) -> dict[str, int]:
    from safetensors.torch import safe_open, save_file

    index_path = input_dir / index_name
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")

    index = _load_json(index_path)
    weight_map_in: dict[str, str] = index.get("weight_map") or {}
    if not weight_map_in:
        raise ValueError(f"Invalid index (missing weight_map): {index_path}")

    keys_by_file: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map_in.items():
        keys_by_file[filename].append(key)

    if dry_run:
        quantizable = 0
        skipped = 0
        for filename, keys in sorted(keys_by_file.items()):
            shard_path = input_dir / filename
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in keys:
                    t = f.get_tensor(key)
                    if should_quantize(key, t):
                        quantizable += 1
                    else:
                        skipped += 1
        return {"quantizable": quantizable, "skipped": skipped, "shards_in": len(keys_by_file)}

    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output dir already exists: {output_dir} (use --force to overwrite)")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_top_level_non_weight_files(src_dir=input_dir, dst_dir=output_dir, index_name=index_name)

    tmp_shard_paths: list[Path] = []
    tmp_shard_names: list[str] = []
    weight_map_out: dict[str, str] = {}
    tensors_out: dict[str, "torch.Tensor"] = {}
    keys_out: list[str] = []
    tensors_out_nbytes = 0
    tmp_shard_idx = 0
    total_size = 0

    def flush() -> None:
        nonlocal tensors_out, tensors_out_nbytes, tmp_shard_idx, total_size, keys_out
        if not tensors_out:
            return
        tmp_shard_idx += 1
        tmp_name = f"model-shard-{tmp_shard_idx:05d}.safetensors"
        tmp_path = output_dir / tmp_name
        save_file(tensors_out, str(tmp_path), metadata={"format": "pt"})
        shard_size = tmp_path.stat().st_size
        total_size += int(shard_size)
        tmp_shard_paths.append(tmp_path)
        tmp_shard_names.append(tmp_name)
        for key in keys_out:
            weight_map_out[key] = tmp_name
        tensors_out = {}
        keys_out = []
        tensors_out_nbytes = 0

    def add_tensor(key: str, tensor) -> None:
        nonlocal tensors_out_nbytes
        tensor_nbytes = _tensor_nbytes(tensor)
        if tensors_out and tensors_out_nbytes + tensor_nbytes > max_shard_bytes:
            flush()
        tensors_out[key] = tensor
        keys_out.append(key)
        tensors_out_nbytes += tensor_nbytes

    quantized_count = 0
    passthrough_count = 0

    for filename, keys in sorted(keys_by_file.items()):
        shard_path = input_dir / filename
        print(f"[load] {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in sorted(keys):
                t_cpu = f.get_tensor(key)
                if should_quantize(key, t_cpu):
                    t = t_cpu.to(device=device, non_blocking=True)
                    q, scale_weight = _quantize_weight_fp8_per_out_dim(
                        t, fp8_dtype=fp8_dtype, save_device=save_device
                    )
                    add_tensor(key, q)
                    add_tensor(_scale_key_from_weight_key(key), scale_weight)
                    quantized_count += 1
                    continue

                passthrough = _to_device(t_cpu, device=save_device)
                add_tensor(key, passthrough)
                passthrough_count += 1

    flush()

    total_shards = len(tmp_shard_paths)
    final_weight_map: dict[str, str] = {}
    rename_map: dict[str, str] = {}
    for idx, tmp_name in enumerate(tmp_shard_names, start=1):
        final_name = f"model-{idx:05d}-of-{total_shards:05d}.safetensors"
        rename_map[tmp_name] = final_name

    for tmp_path in tmp_shard_paths:
        tmp_name = tmp_path.name
        tmp_path.rename(output_dir / rename_map[tmp_name])

    for key, tmp_name in weight_map_out.items():
        final_weight_map[key] = rename_map[tmp_name]

    _write_index(
        output_dir / index_name,
        weight_map=final_weight_map,
        total_size=total_size,
        metadata={
            "fp8_scaled_dtype": str(fp8_dtype).replace("torch.", ""),
            "fp8_scaled_algorithm": "per_out_dim_max_abs",
            "fp8_scaled_scale_key_suffix": ".scale_weight",
        },
    )
    return {
        "quantized_weights": quantized_count,
        "passthrough_tensors": passthrough_count,
        "shards_out": total_shards,
        "total_size_bytes": total_size,
    }


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
        return
    except OSError:
        shutil.copy2(src, dst)


def _sync_pretrained_models(
    src_root: Path,
    dst_root: Path,
    *,
    skip_rel_dirs: set[Path],
    dry_run: bool,
) -> int:
    if not src_root.is_dir():
        return 0
    if not dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    for dirpath, dirnames, filenames in os.walk(src_root):
        dirpath = Path(dirpath)
        rel_dir = dirpath.relative_to(src_root)

        if any(rel_dir == d or rel_dir.is_relative_to(d) for d in skip_rel_dirs):
            dirnames[:] = []
            continue

        dirnames[:] = [d for d in dirnames if d not in {"._____temp", ".git"}]
        if rel_dir.parts and any(p in {"._____temp", ".git"} for p in rel_dir.parts):
            continue

        for name in filenames:
            if name in {".DS_Store"}:
                continue
            src = dirpath / name
            dst = dst_root / rel_dir / name
            if dst.exists():
                continue
            copied += 1
            if dry_run:
                continue
            _link_or_copy(src, dst)
    return copied


def _verify_fp8_scaled_checkpoint_dir(ckpt_dir: Path, *, index_name: str) -> dict[str, int]:
    from safetensors.torch import safe_open
    import torch

    index_path = ckpt_dir / index_name
    index = _load_json(index_path)
    weight_map: dict[str, str] = index.get("weight_map") or {}
    if not weight_map:
        raise ValueError(f"Invalid index (missing weight_map): {index_path}")

    keys_by_file: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map.items():
        keys_by_file[filename].append(key)

    fp8_dtypes = {getattr(torch, "float8_e5m2", None), getattr(torch, "float8_e4m3fn", None)}
    fp8_dtypes = {d for d in fp8_dtypes if d is not None}

    scale_meta: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
    for filename in sorted(keys_by_file):
        shard_path = ckpt_dir / filename
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith(".scale_weight"):
                    t = f.get_tensor(key)
                    scale_meta[key] = (tuple(t.shape), t.dtype)

    fp8_weights = 0
    scale_tensors = 0
    other_tensors = 0
    for filename in sorted(keys_by_file):
        shard_path = ckpt_dir / filename
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if key.endswith(".scale_weight"):
                    scale_tensors += 1
                    continue
                if t.dtype in fp8_dtypes and key.endswith(".weight"):
                    fp8_weights += 1
                    scale_key = _scale_key_from_weight_key(key)
                    meta = scale_meta.get(scale_key)
                    if meta is None:
                        raise KeyError(f"Missing scale tensor for FP8 weight: {key} (expected {scale_key})")
                    shape, dt = meta
                    if dt != torch.float16:
                        raise TypeError(f"Scale tensor dtype must be float16: {scale_key} (got {dt})")
                    if shape != (int(t.shape[0]),):
                        raise ValueError(
                            f"Scale tensor shape mismatch for {key}: weight={tuple(t.shape)} scale={shape}"
                        )
                else:
                    other_tensors += 1
    return {
        "shards": len(keys_by_file),
        "fp8_weights": fp8_weights,
        "scale_tensors": scale_tensors,
        "other_tensors": other_tensors,
        "total_tensors": fp8_weights + scale_tensors + other_tensors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "One-to-All FP8 quantization (best default, hardcoded).\n\n"
            "What this script does:\n"
            "- Quantize One-to-All transformer 2D `*.weight` (skip embedding-like weights).\n"
            "- Populate `<out>/pretrained_models` from `<in>/pretrained_models` (hardlink/copy).\n"
            "- Quantize Wan Diffusers `text_encoder/` 2D weights (but keep `shared.weight` unquantized).\n\n"
            "Run via worker env:\n"
            "  uv sync --project apps/worker --extra fp8_quant\n"
            "  uv run --project apps/worker scripts/quantize_one_to_all_fp8.py --model 14b\n"
        )
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ONE_TO_ALL_MODEL_NAME") or "14b",
        help="Model alias or directory name (default: 14b).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing FP8 outputs; do not write anything.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be quantized/copied.")
    parser.add_argument("--force", action="store_true", help="Overwrite output dirs if they exist.")
    args = parser.parse_args()

    try:
        from safetensors.torch import safe_open  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "safetensors":
            raise
        print(
            "Missing dependency: safetensors\n"
            "Install in worker env:\n"
            "  uv sync --project apps/worker --extra fp8_quant",
            file=sys.stderr,
        )
        return 2

    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        print(
            "Missing dependency: torch\n"
            "Install in worker env:\n"
            "  uv sync --project apps/worker --extra fp8_quant",
            file=sys.stderr,
        )
        return 2

    import torch

    if not torch.cuda.is_available():
        print("CUDA is not available; FP8 quantization requires a CUDA build of PyTorch.", file=sys.stderr)
        return 1

    fp8_dtype = _fp8_dtype()
    save_device = "cpu" if _try_fp8_cpu_transfer(fp8_dtype) else "cuda"
    if save_device != "cpu":
        print("[warn] FP8 tensors cannot be moved to CPU in this PyTorch build; saving from CUDA tensors.")

    repo_root = _repo_root()
    models_dir = _resolve_from_repo_root(repo_root, os.environ.get("MODELS_DIR", "models"))

    from py_core.one_to_all_model import resolve_one_to_all_model_dir_name

    input_dir = models_dir / resolve_one_to_all_model_dir_name(args.model)
    output_dir = models_dir / f"{input_dir.name}-FP8"
    index_name = "model.safetensors.index.json"
    max_shard_bytes = int(2.0 * 1024**3)

    if args.verify_only:
        if not output_dir.is_dir():
            print(f"Missing output dir: {output_dir}", file=sys.stderr)
            return 1
        stats = _verify_fp8_scaled_checkpoint_dir(output_dir, index_name=index_name)
        print(f"[ok] fp8_checkpoint={output_dir}")
        print(f"[ok] {stats}")
        return 0

    def should_quantize_transformer(key: str, t) -> bool:
        if not key.endswith(".weight"):
            return False
        if t.ndim != 2:
            return False
        if not t.is_floating_point():
            return False
        if _is_embedding_like_weight_key(key):
            return False
        return True

    def should_quantize_text_encoder(key: str, t) -> bool:
        if not key.endswith(".weight"):
            return False
        if t.ndim != 2:
            return False
        if not t.is_floating_point():
            return False
        if re.search(r"(?:^|\\.)shared\\.weight$", key):
            return False
        return True

    print(f"[cfg] input_dir={input_dir}")
    print(f"[cfg] output_dir={output_dir}")
    print(f"[cfg] fp8_dtype={str(fp8_dtype).replace('torch.', '')}")

    if not input_dir.is_dir():
        print(f"Missing input model dir: {input_dir}", file=sys.stderr)
        return 1

    if output_dir.exists() and not args.force:
        print(f"[skip] output_dir already exists, keep as-is: {output_dir}")
        if args.dry_run:
            return 0
    else:
        transformer_stats = _quantize_indexed_checkpoint_dir(
            input_dir,
            output_dir,
            index_name=index_name,
            should_quantize=should_quantize_transformer,
            fp8_dtype=fp8_dtype,
            device="cuda",
            save_device=save_device,
            max_shard_bytes=max_shard_bytes,
            force=bool(args.force),
            dry_run=bool(args.dry_run),
        )
        print(f"[transformer] {transformer_stats}")
        if args.dry_run:
            return 0

    src_pretrained_root = input_dir / "pretrained_models"
    dst_pretrained_root = output_dir / "pretrained_models"
    skip_rel_dirs = {
        Path(_WAN_DIFFUSERS_DIRNAME) / _WAN_TEXT_ENCODER_SUBDIR,
    }
    copied = _sync_pretrained_models(
        src_pretrained_root, dst_pretrained_root, skip_rel_dirs=skip_rel_dirs, dry_run=False
    )
    print(f"[pretrained_models] copied_files={copied}")

    te_in_dir = src_pretrained_root / _WAN_DIFFUSERS_DIRNAME / _WAN_TEXT_ENCODER_SUBDIR
    te_out_dir = dst_pretrained_root / _WAN_DIFFUSERS_DIRNAME / _WAN_TEXT_ENCODER_SUBDIR
    if te_in_dir.is_dir():
        te_index = te_out_dir / index_name
        if te_index.exists() and not args.force:
            print(f"[skip] text_encoder output already exists, keep as-is: {te_out_dir}")
        else:
            te_stats = _quantize_indexed_checkpoint_dir(
                te_in_dir,
                te_out_dir,
                index_name=index_name,
                should_quantize=should_quantize_text_encoder,
                fp8_dtype=fp8_dtype,
                device="cuda",
                save_device=save_device,
                max_shard_bytes=max_shard_bytes,
                force=True,
                dry_run=False,
            )
            print(f"[text_encoder] {te_stats}")
    else:
        print(f"[warn] Wan text_encoder not found, skipped: {te_in_dir}")

    stats = _verify_fp8_scaled_checkpoint_dir(output_dir, index_name=index_name)
    print(f"[ok] fp8_checkpoint={output_dir}")
    print(f"[ok] {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

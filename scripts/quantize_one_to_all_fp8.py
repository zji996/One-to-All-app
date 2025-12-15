from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


_PATCH_EMBEDDING_WEIGHT_RE = re.compile(r"(?:^|\.)patch_embedding\.weight$")
_EMBEDDING_WEIGHT_RE = re.compile(r"(?:^|\.)embeddings?\.weight$")
_TOKEN_EMBEDDING_WEIGHT_RE = re.compile(r"(?:^|\.)token_embeddings?\.weight$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_dir_from_repo_root(repo_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def _parse_dtype(dtype_str: str):
    import torch

    normalized = dtype_str.strip().lower()
    if normalized in {"e5m2", "float8_e5m2", "fp8_e5m2"}:
        dtype = getattr(torch, "float8_e5m2", None)
        if dtype is None:
            raise RuntimeError("Missing torch.float8_e5m2; please upgrade PyTorch.")
        return dtype
    if normalized in {"e4m3fn", "float8_e4m3fn", "fp8_e4m3fn"}:
        dtype = getattr(torch, "float8_e4m3fn", None)
        if dtype is None:
            raise RuntimeError("Missing torch.float8_e4m3fn; please upgrade PyTorch.")
        return dtype
    raise ValueError(f"Unsupported --dtype: {dtype_str} (expected e5m2 / e4m3fn).")


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


@dataclass(frozen=True)
class QuantSpec:
    include_re: re.Pattern[str] | None
    exclude_re: re.Pattern[str] | None
    skip_embedding_like: bool

    def _is_embedding_like_weight_key(self, key: str) -> bool:
        if "embedder" in key:
            return False
        if _PATCH_EMBEDDING_WEIGHT_RE.search(key):
            return True
        if _EMBEDDING_WEIGHT_RE.search(key):
            return True
        if _TOKEN_EMBEDDING_WEIGHT_RE.search(key):
            return True
        return False

    def should_quantize_key(self, key: str) -> bool:
        if self.include_re and not self.include_re.search(key):
            return False
        if self.exclude_re and self.exclude_re.search(key):
            return False
        if self.skip_embedding_like and self._is_embedding_like_weight_key(key):
            return False
        return True


def _quantize_linear_weight_fp8_per_row(
    weight,
    *,
    fp8_dtype,
    save_device: str,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch

    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight, got shape={tuple(weight.shape)}")
    if not weight.is_floating_point():
        raise ValueError(f"Expected floating weight, got dtype={weight.dtype}")

    weight_f32 = weight.to(torch.float32)
    fp8_max = torch.finfo(fp8_dtype).max

    max_abs_per_row = weight_f32.abs().amax(dim=1)
    scale = (max_abs_per_row / fp8_max).clamp_min(1e-8)
    scale = torch.where(max_abs_per_row == 0, torch.ones_like(scale), scale)

    q = (weight_f32 / scale.unsqueeze(1)).clamp(min=-fp8_max, max=fp8_max).to(fp8_dtype)
    scale_weight = scale.to(torch.float16)

    q = _to_device(q, device=save_device)
    scale_weight = _to_device(scale_weight, device=save_device)
    return q, scale_weight


def _copy_non_weight_files(*, src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.iterdir():
        if src.is_dir():
            if src.name == "._____temp":
                continue
            continue
        if src.name.endswith(".safetensors"):
            continue
        if src.name == "model.safetensors.index.json":
            continue
        shutil.copy2(src, dst_dir / src.name)


def _load_index(path: Path) -> dict:
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Offline FP8 quantization for One-to-All safetensors checkpoints.\n\n"
            "Runs via worker env:\n"
            "  uv sync --project apps/worker --extra fp8_quant\n"
            "  uv run --project apps/worker scripts/quantize_one_to_all_fp8.py --model 14b\n\n"
            "By default this quantizes 2D tensors that match /\\.weight$/ and writes to MODELS_DIR/<name>-FP8."
        )
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ONE_TO_ALL_MODEL_NAME") or "14b",
        help="Model alias or directory name (default: 14b).",
    )
    parser.add_argument(
        "--in-dir",
        default=os.environ.get("ONE_TO_ALL_MODEL_DIR") or None,
        help="Override input subdir under MODELS_DIR (default derived from --model).",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("ONE_TO_ALL_FP8_MODEL_DIR") or None,
        help="Override output subdir under MODELS_DIR (default: <in-dir>-FP8).",
    )
    parser.add_argument(
        "--dtype",
        default=os.environ.get("ONE_TO_ALL_FP8_DTYPE") or "e5m2",
        help="FP8 dtype: e5m2 / e4m3fn (default: e5m2).",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("ONE_TO_ALL_FP8_DEVICE") or "cuda",
        help="Quantization device (default: cuda).",
    )
    parser.add_argument(
        "--include-regex",
        default=os.environ.get("ONE_TO_ALL_FP8_INCLUDE_REGEX") or r"\.weight$",
        help=r"Only quantize keys that match this regex (default: \.weight$).",
    )
    parser.add_argument(
        "--exclude-regex",
        default=os.environ.get("ONE_TO_ALL_FP8_EXCLUDE_REGEX") or "",
        help="Skip quantization for keys that match this regex (default: empty).",
    )
    parser.add_argument(
        "--skip-embedding-like",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Skip quantizing embedding-like weights (e.g. *.embeddings.weight, *.patch_embedding.weight), "
            "but do not exclude *embedder* linears (default: true)."
        ),
    )
    parser.add_argument(
        "--max-shard-size-gb",
        type=float,
        default=float(os.environ.get("ONE_TO_ALL_FP8_MAX_SHARD_GB") or "2.0"),
        help="Max output shard size in GiB (default: 2.0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan tensors and print counts without writing output.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it exists.",
    )
    args = parser.parse_args()

    try:
        from safetensors.torch import safe_open, save_file
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
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        print(
            "Missing dependency: torch\n"
            "Install in worker env:\n"
            "  uv sync --project apps/worker --extra fp8_quant\n"
            "Or install the latest build you want (CUDA/CPU) into the worker env.",
            file=sys.stderr,
        )
        return 2

    fp8_dtype = _parse_dtype(args.dtype)

    repo_root = _repo_root()
    models_dir = _resolve_dir_from_repo_root(repo_root, os.environ.get("MODELS_DIR", "models"))

    from py_core.one_to_all_model import resolve_one_to_all_model_dir_name

    in_subdir = args.in_dir or resolve_one_to_all_model_dir_name(args.model)
    input_dir = models_dir / in_subdir
    output_subdir = args.out_dir or f"{input_dir.name}-FP8"
    output_dir = models_dir / output_subdir

    index_path = input_dir / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"Missing index file: {index_path}", file=sys.stderr)
        return 1

    if output_dir.exists():
        if not args.force:
            print(f"Output dir already exists: {output_dir} (use --force to overwrite)", file=sys.stderr)
            return 1
        shutil.rmtree(output_dir)

    include_re = re.compile(args.include_regex) if args.include_regex else None
    exclude_re = re.compile(args.exclude_regex) if args.exclude_regex else None
    quant_spec = QuantSpec(
        include_re=include_re,
        exclude_re=exclude_re,
        skip_embedding_like=bool(args.skip_embedding_like),
    )

    if args.device != "cuda":
        print("Only cuda device is supported for FP8 quantization.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("CUDA is not available; FP8 quantization requires a CUDA build of PyTorch.", file=sys.stderr)
        return 1

    save_device = "cpu" if _try_fp8_cpu_transfer(fp8_dtype) else "cuda"
    if save_device != "cpu":
        print("[warn] FP8 tensors cannot be moved to CPU in this PyTorch build; saving from CUDA tensors.")

    index = _load_index(index_path)
    weight_map_in: dict[str, str] = index.get("weight_map") or {}
    if not weight_map_in:
        print(f"Invalid index (missing weight_map): {index_path}", file=sys.stderr)
        return 1

    keys_by_file: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map_in.items():
        keys_by_file[filename].append(key)

    max_shard_bytes = int(args.max_shard_size_gb * 1024**3)
    if max_shard_bytes <= 0:
        raise ValueError("--max-shard-size-gb must be > 0")

    if args.dry_run:
        quantizable = 0
        skipped_not_weight = 0
        skipped_not_2d = 0
        skipped_not_float = 0
        skipped_excluded = 0
        for filename, keys in sorted(keys_by_file.items()):
            shard_path = input_dir / filename
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in keys:
                    t = f.get_tensor(key)
                    if not key.endswith(".weight"):
                        skipped_not_weight += 1
                        continue
                    if t.ndim != 2:
                        skipped_not_2d += 1
                        continue
                    if not t.is_floating_point():
                        skipped_not_float += 1
                        continue
                    if not quant_spec.should_quantize_key(key):
                        skipped_excluded += 1
                        continue
                    quantizable += 1
        print(f"[dry-run] input_dir={input_dir}")
        print(f"[dry-run] quantizable_2d_weight_tensors={quantizable}")
        print(
            "[dry-run] skipped="
            f"not_weight={skipped_not_weight} "
            f"not_2d={skipped_not_2d} "
            f"not_float={skipped_not_float} "
            f"excluded={skipped_excluded}"
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_non_weight_files(src_dir=input_dir, dst_dir=output_dir)

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
                if (
                    key.endswith(".weight")
                    and t_cpu.ndim == 2
                    and t_cpu.is_floating_point()
                    and quant_spec.should_quantize_key(key)
                ):
                    t = t_cpu.to(device=args.device, non_blocking=True)
                    q, scale_weight = _quantize_linear_weight_fp8_per_row(
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
        output_dir / "model.safetensors.index.json",
        weight_map=final_weight_map,
        total_size=total_size,
        metadata={
            "one_to_all_fp8_dtype": str(fp8_dtype).replace("torch.", ""),
            "one_to_all_fp8_algorithm": "per_row_max_abs",
            "one_to_all_fp8_scale_key_suffix": ".scale_weight",
            "one_to_all_fp8_include_regex": args.include_regex,
            "one_to_all_fp8_exclude_regex": args.exclude_regex,
            "one_to_all_fp8_skip_embedding_like": bool(args.skip_embedding_like),
        },
    )

    print(f"[done] input_dir={input_dir}")
    print(f"[done] output_dir={output_dir}")
    print(f"[done] quantized_weights={quantized_count} passthrough_tensors={passthrough_count}")
    print(f"[done] shards={total_shards} total_size_bytes={total_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

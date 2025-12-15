from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_dir_from_repo_root(repo_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


@dataclass(frozen=True)
class Float8Layout:
    name: str
    exponent_bits: int
    mantissa_bits: int
    exponent_shift: int

    @property
    def exponent_mask(self) -> int:
        return (1 << self.exponent_bits) - 1

    @property
    def mantissa_mask(self) -> int:
        return (1 << self.mantissa_bits) - 1

    @property
    def sign_shift(self) -> int:
        return 7

    @property
    def sm_cardinality(self) -> int:
        return 1 << (1 + self.mantissa_bits)

    @property
    def exponent_cardinality(self) -> int:
        return 1 << self.exponent_bits


def _layout_for_dtype(dtype) -> Float8Layout | None:
    import torch

    if dtype == getattr(torch, "float8_e5m2", object()):
        return Float8Layout(name="float8_e5m2", exponent_bits=5, mantissa_bits=2, exponent_shift=2)
    if dtype == getattr(torch, "float8_e4m3fn", object()):
        return Float8Layout(name="float8_e4m3fn", exponent_bits=4, mantissa_bits=3, exponent_shift=3)
    return None


def _shannon_entropy_from_counts(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    inv_total = 1.0 / float(total)
    entropy = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = float(c) * inv_total
        entropy -= p * math.log2(p)
    return entropy


def _human_bytes(n: int) -> str:
    step = 1024.0
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    v = float(n)
    for u in units:
        if v < step:
            return f"{v:.2f}{u}"
        v /= step
    return f"{v:.2f}PiB"


def _load_index(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze byte-level compressibility of One-to-All FP8 checkpoints (DFloat-style stats).\n\n"
            "Example:\n"
            "  uv run --project apps/worker scripts/analyze_one_to_all_fp8_compressibility.py --model 14b\n"
            "  uv run --project apps/worker scripts/analyze_one_to_all_fp8_compressibility.py "
            "--ckpt-dir models/One-to-All-14b-FP8 --sample-stride 16\n"
        )
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ONE_TO_ALL_MODEL_NAME") or "14b",
        help="Model alias or directory name (default: 14b).",
    )
    parser.add_argument(
        "--ckpt-dir",
        default=os.environ.get("ONE_TO_ALL_FP8_CKPT_DIR") or None,
        help="FP8 checkpoint directory (default: MODELS_DIR/<model>-FP8).",
    )
    parser.add_argument(
        "--include-regex",
        default=os.environ.get("ONE_TO_ALL_FP8_ANALYZE_INCLUDE_REGEX") or r"\.weight$",
        help=r"Only analyze keys that match this regex (default: \.weight$).",
    )
    parser.add_argument(
        "--exclude-regex",
        default=os.environ.get("ONE_TO_ALL_FP8_ANALYZE_EXCLUDE_REGEX") or "",
        help="Skip keys that match this regex (default: empty).",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=int(os.environ.get("ONE_TO_ALL_FP8_ANALYZE_SAMPLE_STRIDE") or "1"),
        help="Analyze every Nth FP8 element (1 = full scan; default: 1).",
    )
    parser.add_argument(
        "--json-out",
        default=os.environ.get("ONE_TO_ALL_FP8_ANALYZE_JSON_OUT") or "",
        help="Optional path to write a JSON summary.",
    )
    args = parser.parse_args()

    if args.sample_stride <= 0:
        raise ValueError("--sample-stride must be >= 1")

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        if exc.name != "numpy":
            raise
        print(
            "Missing dependency: numpy\n"
            "Install in worker env:\n"
            "  uv sync --project apps/worker --extra fp8_quant",
            file=sys.stderr,
        )
        return 2

    try:
        from safetensors.torch import safe_open
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

    import torch

    repo_root = _repo_root()
    models_dir = _resolve_dir_from_repo_root(repo_root, os.environ.get("MODELS_DIR", "models"))
    if args.ckpt_dir:
        ckpt_dir = _resolve_dir_from_repo_root(repo_root, args.ckpt_dir)
    else:
        from py_core.one_to_all_model import resolve_one_to_all_model_dir_name

        model_dir_name = resolve_one_to_all_model_dir_name(args.model)
        ckpt_dir = models_dir / f"{model_dir_name}-FP8"

    index_path = ckpt_dir / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"Missing index file: {index_path}", file=sys.stderr)
        return 1

    include_re = re.compile(args.include_regex) if args.include_regex else None
    exclude_re = re.compile(args.exclude_regex) if args.exclude_regex else None

    index = _load_index(index_path)
    weight_map: dict[str, str] = index.get("weight_map") or {}
    if not weight_map:
        print(f"Invalid index (missing weight_map): {index_path}", file=sys.stderr)
        return 1

    keys_by_file: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map.items():
        if include_re and not include_re.search(key):
            continue
        if exclude_re and exclude_re.search(key):
            continue
        keys_by_file[filename].append(key)

    start = time.time()
    byte_counts = np.zeros(256, dtype=np.int64)
    fp8_layout: Float8Layout | None = None

    total_fp8_elems_scanned = 0
    total_fp8_elems_seen = 0
    total_scale_elems = 0
    fp8_tensors = 0
    non_fp8_tensors = 0

    for filename, keys in sorted(keys_by_file.items()):
        shard_path = ckpt_dir / filename
        if not shard_path.exists():
            print(f"Missing shard: {shard_path}", file=sys.stderr)
            return 1
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in sorted(keys):
                t = f.get_tensor(key)

                if key.endswith(".scale_weight"):
                    total_scale_elems += int(t.numel())
                    continue

                layout = _layout_for_dtype(t.dtype)
                if layout is None:
                    non_fp8_tensors += 1
                    continue
                if fp8_layout is None:
                    fp8_layout = layout
                elif fp8_layout != layout:
                    print(
                        f"Mixed FP8 dtypes detected: {fp8_layout.name} and {layout.name}; "
                        "run analysis per checkpoint dtype.",
                        file=sys.stderr,
                    )
                    return 1

                fp8_tensors += 1
                total_fp8_elems_seen += int(t.numel())

                u8 = t.view(torch.uint8)
                if args.sample_stride == 1:
                    sample = u8.reshape(-1).numpy()
                else:
                    sample = u8.reshape(-1)[:: args.sample_stride].numpy()

                byte_counts += np.bincount(sample, minlength=256).astype(np.int64, copy=False)
                total_fp8_elems_scanned += int(sample.size)

    if fp8_layout is None:
        print(f"No FP8 tensors found under {ckpt_dir}.", file=sys.stderr)
        return 1

    exponent_counts = [0] * fp8_layout.exponent_cardinality
    sm_counts = [0] * fp8_layout.sm_cardinality
    full_byte_counts = [int(x) for x in byte_counts.tolist()]
    for byte_value, c in enumerate(full_byte_counts):
        if c <= 0:
            continue
        exp = (byte_value >> fp8_layout.exponent_shift) & fp8_layout.exponent_mask
        mant = byte_value & fp8_layout.mantissa_mask
        sign = (byte_value >> fp8_layout.sign_shift) & 0x1
        sm = (sign << fp8_layout.mantissa_bits) | mant
        exponent_counts[exp] += c
        sm_counts[sm] += c

    entropy_byte = _shannon_entropy_from_counts(full_byte_counts)
    entropy_exp = _shannon_entropy_from_counts(exponent_counts)
    entropy_sm = _shannon_entropy_from_counts(sm_counts)

    estimated_bits_per_weight = (1 + fp8_layout.mantissa_bits) + entropy_exp
    elapsed_s = time.time() - start

    index_total_size = None
    md = index.get("metadata") or {}
    if "total_size" in md:
        try:
            index_total_size = int(md["total_size"])
        except Exception:
            index_total_size = None

    fp8_weight_bytes = total_fp8_elems_seen * 1
    scale_weight_bytes = total_scale_elems * 2

    summary = {
        "ckpt_dir": str(ckpt_dir),
        "dtype": fp8_layout.name,
        "sample_stride": args.sample_stride,
        "fp8_tensors": fp8_tensors,
        "non_fp8_tensors": non_fp8_tensors,
        "fp8_elems_seen": total_fp8_elems_seen,
        "fp8_elems_scanned": total_fp8_elems_scanned,
        "fp8_weight_bytes_est": fp8_weight_bytes,
        "scale_weight_elems": total_scale_elems,
        "scale_weight_bytes_est": scale_weight_bytes,
        "index_total_size_bytes": index_total_size,
        "entropy": {
            "full_byte_bits": entropy_byte,
            "exponent_bits": entropy_exp,
            "sign_mantissa_bits": entropy_sm,
        },
        "dfloat_style_estimate": {
            "sign_mantissa_fixed_bits": 1 + fp8_layout.mantissa_bits,
            "exponent_entropy_bits": entropy_exp,
            "estimated_bits_per_weight": estimated_bits_per_weight,
            "estimated_ratio_vs_raw_fp8": estimated_bits_per_weight / 8.0,
        },
        "elapsed_seconds": elapsed_s,
    }

    print(f"[ok] ckpt_dir={ckpt_dir}")
    print(f"[ok] dtype={fp8_layout.name} sample_stride={args.sample_stride}")
    print(f"[ok] fp8_tensors={fp8_tensors} non_fp8_tensors={non_fp8_tensors}")
    print(
        "[ok] fp8_elems="
        f"seen={total_fp8_elems_seen:,} scanned={total_fp8_elems_scanned:,} (stride={args.sample_stride})"
    )
    print(f"[ok] fp8_weight_bytes_est={fp8_weight_bytes:,} ({_human_bytes(fp8_weight_bytes)})")
    print(f"[ok] scale_weight_bytes_est={scale_weight_bytes:,} ({_human_bytes(scale_weight_bytes)})")
    if index_total_size is not None:
        print(f"[ok] index_total_size_bytes={index_total_size:,} ({_human_bytes(index_total_size)})")
    print(
        "[entropy] full_byte_bits="
        f"{entropy_byte:.4f} (vs 8.0000 raw), exponent_bits={entropy_exp:.4f} (vs {fp8_layout.exponent_bits}), "
        f"sign_mantissa_bits={entropy_sm:.4f} (vs {1 + fp8_layout.mantissa_bits})"
    )
    print(
        "[estimate] dfloat-style bits/weight="
        f"{estimated_bits_per_weight:.4f} "
        f"(fixed {1 + fp8_layout.mantissa_bits} + exp_entropy {entropy_exp:.4f}), "
        f"ratio_vs_raw_fp8={estimated_bits_per_weight/8.0:.4f}"
    )

    if args.json_out:
        out_path = _resolve_dir_from_repo_root(repo_root, args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[ok] wrote_json={out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

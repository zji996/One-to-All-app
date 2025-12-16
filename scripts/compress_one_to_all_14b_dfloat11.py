from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_from_repo_root(repo_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def _copy_file(src: Path, dst: Path, *, strategy: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if strategy == "symlink":
        dst.symlink_to(src)
        return

    if strategy == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass

    shutil.copy2(src, dst)


def _sync_tree(src_root: Path, dst_root: Path, *, strategy: str) -> None:
    if not src_root.is_dir():
        raise FileNotFoundError(f"source dir not found: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(src_root):
        dirpath = Path(dirpath)
        rel = dirpath.relative_to(src_root)

        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", "._____temp"}]
        if rel.parts and any(p in {".git", "__pycache__", "._____temp"} for p in rel.parts):
            continue

        out_dir = dst_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        for name in filenames:
            if name in {".DS_Store"}:
                continue
            _copy_file(dirpath / name, out_dir / name, strategy=strategy)


def _remove_matching(root: Path, *, patterns: list[str]) -> int:
    removed = 0
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file() or p.is_symlink():
                p.unlink()
                removed += 1
    return removed


@contextmanager
def _third_party_video_generation_on_syspath() -> Path:
    repo_root = _repo_root()
    video_generation_dir = repo_root / "third_party" / "One-to-All-Animation" / "video-generation"
    if not video_generation_dir.is_dir():
        raise FileNotFoundError(
            "One-to-All-Animation video-generation dir not found: "
            f"{video_generation_dir} (did you init submodules?)"
        )
    old = list(sys.path)
    sys.path.insert(0, str(video_generation_dir))
    try:
        yield video_generation_dir
    finally:
        sys.path[:] = old


def _build_pattern_dict_for_modules(module_names: list[str]) -> dict[str, list[str]]:
    patterns: set[str] = set()
    for name in module_names:
        parts = []
        for seg in name.split("."):
            if seg.isdigit():
                parts.append(r"\d+")
            else:
                parts.append(re.escape(seg))
        patterns.add(r"\.".join(parts))
    return {p: [] for p in sorted(patterns)}


def _build_dfloat11_luts_from_table(table) -> "torch.Tensor":
    import numpy as np
    import torch

    prefixes: list[str] = [""]
    for key, (bits, val) in table.items():
        if not isinstance(key, int):
            continue
        if bits <= 0:
            continue
        prefix = bin(val)[2:].rjust(bits, "0")[: ((bits - 1) // 8 * 8)]
        if prefix not in prefixes:
            prefixes.append(prefix)
    prefixes.sort(key=len)

    luts = np.zeros((len(prefixes), 256), dtype=np.uint8)
    for prefix_index, prefix in enumerate(prefixes):
        bytes_dict: dict[int, int] = {}
        prefix_len_bytes = len(prefix) // 8

        for key, (bits, val) in table.items():
            if not isinstance(key, int):
                continue
            bin_val = bin(val)[2:].rjust(bits, "0")
            if not bin_val.startswith(prefix):
                continue

            if (bits - 1) // 8 == prefix_len_bytes:
                dict_key = int(bin_val[(prefix_len_bytes * 8) :].ljust(8, "0"), 2)
                dict_value = key
            else:
                dict_key = int(bin_val[(prefix_len_bytes * 8) : (prefix_len_bytes * 8 + 8)], 2)
                dict_value = 256 - prefixes.index(bin_val[: (prefix_len_bytes * 8 + 8)])

            prev = bytes_dict.get(dict_key)
            if prev is not None and prev != dict_value:
                raise ValueError(f"LUT conflict at prefix={prefix!r} key={dict_key}: {prev} != {dict_value}")
            bytes_dict[dict_key] = dict_value

        curr_val = 0
        for i in range(256):
            if i in bytes_dict:
                curr_val = bytes_dict[i]
            luts[prefix_index, i] = curr_val

    lens = np.zeros((1, 256), dtype=np.uint8)
    for key, (bits, _val) in table.items():
        if isinstance(key, int):
            lens[-1, key] = bits

    return torch.from_numpy(np.concatenate((luts, lens), axis=0))


def _get_32bit_huffman_codec(counter: dict[int, int]):
    from copy import copy

    import numpy as np
    from dahuffman import HuffmanCodec

    codec = HuffmanCodec.from_frequencies(counter)
    table = codec.get_code_table()
    max_len = max((l for k, (l, _v) in table.items() if isinstance(k, int)), default=0)
    if max_len <= 32:
        return codec

    min_k = 2
    freq = np.array(list(counter.values()))
    while max_len > 32:
        min_indices = np.argpartition(freq, min_k)[:min_k]
        min_k += 1
        min_keys = np.array(list(counter.keys()))[min_indices]

        compressed_counter = copy(counter)
        for k in min_keys:
            compressed_counter[int(k)] = 1
        codec = HuffmanCodec.from_frequencies(compressed_counter)
        table = codec.get_code_table()
        max_len = max((l for k, (l, _v) in table.items() if isinstance(k, int)), default=0)

    return codec


def _encode_exponents_u8_numpy(
    symbols_u8,
    *,
    codec,
    bytes_per_thread: int,
    threads_per_block: int,
):
    import numpy as np

    encoded = bytearray()
    gaps: list[int] = []
    output_positions: list[int] = []

    buffer = 0
    size = 0
    total_size = 0
    element_count = 0

    chunk_bits = 8 * int(bytes_per_thread)
    block_bits = chunk_bits * int(threads_per_block)

    for sym in symbols_u8:
        if total_size // chunk_bits + 1 > len(gaps):
            gaps.append(total_size - (total_size // chunk_bits) * chunk_bits)
        if total_size // block_bits + 1 > len(output_positions):
            output_positions.append(element_count)

        b, v = codec._table[int(sym)]
        buffer = (buffer << b) + v
        size += b
        total_size += b
        element_count += 1

        while size >= 8:
            byte = buffer >> (size - 8)
            encoded.append(byte & 0xFF)
            buffer -= byte << (size - 8)
            size -= 8

    if size > 0:
        if total_size // chunk_bits + 1 > len(gaps):
            gaps.append(total_size - (total_size // chunk_bits) * chunk_bits)
        if total_size // block_bits + 1 > len(output_positions):
            output_positions.append(element_count)

        b, v = codec._table[codec._eof]
        buffer = (buffer << b) + v
        size += b
        if size >= 8:
            byte = buffer >> (size - 8)
        else:
            byte = buffer << (8 - size)
        encoded.append(byte & 0xFF)

    output_positions.append(int(len(symbols_u8)))

    encoded_u8 = np.frombuffer(encoded, dtype=np.uint8)
    blocks_per_grid = int(np.ceil(encoded_u8.size / (int(threads_per_block) * int(bytes_per_thread)))) or 1
    n_threads = int(threads_per_block) * blocks_per_grid
    if len(gaps) < n_threads:
        gaps.extend([0] * (n_threads - len(gaps)))

    gap_bits = np.empty(len(gaps) * 5, dtype=np.uint8)
    for i, gap in enumerate(gaps):
        g = int(gap) & 0x1F
        gap_bits[i * 5 + 0] = (g >> 4) & 1
        gap_bits[i * 5 + 1] = (g >> 3) & 1
        gap_bits[i * 5 + 2] = (g >> 2) & 1
        gap_bits[i * 5 + 3] = (g >> 1) & 1
        gap_bits[i * 5 + 4] = g & 1

    packed_gaps = np.packbits(gap_bits, bitorder="big")
    output_positions_u32 = np.array(output_positions, dtype=np.uint32)
    return encoded_u8, packed_gaps, output_positions_u32


def _encode_bf16_weight_to_df11_tensors(weight_bf16, *, bytes_per_thread: int, threads_per_block: int):
    import numpy as np
    import torch

    if weight_bf16.dtype != torch.bfloat16:
        raise TypeError(f"Expected bf16 weight, got {weight_bf16.dtype}")

    w_int16 = weight_bf16.view(torch.int16)
    exponent_u8 = ((w_int16 >> 7) & 0xFF).to(torch.uint8).contiguous()
    sign_mantissa_u8 = (((w_int16 >> 8) & 0x80) | (w_int16 & 0x7F)).to(torch.uint8).contiguous()

    # Count exponent frequencies (vectorized).
    vals, freqs = torch.unique(exponent_u8, return_counts=True)
    counter = {int(v): int(f) for v, f in zip(vals.tolist(), freqs.tolist())}

    codec = _get_32bit_huffman_codec(counter)
    table = codec.get_code_table()
    luts = _build_dfloat11_luts_from_table(table)

    encoded_u8, packed_gaps_u8, output_positions_u32 = _encode_exponents_u8_numpy(
        exponent_u8.cpu().numpy().reshape(-1),
        codec=codec,
        bytes_per_thread=int(bytes_per_thread),
        threads_per_block=int(threads_per_block),
    )

    encoded = torch.from_numpy(np.ascontiguousarray(encoded_u8))
    gaps = torch.from_numpy(np.ascontiguousarray(packed_gaps_u8))
    output_positions = torch.from_numpy(np.ascontiguousarray(output_positions_u32)).view(torch.uint8)
    split_positions = torch.empty((0,), dtype=torch.int64)

    return {
        "luts": luts,
        "encoded_exponent": encoded,
        "sign_mantissa": sign_mantissa_u8.cpu(),
        "output_positions": output_positions,
        "gaps": gaps,
        "split_positions": split_positions,
    }


def _compress_safetensors_dir_to_df11(
    *,
    src_dir: Path,
    dst_dir: Path,
    compressible_module_names: set[str],
    pattern_dict: dict[str, list[str]],
    bytes_per_thread: int = 8,
    threads_per_block: int = 512,
    flush_bytes: int = 1_000_000_000,
) -> None:
    import torch
    from safetensors.torch import safe_open, save_file

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(src_dir.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No .safetensors files found under: {src_dir}")

    compressed = 0
    kept_weights = 0
    out_part = 0
    out_tensors: dict[str, "torch.Tensor"] = {}
    out_size = 0

    def _flush() -> None:
        nonlocal out_part, out_tensors, out_size
        if not out_tensors:
            return
        out_path = dst_dir / f"model_df11-part-{out_part:05d}.safetensors"
        save_file(out_tensors, str(out_path))
        out_part += 1
        out_tensors = {}
        out_size = 0

    for shard_path in shard_paths:
        print(f"[df11] reading shard: {shard_path.name}")
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith(".weight"):
                    module_name = key[: -len(".weight")]
                    if module_name in compressible_module_names:
                        w = f.get_tensor(key)
                        if w.dtype == torch.bfloat16:
                            df11 = _encode_bf16_weight_to_df11_tensors(
                                w, bytes_per_thread=bytes_per_thread, threads_per_block=threads_per_block
                            )
                            for name, tensor in df11.items():
                                out_tensors[f"{module_name}.{name}"] = tensor
                                out_size += int(tensor.nbytes)
                            if out_size >= flush_bytes:
                                _flush()
                            compressed += 1
                            if compressed % 200 == 0:
                                print(f"[df11] compressed_weights={compressed} kept_weights={kept_weights}")
                            continue

                t = f.get_tensor(key)
                out_tensors[key] = t
                out_size += int(t.nbytes)
                if key.endswith(".weight"):
                    kept_weights += 1
                if out_size >= flush_bytes:
                    _flush()

        _flush()

    _flush()

    config_path = dst_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "dfloat11_config": {
                    "version": "0.5.0",
                    "threads_per_block": (int(threads_per_block),),
                    "bytes_per_thread": int(bytes_per_thread),
                    "pattern_dict": pattern_dict,
                }
            },
            fp,
            indent=2,
        )


def _build_one_to_all_transformer_compressible_modules() -> tuple[set[str], dict[str, list[str]]]:
    import torch
    import torch.nn as nn
    from accelerate import init_empty_weights

    with _third_party_video_generation_on_syspath() as video_generation_dir:
        from opensora.model_variants.wanx_diffusers_src import (
            WanTransformer3DModel_Refextractor_2D_Controlnet_prefix,
        )

        config_path = video_generation_dir / "configs" / "wan2.1_t2v_14b.json"
        controlnet_cfg = video_generation_dir / "configs" / "wan2.1_t2v_14b_controlnet_1.json"
        refextractor_cfg = (
            video_generation_dir
            / "configs"
            / "wan2.1_t2v_14b_refextractor_2d_withmask2.json"
        )

        with init_empty_weights():
            model = WanTransformer3DModel_Refextractor_2D_Controlnet_prefix.from_config(str(config_path))
            model.set_up_controlnet(str(controlnet_cfg), torch.bfloat16)
            model.set_up_refextractor(str(refextractor_cfg), torch.bfloat16)

    names = {
        full_name
        for full_name, sub_module in model.named_modules()
        if isinstance(sub_module, (nn.Linear, nn.Embedding))
    }
    return names, _build_pattern_dict_for_modules(sorted(names))


def _build_umt5_encoder_compressible_modules(text_encoder_dir: Path) -> tuple[set[str], dict[str, list[str]]]:
    import torch.nn as nn
    from accelerate import init_empty_weights
    from transformers import AutoConfig, UMT5EncoderModel

    cfg = AutoConfig.from_pretrained(str(text_encoder_dir))
    with init_empty_weights():
        model = UMT5EncoderModel(cfg)

    names = {
        full_name
        for full_name, sub_module in model.named_modules()
        if isinstance(sub_module, (nn.Linear, nn.Embedding))
    }
    return names, _build_pattern_dict_for_modules(sorted(names))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compress a local One-to-All-14b model repo into a DF11-exported repo directory.\n\n"
            "Run via worker env:\n"
            "  uv sync --project apps/worker --extra one_to_all_animation --extra df11\n"
            "  uv run --project apps/worker python scripts/compress_one_to_all_14b_dfloat11.py\n"
        )
    )
    parser.add_argument(
        "--ckpt-dir",
        default=os.environ.get("ONE_TO_ALL_14B_DIR") or "models/One-to-All-14b",
        help="Source One-to-All-14b directory (default: models/One-to-All-14b).",
    )
    parser.add_argument(
        "--save-dir",
        default=os.environ.get("ONE_TO_ALL_14B_DF11_DIR") or "models/One-to-All-14b-DF11",
        help="Target output directory (default: models/One-to-All-14b-DF11).",
    )
    parser.add_argument(
        "--strategy",
        choices=["copy", "hardlink", "symlink"],
        default="hardlink",
        help="How to place non-generated files: copy | hardlink | symlink (default: hardlink).",
    )
    parser.add_argument("--skip-transformer", action="store_true", help="Skip One-to-All transformer DF11 export.")
    parser.add_argument("--skip-text-encoder", action="store_true", help="Skip Wan text_encoder DF11 export.")
    parser.add_argument(
        "--keep-bf16",
        action="store_true",
        help="Keep original BF16/FP32 safetensors shards in the exported repo (default: remove when DF11 exists).",
    )
    parser.add_argument(
        "--flush-bytes",
        type=int,
        default=1_000_000_000,
        help="Flush safetensors part when buffered tensors exceed this size (default: 1e9).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing save-dir (deletes it first).",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    src_dir = _resolve_from_repo_root(repo_root, str(args.ckpt_dir))
    dst_dir = _resolve_from_repo_root(repo_root, str(args.save_dir))

    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source model dir not found: {src_dir}")

    if dst_dir.exists() and args.overwrite:
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"[copy] {src_dir} -> {dst_dir} (strategy={args.strategy})")
    _sync_tree(src_dir, dst_dir, strategy=str(args.strategy))

    wan_dir = src_dir / "pretrained_models" / "Wan2.1-T2V-14B-Diffusers"
    if not wan_dir.is_dir():
        raise FileNotFoundError(f"Missing Wan base model dir: {wan_dir}")

    # 1) One-to-All finetuned transformer (checkpoint root)
    transformer_df11_dir = dst_dir / "transformer_df11"
    if not args.skip_transformer:
        print("[df11] building compressible module list for One-to-All transformer (meta init)")
        names, pattern_dict = _build_one_to_all_transformer_compressible_modules()
        print(f"[df11] transformer compressible_modules={len(names)} patterns={len(pattern_dict)}")

        print(f"[df11] exporting One-to-All transformer -> {transformer_df11_dir}")
        _compress_safetensors_dir_to_df11(
            src_dir=src_dir,
            dst_dir=transformer_df11_dir,
            compressible_module_names=names,
            pattern_dict=pattern_dict,
            flush_bytes=int(args.flush_bytes),
        )

        removed = _remove_matching(dst_dir, patterns=["model-*.safetensors", "model.safetensors.index.json"])
        print(f"[clean] removed root bf16 shards in export_dir: {removed}")
    elif not args.keep_bf16 and transformer_df11_dir.is_dir():
        removed = _remove_matching(dst_dir, patterns=["model-*.safetensors", "model.safetensors.index.json"])
        if removed:
            print(f"[clean] removed root bf16 shards (df11 already present): {removed}")

    # 2) Wan text_encoder
    if not args.skip_text_encoder:
        te_src_dir = wan_dir / "text_encoder"
        te_dst_dir = dst_dir / "pretrained_models" / "Wan2.1-T2V-14B-Diffusers" / "text_encoder_df11"
        print("[df11] building compressible module list for Wan text_encoder (meta init)")
        te_names, te_pattern_dict = _build_umt5_encoder_compressible_modules(te_src_dir)
        print(f"[df11] text_encoder compressible_modules={len(te_names)} patterns={len(te_pattern_dict)}")

        print(f"[df11] exporting Wan text_encoder -> {te_dst_dir}")
        _compress_safetensors_dir_to_df11(
            src_dir=te_src_dir,
            dst_dir=te_dst_dir,
            compressible_module_names=te_names,
            pattern_dict=te_pattern_dict,
            flush_bytes=int(args.flush_bytes),
        )

        te_export_root = dst_dir / "pretrained_models" / "Wan2.1-T2V-14B-Diffusers" / "text_encoder"
        removed = _remove_matching(te_export_root, patterns=["model-*.safetensors", "model.safetensors.index.json"])
        print(f"[clean] removed bf16 shards in export text_encoder/: {removed}")
    else:
        te_df11_dir = dst_dir / "pretrained_models" / "Wan2.1-T2V-14B-Diffusers" / "text_encoder_df11"
        te_export_root = dst_dir / "pretrained_models" / "Wan2.1-T2V-14B-Diffusers" / "text_encoder"
        if not args.keep_bf16 and te_df11_dir.is_dir():
            removed = _remove_matching(te_export_root, patterns=["model-*.safetensors", "model.safetensors.index.json"])
            if removed:
                print(f"[clean] removed bf16 shards in export text_encoder/ (df11 already present): {removed}")

    print("[done] ok")
    print(f"[done] save_dir={dst_dir}")
    print("[hint] To use DF11 in worker:")
    print(f"  ONE_TO_ALL_MODEL_DIR={dst_dir.name}")
    print("  ONE_TO_ALL_USE_DF11=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

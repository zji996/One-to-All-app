# one_to_all_worker

## Dev run

1. Start dependencies (Redis + Postgres)
   - `docker compose -f infra/docker-compose.dev.yml up -d redis postgres`

2. Install deps
   - `uv sync --project apps/worker`

3. Start worker
   - `uv run --project apps/worker python main.py`

## Use pip (inside project venv)

- `uv run --project apps/worker pip list`
- `uv run --project apps/worker python -m pip list`

## Enable One-to-All-Animation inference deps (optional)

This is heavy and follows upstream requirements:

- `uv sync --project apps/worker --extra one_to_all_animation`

Torch / flash-attn installation should follow `third_party/One-to-All-Animation/README.md`.

## One-to-All-Animation pretrained models (required for inference)

Upstream inference also needs the base Wan2.1 Diffusers weights + pose preprocess checkpoints.

- Download into `models/One-to-All-Animation/pretrained_models/`:
  - `uv sync --project apps/api --extra model_download`
  - If you already downloaded into the submodule, migrate first:
    - `uv run --project apps/api scripts/download_one_to_all_animation_pretrained.py --move-from-third-party`
  - Download (or refresh missing files):
    - `uv run --project apps/api scripts/download_one_to_all_animation_pretrained.py --with-wan-14b`

## FP8 (E5M2) quantize checkpoints (optional)

This creates a new checkpoint folder with `-FP8` suffix under `MODELS_DIR`, intended for fp8_scaled-style
loading where `*.weight` tensors are stored as FP8 and `*.scale_weight` is added per layer.

1. Install deps in worker env (uses latest PyTorch unless you pin it yourself):
   - `uv sync --project apps/worker --extra fp8_quant`

2. Quantize:
   - `uv run --project apps/worker scripts/quantize_one_to_all_fp8.py --model 14b --dtype e5m2`

Output example: `models/One-to-All-14b-FP8/`.
Embedding-like weights (`*.embeddings.weight`, `*.patch_embedding.weight`) are skipped by default; pass
`--no-skip-embedding-like` to include them.

## Analyze FP8 compressibility (optional)

This scans FP8 weights (byte-level) and reports exponent/sign+mantissa entropy (DFloat-style), to estimate
how much additional *lossless* compression might exist in the stored FP8 bytes.

- Full scan (slow on 14B): `uv run --project apps/worker scripts/analyze_one_to_all_fp8_compressibility.py --model 14b`
- Quick sample (approx): `uv run --project apps/worker scripts/analyze_one_to_all_fp8_compressibility.py --model 14b --sample-stride 16`

## Env

Copy `env.example` to `.env` (do not commit `.env`).

## Checkpoints

Worker reads checkpoints from `MODELS_DIR/ONE_TO_ALL_MODEL_DIR` (defaults to `models/One-to-All-14b-FP8`).

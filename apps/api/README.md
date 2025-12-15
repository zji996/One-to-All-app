# one_to_all_api

## Dev run

1. Start dependencies (Redis + Postgres)
   - `docker compose -f infra/docker-compose.dev.yml up -d redis postgres`

2. Install deps
   - `uv sync --project apps/api`

3. Run API
   - `uv run --project apps/api uvicorn main:app --reload --host 0.0.0.0 --port 8000`

## Use pip (inside project venv)

- `uv run --project apps/api pip list`
- `uv run --project apps/api python -m pip list`

## Create a job

- `POST /jobs/one-to-all` with JSON body:
  - `reference_image`: local path or S3 key/URL
  - `motion_video`: local path or S3 key/URL
  - `output_s3_key` (optional): if set and S3 configured, worker uploads result there
- Or upload files: `POST /jobs/one-to-all/upload` (multipart)
  - After success: `GET /jobs/{task_id}/download` (returns mp4 if the API host can access `DATA_DIR`)

## Env

Copy `env.example` to `.env` (do not commit `.env`).

## Download checkpoints (ModelScope)

- Install optional deps: `uv sync --project apps/api --extra model_download`
- Download (skips if already present): `uv run --project apps/api scripts/download_modelscope_one_to_all.py --model 14b`

## Quantize checkpoints to FP8 (optional)

- Install deps in worker env: `uv sync --project apps/worker --extra fp8_quant`
- Quantize to `models/<name>-FP8/`: `uv run --project apps/worker scripts/quantize_one_to_all_fp8.py --model 14b --dtype e5m2`

## One-to-All-Animation pretrained models (required for inference)

Upstream inference also needs the base Wan2.1 Diffusers weights + pose preprocess checkpoints.

- Download into `models/One-to-All-Animation/pretrained_models/`:
  - `uv sync --project apps/api --extra model_download`
  - If you already downloaded into the submodule, migrate first:
    - `uv run --project apps/api scripts/download_one_to_all_animation_pretrained.py --move-from-third-party`
  - Download (or refresh missing files):
    - `uv run --project apps/api scripts/download_one_to_all_animation_pretrained.py --with-wan-14b`

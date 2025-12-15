from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from sqlalchemy import select

from py_core.celery_app import celery_app
from py_core.db import GenerationJob
from py_core.db_session import db_session
from py_core.one_to_all_model import resolve_one_to_all_checkpoint_dir
from py_core.s3 import download_file, s3_is_configured, upload_file
from py_core.settings import settings


@celery_app.task(name="one_to_all.run_one_to_all", bind=True)
def run_one_to_all(self, payload: dict[str, Any]) -> dict[str, Any]:
    """
    Run One-to-All-Animation inference via `third_party/One-to-All-Animation`.
    """

    task_id = getattr(self.request, "id", None)
    if task_id:
        try:
            with db_session() as session:
                job = session.scalar(
                    select(GenerationJob).where(GenerationJob.celery_task_id == task_id)
                )
                if job is None:
                    session.add(
                        GenerationJob(
                            celery_task_id=task_id,
                            status="STARTED",
                            request_payload=payload,
                        )
                    )
                else:
                    job.status = "STARTED"
        except Exception:
            pass

    try:
        model_dir = resolve_one_to_all_checkpoint_dir(payload.get("model_name"))

        data_dir = Path(settings.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        run_id = str(int(time.time() * 1000))
        job_dir = data_dir / "one_to_all" / "jobs" / (task_id or run_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        def _ensure_local_input(value: str, *, name: str) -> Path:
            value = value.strip()
            if not value:
                raise ValueError(f"Missing required field: {name}")

            if value.startswith("s3://"):
                # s3://bucket/key -> key
                parts = value[5:].split("/", 1)
                if len(parts) == 2:
                    value = parts[1]

            local_path = Path(value)
            if local_path.exists():
                return local_path.resolve()

            if not s3_is_configured():
                raise FileNotFoundError(
                    f"{name} not found locally ({value}) and S3 is not configured"
                )

            key = value
            filename = Path(key).name or f"{name}.bin"
            target = job_dir / filename
            download_file(key, str(target))
            return target

        reference_image = _ensure_local_input(
            str(payload.get("reference_image") or ""), name="reference_image"
        )
        motion_video = _ensure_local_input(
            str(payload.get("motion_video") or ""), name="motion_video"
        )

        extra = payload.get("extra") or {}
        if not isinstance(extra, dict):
            extra = {}

        try:
            from py_core.one_to_all_animation_infer import (
                OneToAllAnimationRunConfig,
                run_one_to_all_animation_inference,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing worker inference dependencies; install extras:\n"
                "  uv sync --project apps/worker --extra one_to_all_animation"
            ) from exc

        cfg = OneToAllAnimationRunConfig(
            reference_image_path=reference_image,
            motion_video_path=motion_video,
            model_name=payload.get("model_name"),
            prompt=str(extra.get("prompt") or ""),
            align_mode=str(extra.get("align_mode") or "ref"),
            frame_interval=int(extra.get("frame_interval") or 1),
            do_align=bool(extra.get("do_align") if "do_align" in extra else True),
            image_guidance_scale=float(extra.get("image_guidance_scale") or 2.0),
            pose_guidance_scale=float(extra.get("pose_guidance_scale") or 1.5),
            num_inference_steps=int(extra.get("num_inference_steps") or 30),
            seed=int(extra.get("seed") or 42),
            device=str(extra.get("device") or os.environ.get("ONE_TO_ALL_DEVICE") or ""),
        )

        result: dict[str, Any] = run_one_to_all_animation_inference(cfg)
        result["one_to_all_model_dir"] = str(model_dir)
        result["one_to_all_model_exists"] = model_dir.exists()

        requested_key = payload.get("output_s3_key")
        if s3_is_configured():
            local_video_path = result.get("local_video_path")
            if isinstance(local_video_path, str) and local_video_path:
                if requested_key:
                    key = str(requested_key)
                else:
                    key = f"{settings.s3_prefix}/results/{Path(local_video_path).name}"
                upload_file(local_video_path, key)
                result["s3_key"] = key
        else:
            result["note"] = "S3 not configured; result is stored locally under DATA_DIR."

        # Keep a tiny local metadata record for debugging.
        (job_dir / "request.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (job_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if task_id:
            try:
                with db_session() as session:
                    job = session.scalar(
                        select(GenerationJob).where(GenerationJob.celery_task_id == task_id)
                    )
                    if job is None:
                        session.add(
                            GenerationJob(
                                celery_task_id=task_id,
                                status="SUCCESS",
                                request_payload=payload,
                                result_payload=result,
                            )
                        )
                    else:
                        job.status = "SUCCESS"
                        job.result_payload = result
                        job.error = None
            except Exception:
                pass

        return result
    except Exception as exc:
        if task_id:
            try:
                with db_session() as session:
                    job = session.scalar(
                        select(GenerationJob).where(GenerationJob.celery_task_id == task_id)
                    )
                    if job is None:
                        session.add(
                            GenerationJob(
                                celery_task_id=task_id,
                                status="FAILURE",
                                request_payload=payload,
                                error=str(exc),
                            )
                        )
                    else:
                        job.status = "FAILURE"
                        job.error = str(exc)
            except Exception:
                pass
        raise

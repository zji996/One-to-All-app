from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from celery.result import AsyncResult
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from py_core.celery_app import celery_app
from py_core.db import GenerationJob
from py_core.db_session import db_session, init_db
from py_core.settings import settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        init_db()
    except Exception:
        pass
    yield


app = FastAPI(title="One-to-All API", version="0.1.0", lifespan=lifespan)


class OneToAllJobRequest(BaseModel):
    reference_image: str = Field(..., description="Local path or S3 key/URL")
    motion_video: str = Field(..., description="Local path or S3 key/URL")
    model_name: str | None = Field(
        default=None, description="e.g. 1.3b_1 / 1.3b_2 / 14b"
    )
    prompt: str = Field(default="", description="Optional text prompt")
    align_mode: str = Field(default="ref", description="ref | pose")
    frame_interval: int = Field(default=1, description="Sample every N frames (>=1)")
    do_align: bool = Field(default=True, description="Whether to retarget poses")
    image_guidance_scale: float = Field(default=2.0)
    pose_guidance_scale: float = Field(default=1.5)
    num_inference_steps: int = Field(default=30)
    seed: int = Field(default=42)
    device: str | None = Field(default=None, description="e.g. cuda:0 / cpu")
    output_s3_key: str | None = Field(
        default=None, description="If set and S3 configured, upload result to this key."
    )
    extra: dict[str, Any] = Field(default_factory=dict)


class JobCreateResponse(BaseModel):
    task_id: str


class JobStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Any | None = None
    created_at: str | None = None
    updated_at: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs/one-to-all", response_model=JobCreateResponse)
def create_one_to_all_job(req: OneToAllJobRequest) -> JobCreateResponse:
    payload = req.model_dump()
    extra = payload.get("extra") or {}
    if not isinstance(extra, dict):
        extra = {}
    extra.setdefault("prompt", req.prompt)
    extra.setdefault("align_mode", req.align_mode)
    extra.setdefault("frame_interval", req.frame_interval)
    extra.setdefault("do_align", req.do_align)
    extra.setdefault("image_guidance_scale", req.image_guidance_scale)
    extra.setdefault("pose_guidance_scale", req.pose_guidance_scale)
    extra.setdefault("num_inference_steps", req.num_inference_steps)
    extra.setdefault("seed", req.seed)
    if req.device:
        extra.setdefault("device", req.device)
    payload["extra"] = extra

    task = celery_app.send_task("one_to_all.run_one_to_all", args=[payload])
    try:
        with db_session() as session:
            session.add(
                GenerationJob(
                    celery_task_id=task.id,
                    status="PENDING",
                    request_payload=payload,
                )
            )
    except Exception:
        pass
    return JobCreateResponse(task_id=task.id)


@app.post("/jobs/one-to-all/upload", response_model=JobCreateResponse)
async def create_one_to_all_job_upload(
    reference_image: UploadFile = File(...),
    motion_video: UploadFile = File(...),
    model_name: str | None = Form(default=None),
    prompt: str = Form(default=""),
    align_mode: str = Form(default="ref"),
    frame_interval: int = Form(default=1),
    do_align: bool = Form(default=True),
    image_guidance_scale: float = Form(default=2.0),
    pose_guidance_scale: float = Form(default=1.5),
    num_inference_steps: int = Form(default=30),
    seed: int = Form(default=42),
    device: str | None = Form(default=None),
    output_s3_key: str | None = Form(default=None),
) -> JobCreateResponse:
    from uuid import uuid4

    data_dir = Path(settings.data_dir)
    upload_dir = data_dir / "one_to_all" / "uploads" / str(uuid4())
    upload_dir.mkdir(parents=True, exist_ok=True)

    def _safe_suffix(filename: str | None) -> str:
        if not filename:
            return ""
        suffix = Path(filename).suffix.lower()
        if len(suffix) > 10:
            return ""
        return suffix

    ref_path = upload_dir / f"reference{_safe_suffix(reference_image.filename)}"
    motion_path = upload_dir / f"motion{_safe_suffix(motion_video.filename)}"

    ref_path.write_bytes(await reference_image.read())
    motion_path.write_bytes(await motion_video.read())

    payload: dict[str, Any] = {
        "reference_image": str(ref_path),
        "motion_video": str(motion_path),
        "model_name": model_name,
        "output_s3_key": output_s3_key,
        "extra": {
            "prompt": prompt,
            "align_mode": align_mode,
            "frame_interval": frame_interval,
            "do_align": do_align,
            "image_guidance_scale": image_guidance_scale,
            "pose_guidance_scale": pose_guidance_scale,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            **({"device": device} if device else {}),
        },
    }

    task = celery_app.send_task("one_to_all.run_one_to_all", args=[payload])
    try:
        with db_session() as session:
            session.add(
                GenerationJob(
                    celery_task_id=task.id,
                    status="PENDING",
                    request_payload=payload,
                )
            )
    except Exception:
        pass
    return JobCreateResponse(task_id=task.id)


@app.get("/jobs/{task_id}", response_model=JobStatusResponse)
def get_job(task_id: str) -> JobStatusResponse:
    res = AsyncResult(task_id, app=celery_app)
    payload: dict[str, Any] = {"task_id": task_id, "status": res.status}
    if res.successful():
        payload["result"] = res.result
    elif res.failed():
        payload["result"] = str(res.result)

    try:
        with db_session() as session:
            job = session.scalar(
                select(GenerationJob).where(GenerationJob.celery_task_id == task_id)
            )
            if job is None:
                job = GenerationJob(
                    celery_task_id=task_id,
                    status=res.status,
                    request_payload={},
                )
                session.add(job)
            job.status = res.status
            if res.successful() and isinstance(res.result, dict):
                job.result_payload = res.result
                job.error = None
            elif res.failed():
                job.error = str(res.result)
            payload["created_at"] = job.created_at.isoformat() if job.created_at else None
            payload["updated_at"] = job.updated_at.isoformat() if job.updated_at else None
    except Exception:
        pass

    return JobStatusResponse(**payload)


@app.get("/jobs/{task_id}/download")
def download_job_result(task_id: str):
    res = AsyncResult(task_id, app=celery_app)
    if not res.successful() or not isinstance(res.result, dict):
        raise HTTPException(status_code=404, detail="result not ready")
    local_video = res.result.get("local_video_path")
    if not isinstance(local_video, str) or not local_video:
        raise HTTPException(status_code=404, detail="no local_video_path in result")
    path = Path(local_video)
    if not path.exists():
        raise HTTPException(status_code=404, detail="local result file missing on server")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


class JobHistoryItem(BaseModel):
    task_id: str
    status: str
    created_at: str | None = None
    updated_at: str | None = None
    request_payload: dict[str, Any]
    result_payload: dict[str, Any] | None = None
    error: str | None = None


@app.get("/jobs", response_model=list[JobHistoryItem])
def list_jobs(limit: int = 50) -> list[JobHistoryItem]:
    limit = max(1, min(limit, 200))
    try:
        with db_session() as session:
            jobs = session.scalars(
                select(GenerationJob).order_by(GenerationJob.created_at.desc()).limit(limit)
            ).all()
            return [
                JobHistoryItem(
                    task_id=j.celery_task_id,
                    status=j.status,
                    created_at=j.created_at.isoformat() if j.created_at else None,
                    updated_at=j.updated_at.isoformat() if j.updated_at else None,
                    request_payload=j.request_payload or {},
                    result_payload=j.result_payload,
                    error=j.error,
                )
                for j in jobs
            ]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}") from exc

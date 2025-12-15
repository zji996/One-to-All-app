from __future__ import annotations

from celery import Celery

from py_core.settings import settings

celery_app = Celery(
    "one_to_all",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_default_queue=settings.celery_default_queue,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

celery_app.autodiscover_tasks(["py_core.tasks"])


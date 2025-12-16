from __future__ import annotations

from py_core.celery_app import celery_app

# Ensure tasks are registered when the worker starts.
import tasks  # noqa: F401

__all__ = ["celery_app"]


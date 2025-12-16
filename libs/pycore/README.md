# py_core

Shared Python code for this monorepo:

- Environment settings (`py_core.config.settings`; legacy: `py_core.settings`)
- Celery app wiring (`py_core.workers.celery_app`; legacy: `py_core.celery_app`)
- DB models/session (`py_core.persistence.*`; legacy: `py_core.db`, `py_core.db_session`)
- Optional S3 helpers (`py_core.storage.s3`; legacy: `py_core.s3`)
- One-to-All model helpers (`py_core.one_to_all.model`; legacy: `py_core.one_to_all_model`)

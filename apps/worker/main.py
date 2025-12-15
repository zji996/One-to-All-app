from __future__ import annotations

import sys

from py_core.celery_app import celery_app


def _has_loglevel_flag(argv: list[str]) -> bool:
    return any(a in ("-l", "--loglevel") or a.startswith("--loglevel=") for a in argv)


if __name__ == "__main__":
    argv = ["worker"]
    if not _has_loglevel_flag(sys.argv[1:]):
        argv += ["--loglevel=INFO"]
    celery_app.worker_main(argv=argv + sys.argv[1:])


from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


def infer_default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


@contextmanager
def torch_default_dtype(dtype) -> Iterator[None]:
    try:
        import torch
    except Exception:
        yield
        return

    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


from py_core.one_to_all_model import resolve_one_to_all_checkpoint_dir
from py_core.settings import settings


def test_default_checkpoint_dir_is_fp8() -> None:
    assert settings.one_to_all_model_dir == "One-to-All-14b-FP8"
    assert resolve_one_to_all_checkpoint_dir(None).name == "One-to-All-14b-FP8"

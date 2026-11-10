"""Dung backend khong giao dien cho kiem thu hinh tren Windows va CI."""
import os
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(autouse=True)
def isolate_models_dir(tmp_path, monkeypatch):
    """Bao ve thu muc models/ that cua du an khoi bi pytest ghi de."""
    import src.utils.helpers as helpers

    test_models = tmp_path / "test_models"
    test_models.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(helpers, "MODELS_DIR", test_models)

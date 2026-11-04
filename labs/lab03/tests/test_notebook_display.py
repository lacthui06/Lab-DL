"""Notebook phải nhận image outputs, đồng thời giữ nguyên artifacts nguồn."""
import importlib.util
import json
from IPython.display import Image
import pytest
from labs.lab03.src.utils.runs import ROOT, hash_file


def _module():
    spec = importlib.util.spec_from_file_location("lab03_display", ROOT / "scripts/display_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_results_display_images_inside_notebook_without_mutation(monkeypatch):
    module = _module()
    published = []
    monkeypatch.setattr(module, "display", published.append)
    before = {str(p): hash_file(p) for p in (ROOT / "outputs/runs").rglob("*") if p.is_file()}
    shown = module.show_results(ROOT)
    images = [item for item in published if isinstance(item, Image)]
    assert len(images) == len(shown) >= 8
    assert all(item.data.startswith(b'\x89PNG') for item in images)
    assert any("validation_threshold" in p.parts for p in shown)
    assert any("test_threshold" in p.parts for p in shown)
    assert any(p.name == "inference.png" for p in shown)
    assert {str(p): hash_file(p) for p in (ROOT / "outputs/runs").rglob("*") if p.is_file()} == before


def test_display_reports_missing_images_instead_of_silent_success(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/workflow.json").write_text(json.dumps({"active_runs": {}}))
    with pytest.raises(FileNotFoundError, match="No saved PNGs"):
        _module().show_results(tmp_path)

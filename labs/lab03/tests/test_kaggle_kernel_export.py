"""Gói kernel Kaggle chỉ đổi MODE sang train; notebook nguồn phải giữ view."""
import importlib.util
import json

import nbformat
import pytest

from labs.lab03.src.utils.runs import ROOT


def _module():
    spec = importlib.util.spec_from_file_location("lab03_kaggle_export", ROOT / "scripts/export_kaggle_kernel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_only_flips_mode_and_keeps_source_notebook_untouched():
    module = _module()
    source = ROOT / "notebooks/02_modeling.ipynb"
    before = source.read_bytes()
    exported = module.train_notebook(source)
    assert source.read_bytes() == before
    original = nbformat.read(source, as_version=4)
    assert len(exported.cells) == len(original.cells)
    changed = [i for i, (a, b) in enumerate(zip(original.cells, exported.cells)) if a.source != b.source]
    assert changed == [2]
    source_lines = original.cells[2].source.splitlines()
    export_lines = exported.cells[2].source.splitlines()
    assert len(source_lines) == len(export_lines)
    differing = [i for i, (a, b) in enumerate(zip(source_lines, export_lines)) if a != b]
    assert len(differing) == 1
    assert source_lines[differing[0]].strip().startswith('MODE = "view"')
    assert export_lines[differing[0]].strip() == 'MODE = "train"'
    code = [c.source for c in exported.cells if c.cell_type == "code"]
    assert sum('MODE = "train"' in s for s in code) == 1
    assert not any('MODE = "view"' in s for s in code)
    assert exported.metadata["kaggle"]["accelerator"] == "gpu"
    assert exported.metadata["kaggle"]["isInternetEnabled"] is True


def test_export_writes_metadata_with_gpu_and_internet(tmp_path):
    module = _module()
    out = module.export("someone", destination=tmp_path / "kernel")
    notebook = nbformat.read(out / module.CODE_FILE, as_version=4)
    assert any('MODE = "train"' in c.source for c in notebook.cells if c.cell_type == "code")
    meta = json.loads((out / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == f"someone/{module.SLUG}"
    assert meta["code_file"] == module.CODE_FILE
    assert meta["enable_gpu"] is True and meta["enable_internet"] is True
    assert meta["is_private"] is True
    assert meta["dataset_sources"] == [] and meta["kernel_sources"] == []


def test_export_rejects_notebook_without_view_marker(tmp_path):
    module = _module()
    notebook = nbformat.read(ROOT / "notebooks/02_modeling.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.source = cell.source.replace('MODE = "view"', 'MODE = "train"')
    broken = tmp_path / "broken.ipynb"
    nbformat.write(notebook, broken)
    with pytest.raises(ValueError, match="MODE"):
        module.train_notebook(broken)

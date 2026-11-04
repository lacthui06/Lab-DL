"""Gói kernel Kaggle có MODE=train, không mang cached outputs hoặc sửa source."""
import importlib.util
import json

import nbformat
import pytest

from src.utils.runs import ROOT


def _module():
    spec = importlib.util.spec_from_file_location("lab03_kaggle_export", ROOT / "scripts/export_kaggle_kernel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_keeps_clean_train_source_notebook_untouched():
    module = _module()
    source = ROOT / "notebooks/02_modeling.ipynb"
    before = source.read_bytes()
    exported = module.train_notebook(source)
    assert source.read_bytes() == before
    original = nbformat.read(source, as_version=4)
    assert len(exported.cells) == len(original.cells)
    assert all(c.source == o.source.replace('MODE = "view"', 'MODE = "train"') for c,o in zip(exported.cells, original.cells))
    assert all(not c.outputs and c.execution_count is None for c in exported.cells if c.cell_type=='code')
    code = [c.source for c in exported.cells if c.cell_type == "code"]
    assert sum('MODE = "train"' in s for s in code) == 1
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


def test_export_rejects_notebook_without_training_mode(tmp_path):
    module = _module()
    notebook = nbformat.read(ROOT / "notebooks/02_modeling.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.source = cell.source.replace('MODE = "view"', 'MODE = "prepare"').replace('MODE = "train"', 'MODE = "prepare"')
    broken = tmp_path / "broken.ipynb"
    nbformat.write(notebook, broken)
    with pytest.raises(ValueError, match="MODE"):
        module.train_notebook(broken)

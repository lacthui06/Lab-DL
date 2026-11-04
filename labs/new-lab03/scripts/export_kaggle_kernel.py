"""Đóng gói notebook modeling để chạy Kaggle GPU; giữ source và artifact riêng."""
from pathlib import Path
import argparse
import json
import re
import sys

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.runs import ROOT

SOURCE = ROOT / "notebooks/02_modeling.ipynb"
DESTINATION = ROOT / "kaggle"
CODE_FILE = "02_modeling_train.ipynb"
SLUG = "lab03-frozen-distilbert-train"
TITLE = "Lab03 Frozen DistilBERT train"


def train_notebook(source=SOURCE):
    """Trả notebook MODE=train không có cached outputs, không sửa file nguồn."""
    notebook = nbformat.read(source, as_version=4)
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":continue
        pattern = r'^([ \t]*)MODE = "(?:view|train)"[^\n]*'
        cell.source,n = re.subn(pattern,lambda m:m.group(1)+'MODE = "train"',cell.source,flags=re.M)
        count += n
        cell.outputs = []
        cell.execution_count = None
    if count != 1:
        raise ValueError('Notebook cần đúng một khai báo MODE="train" hoặc MODE="view"')
    notebook.metadata["lab03"] = {
        "execution_mode": "train",
        "generated_from": source.relative_to(ROOT).as_posix(),
        "purpose": "Kaggle GPU training; outputs ghi về /kaggle/working",
    }
    notebook.metadata["kaggle"] = {
        "accelerator": "gpu",
        "isInternetEnabled": True,
        "language": "python",
        "sourceType": "notebook",
    }
    nbformat.validate(notebook)
    return notebook


def metadata(username):
    return {
        "id": f"{username}/{SLUG}",
        "title": TITLE,
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }


def export(username, destination=DESTINATION):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    notebook = train_notebook()
    nbformat.write(notebook, destination / CODE_FILE)
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata(username), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True, help="Username Kaggle, ví dụ lacthui06")
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args()
    out = export(args.username, args.destination)
    print("Kernel package:", out)
    for path in sorted(out.iterdir()):
        print(" ", path.name)

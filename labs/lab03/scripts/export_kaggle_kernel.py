"""Sinh gói kernel Kaggle (notebook train + metadata) từ notebook modeling gốc.

Notebook gốc trong notebooks/ giữ MODE="view" để renderer và test không bao giờ
train ngoài ý muốn. Bản Kaggle được sinh ra với MODE="train" và đặt trong
labs/lab03/kaggle/ để `kaggle kernels push` dùng trực tiếp.
"""
from pathlib import Path
import argparse
import json
import sys

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs.lab03.src.utils.runs import ROOT

SOURCE = ROOT / "notebooks/02_modeling.ipynb"
DESTINATION = ROOT / "kaggle"
CODE_FILE = "02_modeling_train.ipynb"
SLUG = "lab03-frozen-distilbert-train"
TITLE = "Lab03 Frozen DistilBERT train"


def train_notebook(source=SOURCE):
    """Trả notebook đã đổi MODE sang train, không sửa file nguồn."""
    notebook = nbformat.read(source, as_version=4)
    replaced = False
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        lines = []
        for line in cell.source.splitlines(keepends=True):
            if line.lstrip().startswith('MODE = "view"'):
                indent = line[: len(line) - len(line.lstrip())]
                line = f'{indent}MODE = "train"\n'
                replaced = True
            lines.append(line)
        cell.source = "".join(lines)
    if not replaced:
        raise ValueError('Không tìm thấy dòng MODE = "view" trong notebook nguồn')
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

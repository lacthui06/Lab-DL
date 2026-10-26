"""Điểm vào train chuẩn Lab01; dùng chung luồng với train_finetune.py."""
from pathlib import Path
import runpy
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("train_finetune.py")), run_name="__main__")

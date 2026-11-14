"""Điều phối chế độ notebook; chỉ import pipeline khi yêu cầu chạy mới."""
from pathlib import Path
import json


def prepare_eda(root: Path, mode: str = "view", data_dir=None) -> Path:
    """View đọc artifact; prepare tạo lại EDA bằng pipeline hiện có."""
    if mode not in {"view", "prepare"}:
        raise ValueError("EDA MODE must be 'view' or 'prepare'.")
    from ..datasets import eda
    if mode == "prepare":
        return eda.build(Path(root), data_dir)
    return eda.context(Path(root))


def prepare_modeling(root: Path, mode: str = "view", config=None, device="auto"):
    """Train tạo run mới; view không import hoặc gọi training."""
    if mode not in {"view", "train"}:
        raise ValueError("Modeling MODE must be 'view' or 'train'.")
    root = Path(root).resolve()
    from . import notebook_views
    if mode == "view":
        return notebook_views.context()
    if config is None:
        raise ValueError("Set TRAIN_CONFIG explicitly before using MODE='train'.")
    config = Path(config)
    config = config if config.is_absolute() else root / config
    if not config.is_file():
        raise FileNotFoundError(f"Training config not found: {config}")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("DEVICE must be 'auto', 'cpu' or 'cuda'.")
    from ..training.experiment import run_suite
    run = run_suite(config, work_dir=root, device_override=device)
    # Dùng đúng run vừa trả về, kể cả khi LAB02_RUN vẫn trỏ tới run cũ.
    return notebook_views.context(run_path=run)


def benchmark_matches(run: Path, benchmark: Path) -> bool:
    """Không dùng benchmark lịch sử cho run mới; không tự chạy lại phép đo."""
    provenance = json.loads((benchmark / 'provenance.json').read_text(encoding='utf-8'))
    if provenance['training_run'] != run.name:
        print(f'Benchmark chưa có cho run {run.name}; bỏ qua evidence của run khác.')
        return False
    return True

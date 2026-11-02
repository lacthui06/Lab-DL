"""Helper utilities for Lab 01: reproducibility, environment, paths, and checkpointing."""

import json
import os
import platform
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Project Base Directories & Device Configuration
# ---------------------------------------------------------------------------
SOURCE_DIR = Path(__file__).resolve().parent.parent.parent
BASE_DIR = Path(os.environ.get(
    "LAB01_WORK_DIR",
    "/kaggle/working" if os.path.exists("/kaggle/working") else str(SOURCE_DIR),
)).resolve()

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
EDA_DIR = OUTPUTS_DIR / "eda"

# Ensure output directories exist
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
EDA_DIR.mkdir(parents=True, exist_ok=True)

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset & Split Constants
RANDOM_SEED = 42
IMAGE_SIZE = 28
VAL_SPLIT = 0.1
SPLIT_VERSION = "split_v2_stratified"
PREPROC_VERSION = "preproc_v2_train_global_std"

CONFIG_DIR = SOURCE_DIR / "config"
PREPROCESSING_CONFIG_PATH = EDA_DIR / "preprocessing_config.json"


def resolve_normalization_stats(
    eda_dir: Optional[Path] = None,
) -> Tuple[Tuple[float, ...], Tuple[float, ...], str]:
    """Loads measured train-set normalization statistics strictly from artifact.

    Args:
        eda_dir: Directory containing preprocessing_config.json (defaults to EDA_DIR).

    Returns:
        (mean_tuple, std_tuple, source_string)
    """
    path = (Path(eda_dir) / "preprocessing_config.json") if eda_dir is not None else PREPROCESSING_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Missing preprocessing artifact at {path}. "
            "Run EDA & Feature Engineering pipeline (run_eda_and_fe_pipeline) first to lock train-set statistics."
        )

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    mean = config["normalization"]["mean"]
    std = config["normalization"]["std"]
    return tuple(float(m) for m in mean), tuple(float(s) for s in std), f"artifact:{path}"





# Class Label Mapping
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


# ---------------------------------------------------------------------------
# Reproducibility & Config Loaders
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Sets random seeds across random, numpy, and torch for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: nn.Module,
    save_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves model state dictionary along with metadata."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, str(save_path))


def load_checkpoint(
    model: nn.Module,
    load_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Loads weights into model from checkpoint file."""
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {load_path}")
    checkpoint = torch.load(str(load_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return checkpoint.get("metadata", {})


def collect_environment_snapshot(save_path: Optional[Path] = None) -> Dict[str, Any]:
    """Records the interpreter and library versions that produced a run."""
    import sys

    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except ImportError:
        sklearn_version = None

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    snapshot = {
        "python_version": sys.version.split()[0],
        "python_version_full": sys.version,
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "platform": platform.platform(),
        "libraries": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn_version,
        },
        "device": device_str,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.is_dir():
            save_path = save_path / "environment_snapshot.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)

    return snapshot

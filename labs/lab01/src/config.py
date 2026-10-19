"""Configuration module for Lab 01: FashionMNIST Classification.
Centralizes paths, hyperparameters, and environment settings.
"""

from pathlib import Path
import json
import os
import warnings
import torch


# Project Base Directories
if os.path.exists("/kaggle/working"):
    BASE_DIR = Path("/kaggle/working")
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
EDA_DIR = OUTPUTS_DIR / "eda"

# Ensure output directories exist
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
EDA_DIR.mkdir(parents=True, exist_ok=True)

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Constants
RANDOM_SEED = 42
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_CLASSES = 10

# Split Constants
VAL_SPLIT = 0.1
SPLIT_VERSION = "split_v2_stratified"
PREPROC_VERSION = "preproc_v2_train_global_std"

# Fallback normalization statistics (used only when artifact is missing).
FALLBACK_MEAN = (0.2860,)
FALLBACK_STD = (0.3205,)

PREPROCESSING_CONFIG_PATH = EDA_DIR / "preprocessing_config.json"


def _resolve_normalization() -> tuple:
    """Loads the measured normalization statistics from artifact, or issues a warning and returns legacy fallback.

    Returns:
        (mean_tuple, std_tuple, source_string) where source_string is either
        "artifact:<path>" or "fallback:legacy".
    """
    path = PREPROCESSING_CONFIG_PATH
    if not path.exists():
        warnings.warn(
            f"Chua tim thay artifact {path}. Su dung thong so fallback ({FALLBACK_MEAN}, {FALLBACK_STD}). "
            f"Hay chay buoc EDA & Feature Engineering trong notebook de khoa thong so do dac chinh xac tren tap train.",
            RuntimeWarning,
            stacklevel=2,
        )
        return FALLBACK_MEAN, FALLBACK_STD, "fallback:legacy"

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    mean = config["normalization"]["mean"]
    std = config["normalization"]["std"]
    return tuple(float(m) for m in mean), tuple(float(s) for s in std), f"artifact:{path}"


FASHION_MNIST_MEAN, FASHION_MNIST_STD, NORMALIZATION_SOURCE = _resolve_normalization()

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


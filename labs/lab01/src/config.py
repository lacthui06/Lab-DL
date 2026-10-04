"""Configuration module for Lab 01: FashionMNIST Classification.
Centralizes paths, hyperparameters, and environment settings.
"""

from dataclasses import dataclass
from pathlib import Path
import torch


# Project Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"

# Ensure directories exist
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Constants
RANDOM_SEED = 42
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_CLASSES = 10

# FashionMNIST Channel Statistics
FASHION_MNIST_MEAN = (0.2860,)
FASHION_MNIST_STD = (0.3205,)

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


@dataclass
class ExperimentConfig:
    """Hyperparameters and configuration for an experiment run."""
    name: str
    model_type: str  # 'baseline_mlp', 'tuned_mlp', 'cnn'
    batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    use_scheduler: bool = True
    use_augmentation: bool = False
    val_split: float = 0.1
    patience: int = 5
    checkpoint_name: str = "best_model.pt"


# Pre-configured configurations for the 3 experimental runs
RUN1_CONFIG = ExperimentConfig(
    name="Run1_Baseline_MLP",
    model_type="baseline_mlp",
    batch_size=64,
    num_epochs=5,
    learning_rate=1e-3,
    weight_decay=0.0,
    use_scheduler=False,
    use_augmentation=False,
    checkpoint_name="baseline_mlp_best.pt",
)

RUN2_CONFIG = ExperimentConfig(
    name="Run2_Tuned_MLP",
    model_type="tuned_mlp",
    batch_size=64,
    num_epochs=10,
    learning_rate=1e-3,
    weight_decay=1e-4,
    use_scheduler=True,
    use_augmentation=True,
    checkpoint_name="tuned_mlp_best.pt",
)

RUN3_CONFIG = ExperimentConfig(
    name="Run3_FashionCNN",
    model_type="cnn",
    batch_size=64,
    num_epochs=10,
    learning_rate=1e-3,
    weight_decay=1e-4,
    use_scheduler=True,
    use_augmentation=True,
    checkpoint_name="fashion_cnn_best.pt",
)

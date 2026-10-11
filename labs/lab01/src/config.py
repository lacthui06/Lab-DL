"""Configuration module for Lab 01: FashionMNIST Classification.
Centralizes paths, hyperparameters, and environment settings.
"""

from dataclasses import dataclass
from pathlib import Path
import torch


# Project Base Directories
import os
if os.path.exists("/kaggle/working"):
    BASE_DIR = Path("/kaggle/working")
else:
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
    model_type: str  # 'baseline_mlp' or 'tuned_mlp'
    batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    use_scheduler: bool = True
    use_augmentation: bool = False
    val_split: float = 0.1
    patience: int = 5
    checkpoint_name: str = "best_model.pt"


# Pre-configured configurations for the 2 experimental runs
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

# ---------------------------------------------------------------------------
# Optimizer Comparison & K-Fold Cross-Validation (Phase 2 experiment)
# ---------------------------------------------------------------------------
# Full design rationale: walkthrough_lab_01.md, Section 1.6.
# All three optimizer arms share an identical learning rate and an identical
# epoch budget so that the optimizer itself is the only manipulated variable.
# The learning rate is held constant (no scheduler) and weight decay is
# disabled, isolating the optimizer update rule from other regularization.

CV_MODEL_TYPE = "tuned_mlp"
CV_EPOCHS = 15
CV_LEARNING_RATE = 1e-3
CV_SGD_MOMENTUM = 0.9
CV_BATCH_SIZE = 64
CV_USE_AUGMENTATION = True
CV_WEIGHT_DECAY = 0.0

KFOLD_SPLITS = 5
KFOLD_SEED = RANDOM_SEED


@dataclass
class OptimizerExperimentConfig:
    """Configuration for a single optimizer arm of the comparison experiment."""
    name: str
    optimizer_type: str  # 'sgd', 'rmsprop', or 'adam'
    learning_rate: float = CV_LEARNING_RATE
    momentum: float = 0.0


OPTIMIZER_CONFIGS = [
    OptimizerExperimentConfig(
        name="SGD_Momentum",
        optimizer_type="sgd",
        learning_rate=CV_LEARNING_RATE,
        momentum=CV_SGD_MOMENTUM,
    ),
    OptimizerExperimentConfig(
        name="RMSprop",
        optimizer_type="rmsprop",
        learning_rate=CV_LEARNING_RATE,
    ),
    OptimizerExperimentConfig(
        name="Adam",
        optimizer_type="adam",
        learning_rate=CV_LEARNING_RATE,
    ),
]

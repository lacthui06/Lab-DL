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
# Phase 1 uses a single optimizer (SGD with momentum) for both runs so that the
# architecture is the only manipulated variable. The learning rate is raised to
# 0.05 because SGD requires a much larger step size than adaptive optimizers to
# reach a comparable accuracy within the fixed epoch budget.
PHASE1_OPTIMIZER_TYPE = "sgd"
PHASE1_LEARNING_RATE = 0.05
PHASE1_SGD_MOMENTUM = 0.9

RUN1_CONFIG = ExperimentConfig(
    name="Run1_Baseline_MLP",
    model_type="baseline_mlp",
    batch_size=64,
    num_epochs=5,
    learning_rate=PHASE1_LEARNING_RATE,
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
    learning_rate=PHASE1_LEARNING_RATE,
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

# Scenario 2b raises only the SGD arm's learning rate. A non-adaptive method
# needs a larger step size than an adaptive one to use the same epoch budget,
# so this value is the single factor that distinguishes 2b from 2a.
CV_FAIR_SGD_LEARNING_RATE = 0.05

KFOLD_SPLITS = 5
KFOLD_SEED = RANDOM_SEED


@dataclass
class OptimizerExperimentConfig:
    """Configuration for a single optimizer arm of the comparison experiment."""
    name: str
    optimizer_type: str  # 'sgd', 'rmsprop', or 'adam'
    learning_rate: float = CV_LEARNING_RATE
    momentum: float = 0.0


def _build_scenario_arms(sgd_learning_rate: float) -> list:
    """Builds the three optimizer arms for one learning-rate scenario.

    Only the SGD arm's learning rate varies between scenarios; RMSprop and Adam
    keep the shared baseline rate. This keeps the 2a-versus-2b contrast
    attributable to the SGD step size alone.
    """
    return [
        OptimizerExperimentConfig(
            name="SGD_Momentum",
            optimizer_type="sgd",
            learning_rate=sgd_learning_rate,
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


@dataclass
class OptimizerScenario:
    """A named learning-rate scenario holding its own list of optimizer arms.

    `artifact_suffix` is appended to every output filename for this scenario so
    that scenario 2a and scenario 2b write distinct artifacts.
    """
    name: str
    artifact_suffix: str
    arms: list


OPTIMIZER_SCENARIOS = [
    OptimizerScenario(
        name="Symmetric",
        artifact_suffix="",
        arms=_build_scenario_arms(CV_LEARNING_RATE),
    ),
    OptimizerScenario(
        name="FairBudget",
        artifact_suffix="_fairbudget",
        arms=_build_scenario_arms(CV_FAIR_SGD_LEARNING_RATE),
    ),
]

# Backward-compatible alias: the symmetric-budget (2a) arm list.
OPTIMIZER_CONFIGS = OPTIMIZER_SCENARIOS[0].arms

# ---------------------------------------------------------------------------
# Scenario 2c - Learning-Rate Sweep (spec Section 1.6.10)
# ---------------------------------------------------------------------------
# Each optimizer is swept over five learning rates under an identical protocol
# (same frozen TunedMLP, same 5-fold stratified split, no scheduler, no weight
# decay, same seeding). Only the update rule and the learning rate vary across
# arms; every other optimizer hyperparameter is held at its scenarios 2a/2b
# value, so a difference between curves is attributable to the update rule and
# its step size rather than to a second tuned knob.
#
# Momentum/beta policy, fixed before the run (spec Section 1.6.10):
#   - SGD uses momentum 0.9, unchanged from 2a/2b.
#   - RMSprop keeps PyTorch's default momentum 0.0. torch.optim.RMSprop does
#     expose a momentum argument, but enabling it would turn the arm into
#     "RMSprop with momentum", a different update rule from textbook RMSprop,
#     and would introduce a second varying factor.
#   - Adam keeps PyTorch's default betas (0.9, 0.999). torch.optim.Adam has no
#     momentum argument at all; its momentum-like term is the first-moment decay
#     beta1, whose default 0.9 is already active.
#
# The grids are offset because the methods have different natural step-size
# scales: SGD with momentum needs the largest step, while the adaptive methods
# rescale each parameter's step by its recent gradient magnitude. The highest
# learning rate in each grid is included to probe the divergence boundary.
#
# The epoch budget is held equal to CV_EPOCHS (15) rather than shortened. A
# shorter budget would confound the sweep: SGD converges slowest, so cutting the
# budget depresses the SGD curve more than the adaptive curves and could make
# SGD look worse for a reason unrelated to its update rule. Matching the budget
# also makes the 1e-3 column reproduce scenario 2a exactly (same folds, same
# seed), which turns that column into a determinism check.
LR_SWEEP_EPOCHS = CV_EPOCHS

LR_SWEEP_GRID = {
    "SGD_Momentum": {
        "optimizer_type": "sgd",
        "momentum": CV_SGD_MOMENTUM,
        "learning_rates": [1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
    },
    "RMSprop": {
        "optimizer_type": "rmsprop",
        "momentum": 0.0,
        "learning_rates": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    },
    "Adam": {
        "optimizer_type": "adam",
        "momentum": 0.0,
        "learning_rates": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    },
}


@dataclass
class LRSweepCell:
    """A single (optimizer, learning-rate) cell of the Scenario 2c sweep."""
    name: str
    optimizer_type: str
    learning_rate: float
    momentum: float = 0.0


def build_lr_sweep_cells() -> list:
    """Flattens LR_SWEEP_GRID into a list of LRSweepCell.

    Produces one cell per (optimizer, learning rate) pair, optimizer-major, so
    the notebook prints one optimizer's whole curve before moving to the next.
    The default grid yields 3 optimizers x 5 learning rates = 15 cells.
    """
    cells = []
    for name, spec in LR_SWEEP_GRID.items():
        for lr in spec["learning_rates"]:
            cells.append(
                LRSweepCell(
                    name=name,
                    optimizer_type=spec["optimizer_type"],
                    learning_rate=lr,
                    momentum=spec["momentum"],
                )
            )
    return cells

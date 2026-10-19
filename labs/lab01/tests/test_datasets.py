"""Dataset and Data Loading tests."""

import sys
from pathlib import Path
import pytest
import torch

LAB01_DIR = Path(__file__).resolve().parent.parent
if str(LAB01_DIR) not in sys.path:
    sys.path.insert(0, str(LAB01_DIR))

from src.config import DATA_DIR, EDA_DIR, RANDOM_SEED
from src.datasets.classification import get_dataloaders
from src.utils.helpers import set_seed


@pytest.fixture(scope="module")
def sample_loaders():
    set_seed(RANDOM_SEED)
    return get_dataloaders(
        data_dir=DATA_DIR,
        split_dir=EDA_DIR,
        batch_size=8,
        val_split=0.1,
        num_workers=0,
    )


def test_data_integrity_and_shape(sample_loaders):
    train_loader, val_loader, test_loader, names = sample_loaders
    assert len(names) == 10, "FashionMNIST must have exactly 10 classes"
    images, targets = next(iter(train_loader))
    assert images.shape == (8, 1, 28, 28), f"Expected shape (8, 1, 28, 28), got {images.shape}"
    assert targets.shape == (8,), f"Expected shape (8,), got {targets.shape}"
    assert targets.min() >= 0 and targets.max() < 10, "Target labels must be in range [0, 9]"


def test_tensor_types_and_normalization(sample_loaders):
    train_loader, _, _, _ = sample_loaders
    images, targets = next(iter(train_loader))
    assert isinstance(images, torch.Tensor), "Images must be torch.Tensor"
    assert images.dtype == torch.float32, "Image tensors must be float32"
    assert isinstance(targets, torch.Tensor), "Targets must be torch.Tensor"
    assert targets.dtype == torch.int64, "Targets must be int64"
    assert abs(images.mean().item()) < 1.0, "Normalized batch mean should be reasonably centered"

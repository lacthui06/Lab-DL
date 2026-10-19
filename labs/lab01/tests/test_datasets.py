"""Dataset, Stratified Split, and EDA tests."""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np

LAB01_DIR = Path(__file__).resolve().parent.parent
if str(LAB01_DIR) not in sys.path:
    sys.path.insert(0, str(LAB01_DIR))

from src.utils.helpers import DATA_DIR, EDA_DIR, RANDOM_SEED, CLASS_NAMES, FASHION_MNIST_MEAN, FASHION_MNIST_STD, FALLBACK_STD, set_seed
from src.datasets.classification import get_dataloaders, get_transforms
from src.datasets.base import make_stratified_split, save_split, load_split
from src.datasets.eda import (
    compute_normalization_stats,
    class_distribution,
    class_similarity_matrix,
    most_similar_pairs,
    spatial_features,
    FOREGROUND_THRESHOLD,
    get_structural_overview,
    check_data_integrity,
    verify_pipeline_handoff,
)
from src.utils.plots import unnormalize


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


def test_1_data_integrity_and_shape(sample_loaders):
    train_loader, val_loader, test_loader, names = sample_loaders
    assert len(names) == 10, "FashionMNIST must have exactly 10 classes"
    images, targets = next(iter(train_loader))
    assert images.shape == (8, 1, 28, 28), f"Expected shape (8, 1, 28, 28), got {images.shape}"
    assert targets.shape == (8,), f"Expected shape (8,), got {targets.shape}"
    assert targets.min() >= 0 and targets.max() < 10, "Target labels must be in range [0, 9]"


def test_2_lazy_loading_and_tensor_types(sample_loaders):
    train_loader, _, _, _ = sample_loaders
    images, targets = next(iter(train_loader))
    assert isinstance(images, torch.Tensor), "Images must be torch.Tensor"
    assert images.dtype == torch.float32, "Image tensors must be float32"
    assert isinstance(targets, torch.Tensor), "Targets must be torch.Tensor"
    assert targets.dtype == torch.int64, "Targets must be int64"
    assert abs(images.mean().item()) < 1.0, "Normalized batch mean should be reasonably centered"
    assert 0.5 < images.std().item() < 2.0, "Normalized batch std should be reasonable"


def test_16_stratified_split_preserves_class_ratio():
    targets = np.array([0] * 100 + [1] * 200 + [2] * 300)
    train_idx, val_idx = make_stratified_split(targets, val_split=0.1, seed=42)
    assert len(train_idx) == 540
    assert len(val_idx) == 60


def test_17_stratified_split_is_disjoint_and_deterministic():
    targets = np.array([0, 1, 2] * 20)
    t1, v1 = make_stratified_split(targets, val_split=0.1, seed=42)
    t2, v2 = make_stratified_split(targets, val_split=0.1, seed=42)
    assert set(t1).isdisjoint(set(v1)), "Train and Val must not overlap"
    assert np.array_equal(t1, t2) and np.array_equal(v1, v2), "Split must be deterministic"


def test_18_split_artifact_round_trips(tmp_path):
    targets = np.array([0, 1, 2] * 10)
    t, v = make_stratified_split(targets, val_split=0.1, seed=42)
    save_split(tmp_path, t, v, targets, seed=42)
    loaded_t, loaded_v = load_split(tmp_path, expected_pool_size=len(targets))
    assert np.array_equal(t, loaded_t)
    assert np.array_equal(v, loaded_v)


def test_19_missing_split_artifact_raises():
    with pytest.raises(FileNotFoundError):
        load_split(Path("/nonexistent/directory"))


def test_20_split_artifact_rejects_wrong_pool_size(tmp_path):
    targets = np.array([0, 1, 2] * 10)
    t, v = make_stratified_split(targets, val_split=0.1, seed=42)
    save_split(tmp_path, t, v, targets, seed=42)
    with pytest.raises(ValueError):
        load_split(tmp_path, expected_pool_size=999)


def test_23_unnormalize_inverts_transform():
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    original = torch.rand(28, 28, generator=generator)
    normalized = get_transforms(augment=False)(original.numpy())
    restored = unnormalize(normalized)
    assert torch.allclose(restored, original.unsqueeze(0), atol=1e-5)


def test_24_eda_helpers_on_synthetic_data():
    images = np.zeros((4, 28, 28), dtype=np.uint8)
    images[0:2, :14, :] = 255
    images[2:4, 14:, :] = 51
    labels = np.array([0, 0, 1, 1])

    stats = compute_normalization_stats(images)
    assert stats["num_images"] == 4
    assert stats["mean"] == pytest.approx(0.3, abs=1e-6)


def test_25_eda_foreground_threshold():
    assert 0.0 < FOREGROUND_THRESHOLD < 0.5


def test_27_core_eda_checks_contract():
    X_train = np.zeros((10, 28, 28), dtype=np.uint8)
    y_train = np.arange(10, dtype=np.uint8)
    X_test = np.ones((5, 28, 28), dtype=np.uint8) * 100
    y_test = np.arange(5, dtype=np.uint8)

    overview = get_structural_overview(X_train, y_train, X_test, y_test, [f"c{i}" for i in range(10)])
    assert len(overview) == 2
    integrity = check_data_integrity(X_train, y_train, X_test, y_test)
    assert len(integrity) == 6


def test_28_eda_pipeline_handoff_contract(sample_loaders):
    train_loader, val_loader, _, _ = sample_loaders
    assert verify_pipeline_handoff(train_loader, val_loader)

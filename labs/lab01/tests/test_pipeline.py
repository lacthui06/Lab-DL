"""Unit tests and TDD sanity checks for Lab 01 (FashionMNIST).
Adheres strictly to agent/skills/test-driven-development/SKILL.md:
1. Data Integrity and Shape Test
2. Lazy Loading and Tensor Types Test
3. Model Forward Shape Test
4. Sanity Overfit 1 Batch Test (MANDATORY)
5. Model Checkpoint Save and Load Equivalence Test
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

# Add lab01 parent dir to path
LAB01_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB01_DIR))

from src.config import DATA_DIR, CLASS_NAMES, RANDOM_SEED, DEVICE
from src.dataset import get_dataloaders
from src.model import build_model, count_parameters
from src.engine import sanity_overfit_check
from src.utils import set_seed, save_checkpoint, load_checkpoint


@pytest.fixture(scope="module")
def sample_loaders():
    """Loads a small sample DataLoader for testing."""
    set_seed(RANDOM_SEED)
    train_loader, val_loader, test_loader, names = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=8,
        val_split=0.1,
        augment=False,
        num_workers=0,
    )
    return train_loader, val_loader, test_loader, names


def test_1_data_integrity_and_shape(sample_loaders):
    """Test 1: Verify data shapes, label range, and class naming."""
    train_loader, val_loader, test_loader, names = sample_loaders
    assert len(names) == 10, "FashionMNIST must have exactly 10 classes"
    
    # Check one batch
    images, targets = next(iter(train_loader))
    assert images.shape == (8, 1, 28, 28), f"Expected shape (8, 1, 28, 28), got {images.shape}"
    assert targets.shape == (8,), f"Expected shape (8,), got {targets.shape}"
    assert targets.min() >= 0 and targets.max() < 10, "Target labels must be in range [0, 9]"


def test_2_lazy_loading_and_tensor_types(sample_loaders):
    """Test 2: Verify tensor types, precision, and normalization range."""
    train_loader, _, _, _ = sample_loaders
    images, targets = next(iter(train_loader))
    
    assert isinstance(images, torch.Tensor), "Images must be torch.Tensor"
    assert images.dtype == torch.float32, "Image tensors must be float32"
    assert isinstance(targets, torch.Tensor), "Targets must be torch.Tensor"
    assert targets.dtype == torch.int64, "Targets must be int64 (LongTensor for CrossEntropy)"
    
    # Normalized data should have mean close to 0 and std close to 1
    mean_val = images.mean().item()
    std_val = images.std().item()
    assert abs(mean_val) < 1.0, f"Normalized batch mean should be reasonably centered, got {mean_val}"
    assert 0.5 < std_val < 2.0, f"Normalized batch std should be reasonable, got {std_val}"


def test_3_model_forward_shapes():
    """Test 3: Verify all model architectures produce correct output logits."""
    dummy_input = torch.randn(4, 1, 28, 28)
    model_types = ["baseline_mlp", "tuned_mlp", "cnn"]

    for m_type in model_types:
        model = build_model(m_type, num_classes=10)
        output = model(dummy_input)
        assert output.shape == (4, 10), f"Model {m_type} output shape must be (4, 10), got {output.shape}"
        assert torch.isfinite(output).all(), f"Model {m_type} output contains NaN or Inf"
        
        params = count_parameters(model)
        assert params["total"] > 0, f"Model {m_type} has 0 parameters"
        assert params["trainable"] > 0, f"Model {m_type} has 0 trainable parameters"


def test_4_sanity_overfit_one_batch(sample_loaders):
    """Test 4 (MANDATORY TDD): Overfit on 1 mini-batch (8 samples) to verify learnability."""
    train_loader, _, _, _ = sample_loaders
    images, targets = next(iter(train_loader))

    # Test with BaselineMLP
    model = build_model("baseline_mlp")
    passed, final_loss, final_acc = sanity_overfit_check(
        model=model,
        sample_images=images,
        sample_targets=targets,
        device=DEVICE,
        num_epochs=40,
        lr=1e-2,
    )
    assert passed, (
        f"Sanity Overfit failed for BaselineMLP! Final loss: {final_loss:.4f}, Acc: {final_acc:.2f}%. "
        f"Mô hình không thể học được trên 8 mẫu, kiến trúc hoặc loss có lỗi!"
    )
    assert final_acc == 100.0, f"Expected 100% accuracy on 8 overfitted samples, got {final_acc}%"


def test_5_save_and_load_checkpoint(tmp_path):
    """Test 5: Verify model saving and loading preserves weights and inference outputs."""
    model_save = build_model("cnn")
    model_save.eval()

    dummy_input = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        out_before = model_save(dummy_input)

    chk_path = tmp_path / "test_cnn.pt"
    save_checkpoint(model_save, chk_path, metadata={"val_acc": 92.5})

    assert chk_path.exists(), "Checkpoint file was not created"

    model_load = build_model("cnn")
    meta = load_checkpoint(model_load, chk_path, device=torch.device("cpu"))
    model_load.eval()

    with torch.no_grad():
        out_after = model_load(dummy_input)

    assert torch.allclose(out_before, out_after, atol=1e-6), "Loaded model outputs do not match original model"
    assert meta.get("val_acc") == 92.5, "Metadata was not correctly preserved in checkpoint"


if __name__ == "__main__":
    print("Running tests via pytest...")
    pytest.main(["-v", __file__])

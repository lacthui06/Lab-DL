"""Training logic and Sanity overfit tests."""

import sys
from pathlib import Path
import pytest
import torch

LAB01_DIR = Path(__file__).resolve().parent.parent
if str(LAB01_DIR) not in sys.path:
    sys.path.insert(0, str(LAB01_DIR))

from src.config import DATA_DIR, EDA_DIR, DEVICE, RANDOM_SEED
from src.datasets.classification import get_dataloaders
from src.models import build_model
from src.training.trainer import Trainer, build_optimizer, sanity_overfit_check
from src.utils.helpers import set_seed, save_checkpoint, load_checkpoint


def test_sanity_overfit_one_batch():
    """Mandatory Sanity Check: Overfit on 1 mini-batch (8 samples) to verify learnability."""
    set_seed(RANDOM_SEED)
    train_loader, _, _, _ = get_dataloaders(DATA_DIR, EDA_DIR, batch_size=8, num_workers=0)
    images, targets = next(iter(train_loader))

    model = build_model("mlp", num_classes=10)
    passed, final_loss, final_acc = sanity_overfit_check(
        model=model,
        sample_images=images,
        sample_targets=targets,
        device=DEVICE,
        num_epochs=40,
        lr=1e-2,
    )
    assert passed, f"Sanity overfit check failed: loss={final_loss:.4f}, acc={final_acc:.2f}%"


def test_save_and_load_checkpoint(tmp_path):
    model = build_model("mlp", num_classes=10).to(DEVICE)
    ckpt_file = tmp_path / "test_model.pt"
    save_checkpoint(model, ckpt_file, metadata={"best_acc": 88.5})
    assert ckpt_file.exists()

    new_model = build_model("mlp", num_classes=10).to(DEVICE)
    meta = load_checkpoint(new_model, ckpt_file, device=DEVICE)
    assert meta["best_acc"] == 88.5

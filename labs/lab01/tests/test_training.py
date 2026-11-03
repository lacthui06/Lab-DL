"""Training, Optimization, Checkpoint, and Environment tests."""

import sys
from pathlib import Path
import pytest
import torch

LAB01_DIR = Path(__file__).resolve().parent.parent
if str(LAB01_DIR) not in sys.path:
    sys.path.insert(0, str(LAB01_DIR))

from src.utils.helpers import (
    DEVICE,
    RANDOM_SEED,
    collect_environment_snapshot,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)
from src.models import build_model
from src.training.trainer import build_optimizer, sanity_overfit_check


def test_4_sanity_overfit_one_batch():
    set_seed(RANDOM_SEED)
    images = torch.randn(8, 784)
    targets = torch.arange(8)

    model = build_model("mlp", num_classes=10)
    passed, final_loss, final_acc = sanity_overfit_check(
        model=model,
        sample_images=images,
        sample_targets=targets,
        device=DEVICE,
        num_epochs=40,
        lr=1e-2,
    )
    assert passed, f"Sanity Overfit failed! Final loss: {final_loss:.4f}, Acc: {final_acc:.2f}%"


def test_5_save_and_load_checkpoint(tmp_path):
    model_save = build_model("mlp")
    model_save.eval()

    dummy_input = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        out_before = model_save(dummy_input)

    chk_path = tmp_path / "test_mlp.pt"
    save_checkpoint(model_save, chk_path, metadata={"val_acc": 92.5})
    assert chk_path.exists()

    model_load = build_model("mlp")
    meta = load_checkpoint(model_load, chk_path, device=torch.device("cpu"))
    model_load.eval()

    with torch.no_grad():
        out_after = model_load(dummy_input)

    assert torch.allclose(out_before, out_after, atol=1e-6)
    assert meta.get("val_acc") == 92.5


def test_6_build_optimizer_factory():
    model = build_model("mlp")
    opt_sgd = build_optimizer("sgd", model, learning_rate=1e-3, momentum=0.9)
    assert isinstance(opt_sgd, torch.optim.SGD)

    opt_rms = build_optimizer("rmsprop", model, learning_rate=1e-3)
    assert isinstance(opt_rms, torch.optim.RMSprop)

    opt_adam = build_optimizer("adam", model, learning_rate=1e-3)
    assert isinstance(opt_adam, torch.optim.Adam)

    with pytest.raises(ValueError):
        build_optimizer("invalid_name", model)


def test_12_environment_snapshot(tmp_path):
    out_path = tmp_path / "environment_snapshot.json"
    snapshot = collect_environment_snapshot(out_path)
    assert out_path.exists()
    assert "python_version" in snapshot
    assert "torch" in snapshot["libraries"]

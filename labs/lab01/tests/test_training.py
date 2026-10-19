"""Training, Optimization, Metrics, and Environment tests."""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np

LAB01_DIR = Path(__file__).resolve().parent.parent
if str(LAB01_DIR) not in sys.path:
    sys.path.insert(0, str(LAB01_DIR))

from src.config import DATA_DIR, EDA_DIR, DEVICE, RANDOM_SEED, CLASS_NAMES
from src.datasets.classification import get_dataloaders
from src.models import build_model
from src.training.trainer import Trainer, build_optimizer, sanity_overfit_check
from src.utils.helpers import set_seed, save_checkpoint, load_checkpoint, collect_environment_snapshot
from src.utils.metrics import compute_tpr_fpr, build_per_class_metrics
from src.experiment import predict_proba, run_optuna_optimizer_study, compare_optimizers_wallclock


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


def test_4_sanity_overfit_one_batch(sample_loaders):
    train_loader, _, _, _ = sample_loaders
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


def test_7_compute_tpr_fpr():
    cm = np.array([
        [8, 2, 0],
        [0, 9, 1],
        [2, 0, 8],
    ])
    tpr, fpr = compute_tpr_fpr([0, 0, 1], [0, 1, 1], num_classes=3)
    assert len(tpr) == 3 and len(fpr) == 3


def test_11_build_per_class_metrics():
    targets = [0, 0, 1, 1, 2, 2]
    preds = [0, 1, 1, 1, 2, 0]
    names = ["class_0", "class_1", "class_2"]
    df = build_per_class_metrics(targets, preds, names)
    assert len(df) == 3
    assert "F1-Score" in df.columns


def test_12_environment_snapshot(tmp_path):
    out_path = tmp_path / "environment_snapshot.json"
    snapshot = collect_environment_snapshot(out_path)
    assert out_path.exists()
    assert "python_version" in snapshot
    assert "torch" in snapshot["libraries"]


def test_26_predict_proba_contract(sample_loaders):
    _, val_loader, _, _ = sample_loaders
    model = build_model("mlp").to(DEVICE)
    y_probs, y_true = predict_proba(model, val_loader, DEVICE)
    assert len(y_probs) == len(y_true)
    assert y_probs.shape[1] == 10
    assert np.allclose(y_probs.sum(axis=1), 1.0, atol=1e-5)


def test_29_experiment_runner_signatures(sample_loaders):
    import pandas as pd
    train_loader, val_loader, _, _ = sample_loaders

    best_results, trials_df = run_optuna_optimizer_study(
        train_loader, val_loader, num_trials=2, epochs_per_trial=1, device=DEVICE
    )
    assert isinstance(best_results, dict)
    assert isinstance(trials_df, pd.DataFrame)

    toy_configs = {"Adam": {"params": {"lr": 0.001}}}
    summary_df, winner_model, winner_name, fig = compare_optimizers_wallclock(
        train_loader, val_loader, toy_configs, num_epochs=1, device=DEVICE
    )
    assert isinstance(summary_df, pd.DataFrame)
    assert isinstance(winner_model, torch.nn.Module)
    assert winner_name == "Adam"

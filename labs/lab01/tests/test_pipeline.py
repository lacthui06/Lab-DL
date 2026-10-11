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

from src.config import DATA_DIR, CLASS_NAMES, RANDOM_SEED, DEVICE, NUM_CLASSES, OPTIMIZER_CONFIGS, OPTIMIZER_SCENARIOS
from src.dataset import get_dataloaders, get_kfold_pool, build_fold_loaders
from src.model import build_model, count_parameters
from src.engine import sanity_overfit_check, build_optimizer, compute_tpr_fpr, build_per_class_metrics
from src.utils import set_seed, save_checkpoint, load_checkpoint, collect_environment_snapshot


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
    model_types = ["baseline_mlp", "tuned_mlp"]

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
    model_save = build_model("tuned_mlp")
    model_save.eval()

    dummy_input = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        out_before = model_save(dummy_input)

    chk_path = tmp_path / "test_tuned_mlp.pt"
    save_checkpoint(model_save, chk_path, metadata={"val_acc": 92.5})

    assert chk_path.exists(), "Checkpoint file was not created"

    model_load = build_model("tuned_mlp")
    meta = load_checkpoint(model_load, chk_path, device=torch.device("cpu"))
    model_load.eval()

    with torch.no_grad():
        out_after = model_load(dummy_input)

    assert torch.allclose(out_before, out_after, atol=1e-6), "Loaded model outputs do not match original model"
    assert meta.get("val_acc") == 92.5, "Metadata was not correctly preserved in checkpoint"


def test_6_build_optimizer_factory():
    """Test 6: Verify the optimizer factory and the two-scenario structure.

    Scenario 2a must be symmetric (every arm at the shared base rate). Scenario
    2b must differ from 2a in exactly one field: the SGD arm's learning rate.
    """
    model = build_model("tuned_mlp")

    opt_sgd = build_optimizer("sgd", model, learning_rate=1e-3, momentum=0.9)
    assert isinstance(opt_sgd, torch.optim.SGD), "Expected torch.optim.SGD"
    assert opt_sgd.param_groups[0]["lr"] == 1e-3, "SGD learning rate mismatch"
    assert opt_sgd.param_groups[0]["momentum"] == 0.9, "SGD momentum mismatch"

    opt_rms = build_optimizer("rmsprop", model, learning_rate=1e-3)
    assert isinstance(opt_rms, torch.optim.RMSprop), "Expected torch.optim.RMSprop"
    assert opt_rms.param_groups[0]["lr"] == 1e-3, "RMSprop learning rate mismatch"

    opt_adam = build_optimizer("adam", model, learning_rate=1e-3)
    assert isinstance(opt_adam, torch.optim.Adam), "Expected torch.optim.Adam"
    assert opt_adam.param_groups[0]["lr"] == 1e-3, "Adam learning rate mismatch"

    # Backward-compatible alias must be the symmetric-budget (2a) arm list
    lrs = {build_optimizer(c.optimizer_type, model, learning_rate=c.learning_rate).param_groups[0]["lr"]
           for c in OPTIMIZER_CONFIGS}
    assert lrs == {1e-3}, f"All symmetric optimizer arms must share lr=1e-3, got {lrs}"

    with pytest.raises(ValueError):
        build_optimizer("nonexistent_optimizer", model)

    # Two scenarios, each with the three classical arms
    assert len(OPTIMIZER_SCENARIOS) == 2, "Phase 2 must define exactly two scenarios"
    sym, fair = OPTIMIZER_SCENARIOS
    assert sym.artifact_suffix == "", "Scenario 2a must keep the unsuffixed artifacts"
    assert fair.artifact_suffix != "", "Scenario 2b must use a distinct artifact suffix"

    sym_lrs = {arm.name: arm.learning_rate for arm in sym.arms}
    fair_lrs = {arm.name: arm.learning_rate for arm in fair.arms}
    assert sym_lrs == {"SGD_Momentum": 1e-3, "RMSprop": 1e-3, "Adam": 1e-3}, f"2a arms wrong: {sym_lrs}"
    assert fair_lrs == {"SGD_Momentum": 0.05, "RMSprop": 1e-3, "Adam": 1e-3}, f"2b arms wrong: {fair_lrs}"

    # The only field that may differ between the scenarios is the SGD learning rate
    changed = {k for k in sym_lrs if sym_lrs[k] != fair_lrs[k]}
    assert changed == {"SGD_Momentum"}, f"2a vs 2b must differ only in SGD lr, got {changed}"


def test_7_compute_tpr_fpr_from_confusion_matrix():
    """Test 7: Verify TPR/FPR derivation against a hand-computed confusion matrix."""
    import numpy as np

    # 3-class toy matrix, rows = true, cols = predicted
    # total = 30, row sums = [10,10,10], col sums = [10,11,9]
    # class 0: TP=8, FN=2 -> TPR=0.80 ; FP=2, TN=18 -> FPR=2/20=0.10
    # class 1: TP=9, FN=1 -> TPR=0.90 ; FP=2, TN=18 -> FPR=2/20=0.10
    # class 2: TP=8, FN=2 -> TPR=0.80 ; FP=1, TN=19 -> FPR=1/20=0.05
    cm = np.array([
        [8, 2, 0],
        [0, 9, 1],
        [2, 0, 8],
    ])
    tpr, fpr = compute_tpr_fpr(cm)

    assert tpr.shape == (3,), f"Expected 3 TPR values, got {tpr.shape}"
    assert fpr.shape == (3,), f"Expected 3 FPR values, got {fpr.shape}"
    assert np.allclose(tpr, [0.80, 0.90, 0.80]), f"TPR mismatch: {tpr}"
    assert np.allclose(fpr, [0.10, 0.10, 0.05]), f"FPR mismatch: {fpr}"


def test_11_build_per_class_metrics_full_set():
    """Test 11: Verify the per-class metric builder returns the full metric set.

    Uses a 3-class toy set where every value is hand-computable, so the test
    pins the contract that each class carries precision, recall/TPR, F1, and FPR
    (FPR being the field scikit-learn's report does not provide).
    """
    # Confusion matrix (rows = true, cols = predicted):
    #   row0 [1,1,0]  row1 [0,2,0]  row2 [1,0,1]   total = 6
    # col sums = [2,3,1]
    # class 0: TP=1 FN=1 -> recall=0.5 ; FP=1 TN=3 -> FPR=1/4=0.25 ; P=1/2=0.5 ; F1=0.5
    # class 1: TP=2 FN=0 -> recall=1.0 ; FP=1 TN=3 -> FPR=1/4=0.25 ; P=2/3 ; F1=0.8
    # class 2: TP=1 FN=1 -> recall=0.5 ; FP=0 TN=4 -> FPR=0.0 ; P=1.0 ; F1=2/3
    targets = [0, 0, 1, 1, 2, 2]
    preds = [0, 1, 1, 1, 2, 0]
    names = ["class_0", "class_1", "class_2"]

    metrics = build_per_class_metrics(targets, preds, names)

    assert set(metrics.keys()) == set(names), f"Unexpected classes: {set(metrics.keys())}"
    for name in names:
        assert set(metrics[name].keys()) == {"precision", "recall_tpr", "f1", "fpr"}, \
            f"{name} is missing part of the metric set: {set(metrics[name].keys())}"

    assert metrics["class_0"] == {"precision": 0.5, "recall_tpr": 0.5, "f1": 0.5, "fpr": 0.25}
    assert metrics["class_1"]["recall_tpr"] == 1.0
    assert abs(metrics["class_1"]["precision"] - 2 / 3) < 1e-4, metrics["class_1"]
    assert abs(metrics["class_1"]["f1"] - 0.8) < 1e-4, metrics["class_1"]
    assert metrics["class_1"]["fpr"] == 0.25
    assert metrics["class_2"]["precision"] == 1.0
    assert metrics["class_2"]["recall_tpr"] == 0.5
    assert abs(metrics["class_2"]["f1"] - 2 / 3) < 1e-4, metrics["class_2"]
    assert metrics["class_2"]["fpr"] == 0.0


def test_12_environment_snapshot_records_python_version(tmp_path):
    """Test 12: Verify the environment snapshot records the Python interpreter.

    Section 1.8 requires every run to persist the interpreter version and the
    library versions. The test checks the returned keys and that the on-disk
    JSON round-trips, so the artifact cannot silently lose the Python version.
    """
    import json
    import sys

    out_path = tmp_path / "environment_snapshot.json"
    snapshot = collect_environment_snapshot(out_path)

    assert out_path.exists(), "environment_snapshot.json was not written"
    assert isinstance(snapshot["python_version"], str)
    assert snapshot["python_version"] == sys.version.split()[0]
    assert snapshot["python_version"].count(".") >= 1, snapshot["python_version"]
    assert snapshot["libraries"]["torch"], "torch version missing"
    assert snapshot["device"] in {"cpu", "cuda"}, snapshot["device"]

    reloaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert reloaded == snapshot, "snapshot did not round-trip through JSON"


def test_13_python_version_satisfies_pinned_dependencies():
    """Test 13: Verify the interpreter satisfies the pinned dependencies' requirement.

    The Python version is only a real constraint if something enforces it. This
    test derives the requirement from the installed packages' `Requires-Python`
    metadata (the same source the comment in `requirements.txt` is derived from)
    and asserts the running interpreter satisfies every one of them, so the
    recorded version cannot drift away from what the pins actually allow.
    """
    import importlib.metadata as md
    import sys

    from packaging.requirements import Requirement

    def _satisfies(version: str, specifier: str) -> bool:
        if not specifier:
            return True
        return Requirement(f"python{specifier}").specifier.contains(version)

    checked = 0
    for pkg in ["torch", "torchvision", "numpy", "scikit-learn", "matplotlib"]:
        try:
            meta = md.metadata(pkg)
        except md.PackageNotFoundError:
            continue
        requires_python = meta.get("Requires-Python")
        if not requires_python:
            continue
        checked += 1
        assert _satisfies(sys.version.split()[0], requires_python), (
            f"Running Python {sys.version.split()[0]} does not satisfy "
            f"{pkg} Requires-Python '{requires_python}'"
        )

    assert checked > 0, "No Requires-Python metadata found; constraint could not be verified"


def test_8_kfold_split_integrity():
    """Test 8: Verify Stratified K-Fold splits are disjoint, exhaustive, and class-balanced."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    set_seed(RANDOM_SEED)
    pool_train, pool_eval, targets, names = get_kfold_pool(
        data_dir=DATA_DIR, augment=False, seed=RANDOM_SEED
    )
    y = np.asarray(targets)
    assert len(y) == 60000, f"Expected 60,000 training-pool samples, got {len(y)}"
    assert len(names) == NUM_CLASSES, "Class-name count must match NUM_CLASSES"

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    seen = np.zeros(len(y), dtype=bool)
    for train_idx, val_idx in skf.split(np.zeros(len(y)), y):
        # Disjointness between the two halves of one fold
        assert len(np.intersect1d(train_idx, val_idx)) == 0, "Train and val indices overlap"
        # Each sample is validated exactly once across folds
        assert not seen[val_idx].any(), "A sample was validated more than once"
        seen[val_idx] = True
        # Stratification: validation fold preserves the balanced class ratio
        counts = np.bincount(y[val_idx], minlength=NUM_CLASSES)
        assert counts.min() > 0, "A class is missing from the validation fold"

    assert seen.all(), "Not every sample was covered exactly once across folds"

    # Fold loaders must build over the given index subsets with correct sizes
    train_idx, val_idx = next(iter(skf.split(np.zeros(len(y)), y)))
    fold_train_loader, fold_val_loader = build_fold_loaders(
        pool_train, pool_eval, train_idx, val_idx, batch_size=64, num_workers=0
    )
    assert len(fold_train_loader.dataset) == len(train_idx), "Fold train subset size mismatch"
    assert len(fold_val_loader.dataset) == len(val_idx), "Fold val subset size mismatch"
    images, labels = next(iter(fold_val_loader))
    assert images.shape[1:] == (1, 28, 28), f"Unexpected image shape {images.shape}"


def test_9_run_optimizer_scenario_smoke():
    """Test 9: Smoke-test the two-scenario runner on a tiny synthetic pool.

    Uses 200 synthetic 1x28x28 samples and 1 epoch so that all six arms
    (2 scenarios x 3 optimizers) execute end to end quickly. This verifies the
    scenario plumbing without any full-scale training.
    """
    from torch.utils.data import TensorDataset
    from src.engine import run_optimizer_scenario

    torch.manual_seed(RANDOM_SEED)
    n = 200
    images = torch.randn(n, 1, 28, 28)
    labels = torch.arange(n) % NUM_CLASSES
    pool = TensorDataset(images, labels)

    results = run_optimizer_scenario(
        scenario=OPTIMIZER_SCENARIOS[0],
        pool_train=pool,
        pool_eval=pool,
        targets=labels,
        model_type="tuned_mlp",
        num_epochs=1,
        batch_size=16,
        device=torch.device("cpu"),
        n_splits=2,
        seed=RANDOM_SEED,
        num_workers=0,
        checkpoint_dir=None,
        verbose=False,
    )

    assert set(results.keys()) == {"SGD_Momentum", "RMSprop", "Adam"}, f"Unexpected arms: {set(results.keys())}"
    for name, res in results.items():
        assert len(res["fold_accuracies"]) == 2, f"{name} did not produce 2 fold scores"
        assert 0.0 <= res["mean_acc"] <= 100.0, f"{name} mean accuracy out of range"
        assert res["confusion_matrix"].shape == (NUM_CLASSES, NUM_CLASSES), f"{name} confusion matrix shape wrong"


def test_10_cv_checkpoints_disjoint_across_scenarios(tmp_path):
    """Test 10: The two scenarios must not overwrite each other's CV checkpoints.

    Both scenarios reuse the same arm names (SGD / RMSprop / Adam). If the
    per-fold checkpoint filename ignored the scenario suffix, running 2a then 2b
    into the same directory would leave only six files instead of twelve. This
    test pins the naming rule documented in spec Section 1.6.7.
    """
    from torch.utils.data import TensorDataset
    from src.engine import run_optimizer_scenario

    torch.manual_seed(RANDOM_SEED)
    n = 200
    images = torch.randn(n, 1, 28, 28)
    labels = torch.arange(n) % NUM_CLASSES
    pool = TensorDataset(images, labels)

    for scenario in OPTIMIZER_SCENARIOS:
        run_optimizer_scenario(
            scenario=scenario,
            pool_train=pool,
            pool_eval=pool,
            targets=labels,
            model_type="tuned_mlp",
            num_epochs=1,
            batch_size=16,
            device=torch.device("cpu"),
            n_splits=2,
            seed=RANDOM_SEED,
            num_workers=0,
            checkpoint_dir=tmp_path,
            verbose=False,
        )

    written = sorted(p.name for p in tmp_path.glob("cv_*.pt"))
    # 2 scenarios x 3 arms x 2 folds = 12 disjoint checkpoints, no overwrites.
    assert len(written) == 12, f"Expected 12 disjoint CV checkpoints, got {len(written)}: {written}"

    sym_suffix = OPTIMIZER_SCENARIOS[0].artifact_suffix
    fair_suffix = OPTIMIZER_SCENARIOS[1].artifact_suffix
    assert any(f"cv_sgd{sym_suffix}_fold1.pt" == name for name in written)
    assert any(f"cv_sgd{fair_suffix}_fold1.pt" == name for name in written)


if __name__ == "__main__":
    print("Running tests via pytest...")
    pytest.main(["-v", __file__])

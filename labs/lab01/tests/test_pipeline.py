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
import matplotlib
matplotlib.use("Agg")
import pytest
import torch
import torch.nn as nn

# Add lab01 parent dir to path
LAB01_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB01_DIR))

from src.config import DATA_DIR, CLASS_NAMES, RANDOM_SEED, DEVICE, NUM_CLASSES
from src.dataset import get_dataloaders
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
    model_types = ["fashion_mlp", "mlp"]

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

    model = build_model("mlp")
    passed, final_loss, final_acc = sanity_overfit_check(
        model=model,
        sample_images=images,
        sample_targets=targets,
        device=DEVICE,
        num_epochs=40,
        lr=1e-2,
    )
    assert passed, (
        f"Sanity Overfit failed! Final loss: {final_loss:.4f}, Acc: {final_acc:.2f}%. "
        f"Mô hình không thể học được trên 8 mẫu, kiến trúc hoặc loss có lỗi!"
    )
    assert final_acc == 100.0, f"Expected 100% accuracy on 8 overfitted samples, got {final_acc}%"


def test_5_save_and_load_checkpoint(tmp_path):
    """Test 5: Verify model saving and loading preserves weights and inference outputs."""
    model_save = build_model("mlp")
    model_save.eval()

    dummy_input = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        out_before = model_save(dummy_input)

    chk_path = tmp_path / "test_mlp.pt"
    save_checkpoint(model_save, chk_path, metadata={"val_acc": 92.5})

    assert chk_path.exists(), "Checkpoint file was not created"

    model_load = build_model("mlp")
    meta = load_checkpoint(model_load, chk_path, device=torch.device("cpu"))
    model_load.eval()

    with torch.no_grad():
        out_after = model_load(dummy_input)

    assert torch.allclose(out_before, out_after, atol=1e-6), "Loaded model outputs do not match original model"
    assert meta.get("val_acc") == 92.5, "Metadata was not correctly preserved in checkpoint"


def test_6_build_optimizer_factory():
    """Test 6: Verify the optimizer factory builds correct optimizer types."""
    model = build_model("mlp")

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

    opt_adamw = build_optimizer("adamw", model, learning_rate=1e-3)
    assert isinstance(opt_adamw, torch.optim.AdamW), "Expected torch.optim.AdamW"

    with pytest.raises(ValueError):
        build_optimizer("nonexistent_optimizer", model)


def test_7_compute_tpr_fpr_from_confusion_matrix():
    """Test 7: Verify TPR/FPR derivation against a hand-computed confusion matrix."""
    import numpy as np

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
    """Test 11: Verify the per-class metric builder returns the full metric set."""
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
    """Test 12: Verify the environment snapshot records the Python interpreter."""
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
    """Test 13: Verify the interpreter satisfies the pinned dependencies' requirement."""
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


# ---------------------------------------------------------------------------
# Split integrity (split_v2_stratified)
# ---------------------------------------------------------------------------

def test_16_stratified_split_preserves_class_ratio():
    """Test 16: The split must be stratified, preserving class balance."""
    import numpy as np
    from src.split import make_stratified_split

    y = np.repeat(np.arange(NUM_CLASSES), 6_000)
    train_idx, val_idx = make_stratified_split(y, val_split=0.1, seed=RANDOM_SEED)

    assert len(train_idx) == 54_000, f"Expected 54,000 train samples, got {len(train_idx)}"
    assert len(val_idx) == 6_000, f"Expected 6,000 validation samples, got {len(val_idx)}"

    val_counts = np.bincount(y[val_idx], minlength=NUM_CLASSES)
    train_counts = np.bincount(y[train_idx], minlength=NUM_CLASSES)

    assert val_counts.tolist() == [600] * NUM_CLASSES, f"Val is not balanced: {val_counts.tolist()}"
    assert train_counts.tolist() == [5400] * NUM_CLASSES, f"Train is not balanced: {train_counts.tolist()}"


def test_17_stratified_split_is_disjoint_exhaustive_and_deterministic():
    """Test 17: Same seed reproduces the same partition, with no overlap and full coverage."""
    import numpy as np
    from src.split import make_stratified_split

    y = np.repeat(np.arange(NUM_CLASSES), 6_000)

    first_train, first_val = make_stratified_split(y, seed=RANDOM_SEED)
    second_train, second_val = make_stratified_split(y, seed=RANDOM_SEED)

    assert np.array_equal(first_train, second_train), "Same seed produced a different train split"
    assert np.array_equal(first_val, second_val), "Same seed produced a different val split"
    assert np.intersect1d(first_train, first_val).size == 0, "Train and val overlap"
    assert len(first_train) + len(first_val) == len(y), "Split does not cover the pool"

    seen = np.zeros(len(y), dtype=bool)
    seen[first_train] = True
    seen[first_val] = True
    assert seen.all(), "Some samples appear in neither partition"


def test_18_split_artifact_round_trips(tmp_path):
    """Test 18: The persisted split must reload to the exact same indices."""
    import numpy as np
    from src.split import make_stratified_split, save_split, load_split, load_split_metadata

    y = np.repeat(np.arange(NUM_CLASSES), 1_000)
    train_idx, val_idx = make_stratified_split(y, seed=RANDOM_SEED)

    save_split(tmp_path, train_idx, val_idx, targets=y, seed=RANDOM_SEED)
    reloaded_train, reloaded_val = load_split(tmp_path, expected_pool_size=len(y))

    assert np.array_equal(reloaded_train, train_idx), "Reloaded train indices differ"
    assert np.array_equal(reloaded_val, val_idx), "Reloaded val indices differ"

    meta = load_split_metadata(tmp_path)
    assert meta["stratified"] is True
    assert meta["seed"] == RANDOM_SEED
    assert meta["train_size"] == len(train_idx)
    assert meta["val_size"] == len(val_idx)
    assert meta["val_class_counts"] == [100] * NUM_CLASSES


def test_19_missing_split_artifact_raises_instead_of_resplitting(tmp_path):
    """Test 19: A missing split artifact must fail loudly, never fall back to a new split."""
    from src.split import load_split

    with pytest.raises(FileNotFoundError):
        load_split(tmp_path)


def test_20_split_artifact_rejects_wrong_pool_size(tmp_path):
    """Test 20: A split recorded for one pool must not be applied to another."""
    import numpy as np
    from src.split import make_stratified_split, save_split, load_split

    y = np.repeat(np.arange(NUM_CLASSES), 100)
    train_idx, val_idx = make_stratified_split(y, seed=RANDOM_SEED)
    save_split(tmp_path, train_idx, val_idx, targets=y, seed=RANDOM_SEED)

    with pytest.raises(ValueError):
        load_split(tmp_path, expected_pool_size=99_999)


# ---------------------------------------------------------------------------
# Normalization statistics
# ---------------------------------------------------------------------------

def test_21_normalization_yields_unit_variance():
    """Test 21: After Normalize, the train tensor must have mean ~0 and std ~1."""
    import numpy as np
    from src.config import (
        DATA_DIR,
        FASHION_MNIST_MEAN,
        FASHION_MNIST_STD,
        NORMALIZATION_SOURCE,
    )
    from src.dataset import get_transforms
    from src.eda import load_raw_arrays

    assert NORMALIZATION_SOURCE.startswith("artifact:"), (
        f"Normalization statistics came from '{NORMALIZATION_SOURCE}'. Run EDA & FE "
        f"to write preprocessing_config.json so src/config.py reads the measured artifact."
    )

    X_train, _, _, _ = load_raw_arrays(DATA_DIR)
    transform = get_transforms(augment=False)

    sample = X_train[:4_000]
    normalized = np.stack([transform(img.copy()).numpy().squeeze(0) for img in sample])

    mean = float(normalized.mean())
    std = float(normalized.std(ddof=0))

    assert abs(mean) < 0.02, f"Normalized mean should be near 0, got {mean:.4f}"
    assert abs(std - 1.0) < 0.02, (
        f"Normalized std should be near 1.0, got {std:.4f}. "
        f"Configured mean={FASHION_MNIST_MEAN}, std={FASHION_MNIST_STD}."
    )


def test_22_legacy_normalization_would_fail_unit_variance():
    """Test 22: The legacy statistics must be demonstrably wrong."""
    import numpy as np
    from src.config import DATA_DIR, FALLBACK_STD
    from src.eda import compute_normalization_stats, load_raw_arrays

    X_train, _, _, _ = load_raw_arrays(DATA_DIR)
    stats = compute_normalization_stats(X_train)

    normalized_std = stats["std"] / FALLBACK_STD[0]

    assert abs(normalized_std - 1.0) > 0.05, (
        f"Legacy std {FALLBACK_STD[0]} happened to normalize the real data correctly "
        f"(resulting std {normalized_std:.4f}); the test premise no longer holds."
    )
    assert stats["std"] > stats["per_image_std_mean"], (
        "The global std must exceed the mean of the per-image stds for this check to be meaningful"
    )
    assert abs(FALLBACK_STD[0] - stats["per_image_std_mean"]) < 5e-3, (
        f"Legacy std {FALLBACK_STD[0]} no longer matches the mean of the per-image "
        f"stds ({stats['per_image_std_mean']:.4f}); re-derive the documented defect."
    )


def test_23_unnormalize_inverts_the_configured_transform():
    """Test 23: `unnormalize` must use the configured statistics, not literals."""
    from src.config import FASHION_MNIST_MEAN, FASHION_MNIST_STD
    from src.dataset import get_transforms
    from src.utils import unnormalize

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    original = torch.rand(28, 28, generator=generator)

    normalized = get_transforms(augment=False)(original.numpy())
    assert normalized.shape == (1, 28, 28), f"Unexpected transform output {normalized.shape}"

    restored = unnormalize(normalized)
    assert torch.allclose(restored, original.unsqueeze(0), atol=1e-5), (
        f"unnormalize did not invert Normalize(mean={FASHION_MNIST_MEAN}, "
        f"std={FASHION_MNIST_STD}); max diff="
        f"{(restored - original.unsqueeze(0)).abs().max().item():.2e}"
    )

    restored_2d = unnormalize(normalized.squeeze(0))
    assert restored_2d.shape == (28, 28), f"Unexpected 2D shape {restored_2d.shape}"


# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------

def test_24_eda_helpers_on_synthetic_data():
    """Test 24: The EDA computations are correct on inputs with known answers."""
    import numpy as np
    from src.eda import (
        compute_normalization_stats,
        class_distribution,
        class_similarity_matrix,
        most_similar_pairs,
        spatial_features,
    )

    images = np.zeros((4, 28, 28), dtype=np.uint8)
    images[0:2, :14, :] = 255
    images[2:4, 14:, :] = 51
    labels = np.array([0, 0, 1, 1])

    stats = compute_normalization_stats(images)
    assert stats["num_images"] == 4
    assert stats["num_pixels"] == 4 * 28 * 28
    assert stats["mean"] == pytest.approx(0.3, abs=1e-6)

    distribution = class_distribution(labels[:2], labels[2:], ["a", "b"])
    assert set(distribution["split"]) == {"train", "val"}
    assert distribution["count"].tolist() == [2, 0, 0, 2]

    spatial = spatial_features(images, labels, ["a", "b"], threshold=0.1)
    assert spatial.loc[0, "foreground_ratio"] == pytest.approx(0.5, abs=1e-6)
    assert spatial.loc[0, "bbox_height"] == 14
    assert spatial.loc[0, "bbox_width"] == 28
    assert spatial.loc[0, "bbox_aspect_ratio"] == pytest.approx(2.0, abs=1e-6)

    mean_images = np.stack([
        images[0:2].astype(np.float32).mean(axis=0) / 255.0,
        images[2:4].astype(np.float32).mean(axis=0) / 255.0,
    ])
    similarity = class_similarity_matrix(mean_images, ["a", "b"], metric="cosine")
    assert similarity.loc["a", "a"] == pytest.approx(1.0, abs=1e-6)
    assert similarity.loc["a", "b"] == pytest.approx(0.0, abs=1e-6)

    pairs = most_similar_pairs(similarity, top_k=5)
    assert len(pairs) == 1, "Two classes yield exactly one unordered pair"
    assert pairs.iloc[0]["class_a"] == "a" and pairs.iloc[0]["class_b"] == "b"


def test_25_eda_foreground_threshold_is_not_zero():
    """Test 25: Foreground must use an explicit threshold, not `pixel != 0`."""
    from src.eda import FOREGROUND_THRESHOLD

    assert 0.0 < FOREGROUND_THRESHOLD < 0.5, (
        f"FOREGROUND_THRESHOLD={FOREGROUND_THRESHOLD} is not a sensible cutoff"
    )


def test_26_predict_proba_contract(sample_loaders):
    """Test 26: predict_proba produces valid probability distribution."""
    import numpy as np
    from src.experiment import predict_proba
    _, val_loader, _, _ = sample_loaders

    model = build_model("mlp").to(DEVICE)
    y_probs, y_true = predict_proba(model, val_loader, DEVICE)

    assert len(y_probs) == len(y_true)
    assert y_probs.shape[1] == 10
    assert (y_probs >= 0.0).all() and (y_probs <= 1.0).all()
    # Row sums of probabilities must equal 1.0
    assert np.allclose(y_probs.sum(axis=1), 1.0, atol=1e-5)


def test_27_core_eda_checks_contract():
    """Test 27: get_structural_overview and check_data_integrity run correctly."""
    import numpy as np
    from src.eda import get_structural_overview, check_data_integrity

    X_train = np.zeros((10, 28, 28), dtype=np.uint8)
    y_train = np.arange(10, dtype=np.uint8)
    X_test = np.ones((5, 28, 28), dtype=np.uint8) * 100
    y_test = np.arange(5, dtype=np.uint8)

    overview = get_structural_overview(X_train, y_train, X_test, y_test, [f"c{i}" for i in range(10)])
    assert len(overview) == 2
    assert "Tập dữ liệu" in overview.columns

    integrity = check_data_integrity(X_train, y_train, X_test, y_test)
    assert len(integrity) == 6
    # In this mock, all zeros in X_train has internal duplicates, but NaN/range/labels pass
    items = dict(zip(integrity["Hạng mục kiểm tra"], integrity["Trạng thái"]))
    assert items["Giá trị thiếu (NaN / Null)"] == "PASSED"
    assert items["Dải giá trị điểm ảnh [0, 255]"] == "PASSED"
    assert items["Dải nhãn phân loại [0, 9]"] == "PASSED"
    assert items["Rò rỉ trùng lặp Train vs Test"] == "PASSED"


def test_28_eda_fe_pipeline_and_guard_contract(sample_loaders, tmp_path):
    """Test 28: run_eda_and_fe_pipeline and verify_pipeline_handoff contract."""
    from src.eda import verify_pipeline_handoff
    train_loader, val_loader, _, _ = sample_loaders

    # Verify existing EDA artifacts pass guard assert
    assert verify_pipeline_handoff(train_loader, val_loader)


def test_29_experiment_runner_signatures(sample_loaders):
    """Test 29: run_optuna_optimizer_study and compare_optimizers_wallclock return signatures."""
    import pandas as pd
    from src.experiment import run_optuna_optimizer_study, compare_optimizers_wallclock
    train_loader, val_loader, _, _ = sample_loaders

    # 1. Optuna study returns (best_results, trials_df)
    best_results, trials_df = run_optuna_optimizer_study(
        train_loader, val_loader, num_trials=2, epochs_per_trial=1, device=DEVICE
    )
    assert isinstance(best_results, dict)
    assert isinstance(trials_df, pd.DataFrame)
    assert "SGD" in best_results and "Adam" in best_results

    # 2. Wallclock compare returns (summary_df, winner_model, winner_name)
    toy_configs = {
        "Adam": {"params": {"lr": 0.001}},
        "SGD": {"params": {"lr": 0.01, "momentum": 0.9}},
    }
    summary_df, winner_model, winner_name, fig = compare_optimizers_wallclock(
        train_loader, val_loader, toy_configs, num_epochs=1, device=DEVICE
    )
    assert isinstance(summary_df, pd.DataFrame)
    assert isinstance(winner_model, torch.nn.Module)
    assert winner_name in ["Adam", "SGD"]


if __name__ == "__main__":
    print("Running tests via pytest...")
    pytest.main(["-v", __file__])

"""Visualization module for EDA, training curves, confusion matrix, and optimizer studies."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def draw_confusion_counts(ax, counts, class_names, title):
    """Annotate every cell with its count; color encodes recall per true class."""
    counts = np.asarray(counts)
    totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0)
    im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for (row, col), count in np.ndenumerate(counts):
        ax.text(col, row, str(int(count)), ha="center", va="center", fontsize=8,
                color="white" if normalized[row, col] > 0.5 else "black")
    ax.set_xticks(range(len(class_names)), class_names, rotation=90)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set(xlabel="Predicted class", ylabel="True class", title=title + "\nCell text: sample count")
    return im


def _save_fig(fig: plt.Figure, save_path: Optional[Path]) -> None:
    """Helper to save a figure safely to target path."""
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=200, bbox_inches="tight")


def plot_eda_figures(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    class_names: Sequence[str],
    save_dir: Optional[Path] = None,
) -> Dict[str, plt.Figure]:
    """Generates standardized visual inspection plots for all EDA image steps."""
    s_dir = Path(save_dir) if save_dir else None
    figs = {}

    train_subset = X_train[train_idx]
    train_scaled = train_subset.astype(np.float32) / 255.0

    # 1. Sample Images Grid (10 classes x 2 samples - Step 4)
    f_samples, axes = plt.subplots(2, 10, figsize=(15, 3.8))
    for c_idx, name in enumerate(class_names):
        c_indices = train_idx[y_train[train_idx] == c_idx][:2]
        for row in range(2):
            ax = axes[row, c_idx]
            if len(c_indices) > row:
                ax.imshow(X_train[c_indices[row]], cmap="gray")
            ax.axis("off")
            if row == 0:
                ax.set_title(name, fontsize=8.5, fontweight="bold")
    f_samples.suptitle("FashionMNIST: Ảnh Mẫu Đại Diện 10 Lớp Trang Phục", fontsize=12, fontweight="bold", y=1.02)
    f_samples.tight_layout()
    _save_fig(f_samples, s_dir / "sample_images.png" if s_dir else None)
    figs["samples"] = f_samples

    # 2. Class Mean Images (Step 4)
    f_means, axes = plt.subplots(1, 10, figsize=(15, 2.2))
    for c_idx, name in enumerate(class_names):
        mean_img = train_subset[y_train[train_idx] == c_idx].mean(axis=0)
        axes[c_idx].imshow(mean_img, cmap="inferno")
        axes[c_idx].axis("off")
        axes[c_idx].set_title(name, fontsize=8.5, fontweight="bold")
    f_means.suptitle("FashionMNIST: Ảnh Trung Bình Pixel Từng Lớp", fontsize=12, fontweight="bold", y=1.05)
    f_means.tight_layout()
    _save_fig(f_means, s_dir / "class_mean_images.png" if s_dir else None)
    figs["means"] = f_means

    # 3. Class Distribution Bar Chart (Step 3)
    f_dist, ax_dist = plt.subplots(figsize=(10, 4))
    t_cnt = np.bincount(y_train[train_idx], minlength=len(class_names))
    v_cnt = np.bincount(y_train[val_idx], minlength=len(class_names))
    x = np.arange(len(class_names))
    w = 0.38
    ax_dist.bar(x - w / 2, t_cnt, w, label=f"Tập Train ({len(train_idx):,})", color="#1f77b4")
    ax_dist.bar(x + w / 2, v_cnt, w, label=f"Tập Val ({len(val_idx):,})", color="#ff7f0e")
    ax_dist.set_ylabel("Số lượng mẫu")
    ax_dist.set_title("Phân Phối Số Lượng Mẫu 10 Lớp Trên Tập Train và Validation", fontsize=11, fontweight="bold")
    ax_dist.set_xticks(x)
    ax_dist.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    ax_dist.legend()
    ax_dist.grid(True, linestyle="--", alpha=0.4)
    f_dist.tight_layout()
    _save_fig(f_dist, s_dir / "class_distribution.png" if s_dir else None)
    figs["distribution"] = f_dist

    # 4. Pixel Intensity Histogram (Step 5)
    f_hist, ax_hist = plt.subplots(figsize=(8.5, 4))
    sample_flat = train_scaled[:10000].reshape(-1)
    global_mean = float(train_scaled.mean())
    global_median = float(np.median(train_scaled))
    ax_hist.hist(sample_flat, bins=50, color="#2b5c8f", edgecolor="black", alpha=0.75, density=True)
    ax_hist.axvline(global_mean, color="red", linestyle="--", linewidth=2, label=f"Mean = {global_mean:.4f}")
    ax_hist.axvline(global_median, color="green", linestyle=":", linewidth=2, label=f"Median = {global_median:.4f}")
    ax_hist.set_title("FashionMNIST: Phân Phối Cường Độ Điểm Ảnh Trên Tập Train", fontsize=11, fontweight="bold")
    ax_hist.set_xlabel("Cường độ điểm ảnh [0, 1] (Pixel Intensity)")
    ax_hist.set_ylabel("Mật độ tần suất (Density)")
    ax_hist.grid(True, linestyle="--", alpha=0.4)
    ax_hist.legend(fontsize=9)
    f_hist.tight_layout()
    _save_fig(f_hist, s_dir / "pixel_histogram.png" if s_dir else None)
    figs["histogram"] = f_hist

    # 5. Spatial Mean Pixel Heatmap (Step 6)
    f_mean_map, ax_mm = plt.subplots(figsize=(5.5, 4.8))
    mean_spatial = train_scaled.mean(axis=0)
    sns.heatmap(
        mean_spatial,
        cmap="viridis",
        cbar_kws={"label": "Cường độ pixel trung bình"},
        ax=ax_mm,
    )
    ax_mm.set_title("Bản Đồ Nhiệt Điểm Ảnh Trung Bình (28x28)", fontsize=11, fontweight="bold")
    ax_mm.set_xlabel("Pixel X")
    ax_mm.set_ylabel("Pixel Y")
    f_mean_map.tight_layout()
    _save_fig(f_mean_map, s_dir / "mean_pixel_heatmap.png" if s_dir else None)
    figs["mean_heatmap"] = f_mean_map

    # 6. Spatial Pixel Variance Heatmap (Step 6)
    f_var_map, ax_vm = plt.subplots(figsize=(5.5, 4.8))
    var_spatial = train_scaled.var(axis=0)
    sns.heatmap(
        var_spatial,
        cmap="magma",
        cbar_kws={"label": "Phương sai điểm ảnh (Pixel Variance)"},
        ax=ax_vm,
    )
    ax_vm.set_title("Bản Đồ Nhiệt Phương Sai Điểm Ảnh (28x28)", fontsize=11, fontweight="bold")
    ax_vm.set_xlabel("Pixel X")
    ax_vm.set_ylabel("Pixel Y")
    f_var_map.tight_layout()
    _save_fig(f_var_map, s_dir / "pixel_variance_heatmap.png" if s_dir else None)
    figs["variance_heatmap"] = f_var_map

    # 7. Inter-Class Cosine Similarity Heatmap (Step 8)
    f_sim, ax_sim = plt.subplots(figsize=(8.5, 7))
    mean_imgs = np.array([train_subset[y_train[train_idx] == c].mean(axis=0) for c in range(len(class_names))])
    flat_means = mean_imgs.reshape(len(mean_imgs), -1).astype(np.float64)
    norms = np.linalg.norm(flat_means, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sim_matrix = (flat_means / norms) @ (flat_means / norms).T
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        cbar_kws={"label": "Cosine Similarity"},
        ax=ax_sim,
    )
    ax_sim.set_title("Ma Trận Tương Đồng Cosine Giữa Ảnh Trung Bình 10 Lớp", fontsize=11, fontweight="bold")
    ax_sim.set_xlabel("Lớp Trang Phục")
    ax_sim.set_ylabel("Lớp Trang Phục")
    ax_sim.tick_params(axis="x", rotation=35)
    f_sim.tight_layout()
    _save_fig(f_sim, s_dir / "class_similarity_heatmap.png" if s_dir else None)
    figs["similarity"] = f_sim

    return figs


def plot_loss_and_acc_curves(
    history: Dict[str, List[float]],
    title: str = "Training & Validation Metrics",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Visualizes training and validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], "o-", color="#1f77b4", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s--", color="#ff7f0e", label="Val Loss", linewidth=2)
    ax1.set_title(f"{title} - Loss Curve", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], "o-", color="#2ca02c", label="Train Accuracy", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "s--", color="#d62728", label="Val Accuracy", linewidth=2)
    ax2.set_title(f"{title} - Accuracy Curve", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig

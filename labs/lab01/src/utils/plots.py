"""Visualization module for EDA, training curves, confusion matrix, predictions, and optimizer studies."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from .helpers import FASHION_MNIST_MEAN, FASHION_MNIST_STD


def unnormalize(
    images: torch.Tensor,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Reverses the normalization applied by get_transforms."""
    if mean is None or std is None:
        mean = FASHION_MNIST_MEAN if mean is None else mean
        std = FASHION_MNIST_STD if std is None else std

    mean_t = torch.as_tensor(mean, dtype=images.dtype, device=images.device)
    std_t = torch.as_tensor(std, dtype=images.dtype, device=images.device)

    added_channel = images.dim() == 2
    if added_channel:
        images = images.unsqueeze(0)

    if images.shape[-3] != mean_t.numel():
        raise ValueError(
            f"Expected {mean_t.numel()} channel(s) at dim -3, got {images.shape[-3]} "
            f"for shape {tuple(images.shape)}"
        )

    shape = [1] * (images.dim() - 3) + [mean_t.numel(), 1, 1]
    restored = (images * std_t.view(shape) + mean_t.view(shape)).clamp(0.0, 1.0)
    return restored.squeeze(0) if added_channel else restored


# ---------------------------------------------------------------------------
# EDA Visualizations
# ---------------------------------------------------------------------------
def plot_eda_figures(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    class_names: Sequence[str],
    save_dir: Optional[Path] = None,
) -> Dict[str, plt.Figure]:
    """Generates standardized visual inspection plots for EDA:
    1. sample_images.png: 10 classes x 2 sample images grid.
    2. class_mean_images.png: Mean intensity image per class.
    3. class_distribution.png: Train vs Val class count distribution.
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_root = save_dir.parent.parent if save_dir.name == "figures" and save_dir.parent.name == "eda" else None
    else:
        out_root = None

    figs = {}

    # 1. Sample Images Grid (10 classes x 2 samples)
    fig_samples, axes = plt.subplots(2, 10, figsize=(15, 3.8))
    for c_idx, name in enumerate(class_names):
        c_indices = train_idx[y_train[train_idx] == c_idx][:2]
        for row in range(2):
            ax = axes[row, c_idx]
            if len(c_indices) > row:
                ax.imshow(X_train[c_indices[row]], cmap="gray")
            ax.axis("off")
            if row == 0:
                ax.set_title(name, fontsize=8.5, fontweight="bold")
    fig_samples.suptitle("FashionMNIST: Ảnh Mẫu Đại Diện 10 Lớp Trang Phục", fontsize=12, fontweight="bold", y=1.02)
    fig_samples.tight_layout()
    if save_dir:
        fig_samples.savefig(str(save_dir / "sample_images.png"), dpi=200, bbox_inches="tight")
        if out_root and out_root.exists():
            fig_samples.savefig(str(out_root / "eda_sample_images.png"), dpi=200, bbox_inches="tight")
    figs["samples"] = fig_samples

    # 2. Class Mean Images
    fig_means, axes = plt.subplots(1, 10, figsize=(15, 2.2))
    for c_idx, name in enumerate(class_names):
        c_imgs = X_train[train_idx[y_train[train_idx] == c_idx]]
        mean_img = c_imgs.mean(axis=0)
        ax = axes[c_idx]
        ax.imshow(mean_img, cmap="inferno")
        ax.axis("off")
        ax.set_title(name, fontsize=8.5, fontweight="bold")
    fig_means.suptitle("FashionMNIST: Ảnh Trung Bình Pixel (Mean Images) Từng Lớp", fontsize=12, fontweight="bold", y=1.05)
    fig_means.tight_layout()
    if save_dir:
        fig_means.savefig(str(save_dir / "class_mean_images.png"), dpi=200, bbox_inches="tight")
        if out_root and out_root.exists():
            fig_means.savefig(str(out_root / "eda_class_mean_images.png"), dpi=200, bbox_inches="tight")
    figs["means"] = fig_means

    # 3. Class Distribution Bar Chart
    fig_dist, ax = plt.subplots(figsize=(10, 4))
    train_counts = np.bincount(y_train[train_idx], minlength=len(class_names))
    val_counts = np.bincount(y_train[val_idx], minlength=len(class_names))
    x = np.arange(len(class_names))
    width = 0.38
    ax.bar(x - width/2, train_counts, width, label=f"Tập Train ({len(train_idx):,})", color="#1f77b4")
    ax.bar(x + width/2, val_counts, width, label=f"Tập Val ({len(val_idx):,})", color="#ff7f0e")
    ax.set_ylabel("Số lượng mẫu")
    ax.set_title("Phân Phối Số Lượng Mẫu 10 Lớp Trên Tập Train và Validation", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig_dist.tight_layout()
    if save_dir:
        fig_dist.savefig(str(save_dir / "class_distribution.png"), dpi=200, bbox_inches="tight")
        if out_root and out_root.exists():
            fig_dist.savefig(str(out_root / "eda_class_distribution.png"), dpi=200, bbox_inches="tight")
    figs["distribution"] = fig_dist

    return figs


# ---------------------------------------------------------------------------
# Training & Evaluation Visualizations
# ---------------------------------------------------------------------------
def plot_loss_and_acc_curves(
    history: Dict[str, List[float]],
    title: str = "Training & Validation Metrics",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Visualizes training and validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(epochs, history["train_loss"], "o-", color="#1f77b4", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "s--", color="#ff7f0e", label="Val Loss", linewidth=2)
    axes[0].set_title(f"{title} - Loss Curve", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("CrossEntropy Loss", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=10)

    # Accuracy curve
    axes[1].plot(epochs, history["train_acc"], "o-", color="#2ca02c", label="Train Accuracy", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "s--", color="#d62728", label="Val Accuracy", linewidth=2)
    axes[1].set_title(f"{title} - Accuracy Curve", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Accuracy (%)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plots integer count heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        cbar_kws={"label": "Số lượng mẫu (Sample Count)"},
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("Ground Truth Class", fontsize=11)
    ax.tick_params(axis="x", rotation=40)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    return fig


def plot_prediction_grid(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: Sequence[str],
    device: torch.device,
    num_images: int = 16,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Displays a grid of test images with Predicted vs Actual labels."""
    model.eval()
    model.to(device)

    images_list, targets_list, preds_list, confs_list = [], [], [], []
    with torch.inference_mode():
        for images, targets in dataloader:
            images_dev = images.to(device)
            outputs = model(images_dev)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            for i in range(images.size(0)):
                images_list.append(images[i])
                targets_list.append(targets[i].item())
                preds_list.append(preds[i].item())
                confs_list.append(confs[i].item() * 100.0)
                if len(images_list) >= num_images:
                    break
            if len(images_list) >= num_images:
                break

    rows = int(np.ceil(np.sqrt(num_images)))
    cols = int(np.ceil(num_images / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.6))
    axes = np.array(axes).reshape(-1)

    for i in range(len(images_list)):
        ax = axes[i]
        img = unnormalize(images_list[i]).squeeze().cpu().numpy()
        p_name = class_names[preds_list[i]]
        t_name = class_names[targets_list[i]]
        correct = preds_list[i] == targets_list[i]
        color = "#2ca02c" if correct else "#d62728"
        status = "CORRECT" if correct else "WRONG"

        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(f"Pred: {p_name} ({confs_list[i]:.1f}%)\nTrue: {t_name} [{status}]", fontsize=9, color=color, fontweight="bold")

    for j in range(len(images_list), len(axes)):
        axes[j].axis("off")

    fig.suptitle("FashionMNIST: Dự đoán mẫu trên tập kiểm thử", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Optimizer & Experiment Visualizations
# ---------------------------------------------------------------------------
def plot_optimizer_comparison(
    run_histories: Dict[str, Dict[str, Any]],
    num_epochs: int,
    target_accuracy: float = 87.0,
    save_fig_path: Optional[Path] = None,
) -> plt.Figure:
    """Plots Validation Accuracy and Validation Loss progression across epochs for compared optimizers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"SGD": "#1f77b4", "RMSprop": "#ff7f0e", "Adam": "#2ca02c"}
    epochs_range = list(range(1, num_epochs + 1))

    # Subplot 1: Validation Accuracy vs Epochs
    for opt_name, hist in run_histories.items():
        c = colors.get(opt_name, "#333333")
        ax1.plot(
            epochs_range,
            hist["val_accs"],
            marker="o",
            linewidth=2,
            label=f"{opt_name} (Đỉnh: {hist['best_acc']:.2f}%)",
            color=c,
        )
    ax1.axhline(target_accuracy, color="red", linestyle=":", label=f"Mục tiêu {target_accuracy}%")
    ax1.set_title("Độ chính xác kiểm định (Validation Accuracy) theo Epoch", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    # Subplot 2: Validation Loss vs Epochs
    for opt_name, hist in run_histories.items():
        c = colors.get(opt_name, "#333333")
        ax2.plot(
            epochs_range,
            hist["val_losses"],
            marker="s",
            linewidth=2,
            label=f"{opt_name} (Đáy: {hist['min_loss']:.4f})",
            color=c,
        )
    ax2.set_title("Hàm mất mát kiểm định (Validation Loss) theo Epoch", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("CrossEntropy Loss", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    fig.suptitle("So sánh 3 Optimizer: Validation Accuracy & Loss Theo Epoch", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_fig_path:
        save_fig_path = Path(save_fig_path)
        save_fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_fig_path), dpi=200, bbox_inches="tight")
    return fig


def plot_roc_curves_fig(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: Sequence[str],
    save_fig_path: Optional[Path] = None,
) -> Tuple[Dict[str, float], plt.Figure]:
    """Computes and plots One-vs-Rest ROC curves and AUC scores for all classes."""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[class_names[i]] = float(auc(fpr[i], tpr[i]))

    # Compute macro-average ROC curve and AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = float(auc(all_fpr, mean_tpr))
    roc_auc["macro_avg"] = macro_auc

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, color in zip(range(n_classes), colors):
        c_name = class_names[i]
        ax.plot(
            fpr[i], tpr[i], color=color, lw=1.5,
            label=f"{c_name} (AUC = {roc_auc[c_name]:.3f})",
        )

    ax.plot(
        all_fpr, mean_tpr, color="black", linestyle="--", lw=2.5,
        label=f"Macro-average (AUC = {macro_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", lw=1.2, label="Đoán ngẫu nhiên (AUC = 0.500)")

    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
    ax.set_title("Đường cong One-vs-Rest (OvR) ROC & Điểm AUC - FashionMLP", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    if save_fig_path:
        save_fig_path = Path(save_fig_path)
        save_fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_fig_path), dpi=200, bbox_inches="tight")

    return roc_auc, fig


def plot_threshold_scan_fig(
    thresholds: np.ndarray,
    precisions: List[float],
    recalls: List[float],
    f1_scores: List[float],
    accuracies: List[float],
    best_t: float,
    best_f1: float,
    title: str = "Đồ thị quét ngưỡng phân loại (Decision Threshold Scan)",
    save_fig_path: Optional[Path] = None,
) -> plt.Figure:
    """Plots 4-curve trade-off for decision threshold scanning."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(thresholds, precisions, label="Precision (Độ chính xác)", color="#1f77b4", linewidth=2)
    ax.plot(thresholds, recalls, label="Recall (Độ nhạy)", color="#2ca02c", linewidth=2)
    ax.plot(thresholds, f1_scores, label="F1-Score (Trung bình điều hòa)", color="#ff7f0e", linewidth=2.5)
    ax.plot(thresholds, accuracies, label="Accuracy (Độ chính xác toàn cục)", color="#9467bd", linestyle="--", linewidth=1.8)

    ax.axvline(best_t, color="red", linestyle=":", linewidth=2, label=f"Ngưỡng F1 tối ưu = {best_t:.2f} (F1={best_f1:.2f})")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Ngưỡng phân loại (Threshold)", fontsize=10)
    ax.set_ylabel("Giá trị chỉ số", fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    if save_fig_path:
        save_fig_path = Path(save_fig_path)
        save_fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_fig_path), dpi=200, bbox_inches="tight")
    return fig

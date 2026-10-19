"""Visualization module for evaluation plots, curves, and prediction grids."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader


def unnormalize(
    images: torch.Tensor,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Reverses the normalization applied by get_transforms."""
    if mean is None or std is None:
        try:
            from ..config import FASHION_MNIST_MEAN, FASHION_MNIST_STD
            mean = FASHION_MNIST_MEAN if mean is None else mean
            std = FASHION_MNIST_STD if std is None else std
        except Exception:
            mean = (0.2860,) if mean is None else mean
            std = (0.3530,) if std is None else std

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
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
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
        xticklabels=class_names,
        yticklabels=class_names,
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
    class_names: List[str],
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

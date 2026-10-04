"""Utility functions for Lab 01: FashionMNIST.
Includes:
- Reproducibility (seed fixing)
- Model checkpoint saving and loading
- Loss and Accuracy curve visualizations
- Predicted vs. Actual image visualization grid
- Confusion matrix plotting
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns


def set_seed(seed: int = 42) -> None:
    """Fixes random seeds across Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_checkpoint(model: nn.Module, save_path: Path, metadata: Optional[dict] = None) -> None:
    """Saves model state dict and optional metadata."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, str(save_path))


def load_checkpoint(model: nn.Module, load_path: Path, device: torch.device) -> dict:
    """Loads weights from checkpoint into the given model."""
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {load_path}")
    checkpoint = torch.load(str(load_path), map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    return checkpoint.get("metadata", {})


def plot_loss_and_acc_curves(
    history: Dict[str, List[float]],
    title: str = "Training & Validation Metrics",
    save_path: Optional[Path] = None,
) -> None:
    """Visualizes training & validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss subplot
    axes[0].plot(epochs, history["train_loss"], "o-", color="#1f77b4", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "s--", color="#ff7f0e", label="Val Loss", linewidth=2)
    axes[0].set_title(f"{title} — Loss Curve", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("CrossEntropy Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=11)

    # Accuracy subplot
    axes[1].plot(epochs, history["train_acc"], "o-", color="#2ca02c", label="Train Accuracy", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "s--", color="#d62728", label="Val Accuracy", linewidth=2)
    axes[1].set_title(f"{title} — Accuracy Curve", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


def plot_prediction_grid(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: List[str],
    device: torch.device,
    num_images: int = 16,
    save_path: Optional[Path] = None,
) -> None:
    """Displays a grid of test images with Predicted vs. Actual labels.
    Correct predictions are labeled in green; errors are labeled in red.
    """
    model.eval()
    model.to(device)

    images_list = []
    targets_list = []
    preds_list = []
    probs_list = []

    with torch.inference_mode():
        for images, targets in dataloader:
            images_dev = images.to(device)
            outputs = model(images_dev)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, dim=1)

            for i in range(images.size(0)):
                images_list.append(images[i])
                targets_list.append(targets[i].item())
                preds_list.append(preds[i].item())
                probs_list.append(confidences[i].item() * 100.0)
                if len(images_list) >= num_images:
                    break
            if len(images_list) >= num_images:
                break

    rows = int(np.ceil(np.sqrt(num_images)))
    cols = int(np.ceil(num_images / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 4.0))
    axes = np.array(axes).reshape(-1)

    for i in range(len(images_list)):
        ax = axes[i]
        # Unnormalize for visualization
        # original transform: (x - 0.2860) / 0.3205 => x = img * 0.3205 + 0.2860
        img = images_list[i].squeeze().cpu().numpy()
        img = img * 0.3205 + 0.2860
        img = np.clip(img, 0.0, 1.0)

        pred_class = class_names[preds_list[i]]
        true_class = class_names[targets_list[i]]
        conf = probs_list[i]
        is_correct = preds_list[i] == targets_list[i]

        ax.imshow(img, cmap="gray")
        ax.axis("off")

        color = "#2ca02c" if is_correct else "#d62728"
        status = "CORRECT" if is_correct else "MISCLASSIFIED"
        title_text = f"Pred: {pred_class} ({conf:.1f}%)\nTrue: {true_class}\n[{status}]"
        ax.set_title(title_text, fontsize=9.5, color=color, fontweight="bold", pad=6)

    # Hide extra subplots
    for j in range(len(images_list), len(axes)):
        axes[j].axis("off")

    plt.suptitle("FashionMNIST: Predicted vs. Actual Test Samples", fontsize=15, fontweight="bold", y=0.98)
    plt.subplots_adjust(top=0.92, bottom=0.04, hspace=0.35, wspace=0.2)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
) -> None:
    """Plots a normalized heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Proportion"},
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("Ground Truth Class", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


def plot_comparison_curves(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: Optional[Path] = None,
) -> None:
    """Compares Validation Accuracy and Validation Loss across multiple runs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (run_name, hist) in enumerate(histories.items()):
        epochs = range(1, len(hist["val_acc"]) + 1)
        c = colors[idx % len(colors)]
        axes[0].plot(epochs, hist["val_acc"], "o-", label=run_name, color=c, linewidth=2)
        axes[1].plot(epochs, hist["val_loss"], "s--", label=run_name, color=c, linewidth=2)

    axes[0].set_title("Validation Accuracy Comparison Across Runs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Val Accuracy (%)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    axes[1].set_title("Validation Loss Comparison Across Runs", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Val Loss", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()

"""Five evidence plots, written from the actual run only."""
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def plot_lengths(lengths, cutoff, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(lengths, bins=40, kde=len(set(lengths)) > 1, ax=ax)
    ax.axvline(cutoff, color="red", linestyle="--", label=f"max_length={cutoff}")
    ax.set(xlabel="Tokens before truncation", ylabel="Training examples")
    ax.legend()
    finish(fig, path)

def plot_learning(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # Ưu tiên train-eval loss: cùng chế độ eval với validation, dropout đã tắt.
    train_key = "train_loss" if any("train_loss" in r for r in history) else "loss"
    for key, label in [(train_key, "Train loss (eval mode)" if train_key == "train_loss" else "Train optimization loss"),
                       ("eval_loss", "Validation loss")]:
        rows = [r for r in history if key in r and "epoch" in r
                and (key != "train_loss" or "train_accuracy" in r)]
        axes[0].plot([r["epoch"] for r in rows], [r[key] for r in rows], marker="o", markersize=3, label=label)
    for key, label in [("train_accuracy", "Train accuracy (eval mode)"), ("eval_accuracy", "Validation accuracy")]:
        rows = [r for r in history if key in r]
        axes[1].plot([r["epoch"] for r in rows], [r[key] for r in rows], marker="o", markersize=3, label=label)
    for ax in axes:
        ax.legend(); ax.grid(alpha=.2); ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].set(ylabel="Accuracy", ylim=(0, 1))
    finish(fig, path)

def plot_confusion(labels, preds, path):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    norm = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float),
                     where=cm.sum(axis=1, keepdims=True) != 0)
    annot = np.array([[f"{cm[i,j]}\n{norm[i,j]:.1%}" for j in range(2)] for i in range(2)])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(norm, annot=annot, fmt="", vmin=0, vmax=1, cmap="Blues", ax=ax,
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    ax.set(xlabel="Predicted", ylabel="True", title="Test confusion matrix (row normalized)")
    finish(fig, path)

def plot_metrics(metrics, path):
    keys = ["accuracy", "binary_precision", "binary_recall", "binary_f1"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(["Accuracy", "Precision", "Recall", "F1"], [metrics[k] for k in keys])
    ax.bar_label(bars, fmt="%.3f"); ax.set(ylim=(0, 1.1))
    finish(fig, path)

def plot_samples(rows, path):
    fig, ax = plt.subplots(figsize=(14, max(3, len(rows) * .85)))
    ax.axis("off")
    body = [[textwrap.fill(r["text"], 85), str(r["true_label"]), str(r["predicted_label"]),
             f"{r['confidence']:.3f}"] for r in rows]
    table = ax.table(cellText=body, colLabels=["Review", "True", "Predicted", "Confidence"],
                     colWidths=[.73, .08, .09, .1], loc="center", cellLoc="left")
    table.auto_set_font_size(False); table.set_fontsize(8)
    for i, row in enumerate(rows, 1):
        for j in range(4):
            table[i, j].set_facecolor("#d9ead3" if row["true_label"] == row["predicted_label"] else "#f4cccc")
            table[i, j].set_height(.9 / (len(rows) + 1))
    finish(fig, path)


def plot_step_learning(history, path, best_step=None):
    """Ghi tách loss tối ưu, validation, learning rate và gradient theo step."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, keys, title in [
        (axes[0, 0], [("loss", "Optimization loss (window mean)"), ("eval_loss", "Validation loss")], "Loss: different measurement contexts"),
        (axes[0, 1], [("eval_accuracy", "Validation accuracy"), ("eval_macro_f1", "Validation macro F1")], "Validation trajectory"),
        (axes[1, 0], [("learning_rate", "Learning rate")], "Scheduler"),
        (axes[1, 1], [("grad_norm", "Gradient norm before clipping")], "Optimization stability"),
    ]:
        for key, label in keys:
            rows = [r for r in history if key in r and "step" in r and r[key] is not None and np.isfinite(r[key])]
            if rows: ax.plot([r["step"] for r in rows], [r[key] for r in rows], label=label)
        ax.set(title=title, xlabel="Optimizer step"); ax.grid(alpha=.2)
        if best_step is not None:
            ax.axvline(best_step, ls="--", color="black", alpha=.6, label="Selected checkpoint")
        if ax.lines: ax.legend(fontsize=8)
    finish(fig, path)

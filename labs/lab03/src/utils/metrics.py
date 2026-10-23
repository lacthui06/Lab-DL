"""Evaluation metrics and visualization utilities."""
from typing import Dict, Any, List
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import evaluate


def get_compute_metrics_fn():
    """Build compute_metrics callback for Hugging Face Trainer."""
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
        precision = precision_metric.compute(predictions=predictions, references=labels, average="macro")["precision"]
        recall = recall_metric.compute(predictions=predictions, references=labels, average="macro")["recall"]

        return {
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "macro_precision": float(precision),
            "macro_recall": float(recall)
        }

    return compute_metrics


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    output_path: str,
    title: str = "Confusion Matrix"
) -> None:
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_environment_snapshot(output_path: str) -> None:
    """Save hardware, library versions and runtime environment details."""
    import sys
    import platform
    import torch
    import transformers
    import datasets
    import evaluate
    import accelerate

    snapshot = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "libraries": {
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "evaluate": evaluate.__version__,
            "accelerate": accelerate.__version__,
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

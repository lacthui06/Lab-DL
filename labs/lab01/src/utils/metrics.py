"""Metrics calculation module for classification evaluation."""

from typing import Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


def compute_tpr_fpr(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes True Positive Rate and False Positive Rate per class from confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    tpr = np.zeros(num_classes, dtype=np.float64)
    fpr = np.zeros(num_classes, dtype=np.float64)

    total_samples = cm.sum()
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total_samples - (tp + fn + fp)

        tpr[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr[c] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return tpr, fpr


def build_per_class_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Builds a comprehensive per-class metrics DataFrame."""
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    tpr, fpr = compute_tpr_fpr(y_true, y_pred, num_classes)

    precisions = precision_score(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)

    rows = []
    for c in range(num_classes):
        rows.append({
            "Class ID": c,
            "Class Name": class_names[c],
            "Support": int(cm[c, :].sum()),
            "Accuracy": round(float(cm[c, c] / cm[c, :].sum() * 100) if cm[c, :].sum() > 0 else 0.0, 2),
            "Precision": round(float(precisions[c] * 100), 2),
            "Recall (TPR)": round(float(recalls[c] * 100), 2),
            "F1-Score": round(float(f1s[c] * 100), 2),
            "FPR": round(float(fpr[c] * 100), 2),
        })

    return pd.DataFrame(rows)

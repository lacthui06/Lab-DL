"""Quét ngưỡng nhị phân và cố định ngưỡng chỉ từ tập validation.

Xác suất đầu vào luôn là P(POSITIVE), không phải confidence của lớp thắng.
Quy tắc quyết định: Positive khi p >= threshold. Tại p=0.5, quy tắc này
khác argmax hai logits nếu logits bằng nhau; khác biệt được ghi rõ thay vì ẩn.
"""
from pathlib import Path
import math
import numpy as np
from ..utils.runs import save_json


def _validate_inputs(labels, positive_probabilities):
    """Kiểm tra nhãn/xác suất trước khi sinh bất kỳ metric hoặc artifact nào."""
    try:
        labels = np.asarray(labels)
        probabilities = np.asarray(positive_probabilities)
    except (TypeError, ValueError) as exc:
        raise ValueError("Labels and positive probabilities must be numeric vectors") from exc
    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Labels and P(POSITIVE) must be one-dimensional vectors")
    if not len(labels) or len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities need equal nonzero lengths")
    if labels.dtype.kind not in "biuf" or probabilities.dtype.kind not in "biuf":
        raise ValueError("Labels and probabilities must contain numeric values")
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must contain only finite binary values 0 and 1")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("P(POSITIVE) must be finite and between 0 and 1")
    return labels.astype(np.int64, copy=False), probabilities.astype(np.float64, copy=False)


def _validate_threshold(threshold):
    """Không chấp nhận NaN/Inf hoặc ngưỡng nằm ngoài đoạn xác suất."""
    if isinstance(threshold, (bool, np.bool_)):
        raise ValueError("Threshold must be a numeric probability, not boolean")
    try:
        value = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Threshold must be a finite probability") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Threshold must be finite and between 0 and 1")
    return value


def _validated_threshold_grid(thresholds):
    """Lưới mặc định có 19 điểm; lưới tùy chỉnh được sắp xếp và bỏ trùng."""
    if thresholds is None:
        return [float(i / 20) for i in range(1, 20)]
    try:
        values = [_validate_threshold(t) for t in thresholds]
    except TypeError as exc:
        raise ValueError("Threshold grid must be a nonempty iterable") from exc
    if not values:
        raise ValueError("Threshold grid cannot be empty")
    return sorted(set(values))


def _ratio(numerator, denominator):
    """Quy ước zero_division=0 cho các trường hợp không có lớp hoặc dự đoán."""
    return float(numerator / denominator) if denominator else 0.0


def _metrics(labels, probabilities, threshold):
    """Tính trực tiếp từ confusion matrix, giữ macro F1 trên đủ hai lớp."""
    predicted = probabilities >= threshold
    positive = labels == 1
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & ~positive))
    fn = int(np.count_nonzero(~predicted & positive))
    tn = int(np.count_nonzero(~predicted & ~positive))
    binary_f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    negative_f1 = _ratio(2 * tn, 2 * tn + fp + fn)
    return {
        "threshold": float(threshold),
        "accuracy": _ratio(tp + tn, len(labels)),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "FPR": _ratio(fp, fp + tn),
        "binary_f1": binary_f1,
        "macro_f1": float((binary_f1 + negative_f1) / 2),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "negative_support": int(tn + fp), "positive_support": int(tp + fn),
        "sample_count": int(len(labels)),
    }


def binary_metrics_at_threshold(labels, positive_probabilities, threshold=0.5):
    """Đo một điểm vận hành cố định; hàm này không chọn hoặc tối ưu ngưỡng."""
    labels, probabilities = _validate_inputs(labels, positive_probabilities)
    threshold = _validate_threshold(threshold)
    return _metrics(labels, probabilities, threshold)


def scan_thresholds(labels, positive_probabilities, thresholds=None):
    """Tính đường metric theo ngưỡng; không suy ra ngưỡng tốt nhất từ dữ liệu."""
    labels, probabilities = _validate_inputs(labels, positive_probabilities)
    grid = _validated_threshold_grid(thresholds)
    return [_metrics(labels, probabilities, threshold) for threshold in grid]


def select_validation_threshold(labels, positive_probabilities, thresholds=None, split="validation"):
    """Chọn macro F1 validation tối đa, ưu tiên gần 0.5 rồi ngưỡng nhỏ hơn.

    Hàm từ chối mọi split khác validation. Sai số 1e-12 chỉ phục vụ xử lý
    bằng điểm số; không được dùng test để quyết định ngưỡng hay đổi tiêu chí.
    """
    if split != "validation":
        raise ValueError("Threshold selection requires split='validation'; test/train selection is forbidden")
    labels, probabilities = _validate_inputs(labels, positive_probabilities)
    curve = scan_thresholds(labels, probabilities, thresholds)
    best_score = max(row["macro_f1"] for row in curve)
    candidates = [row for row in curve if math.isclose(row["macro_f1"], best_score, rel_tol=0, abs_tol=1e-12)]
    selected = min(candidates, key=lambda row: (round(abs(row["threshold"] - .5), 12), row["threshold"]))
    return {
        "split": split,
        "purpose": "validation threshold selection; freeze before test evaluation",
        "selection_performed": True,
        "selection_metric": "macro_f1",
        "selected_threshold": selected["threshold"],
        "selected_metrics": selected,
        "baseline_threshold": 0.5,
        "baseline_metrics": _metrics(labels, probabilities, 0.5),
        "tie_break": "maximum validation macro_f1; closest to 0.5; smaller threshold (score atol=1e-12)",
        "decision_rule": "Positive if P(POSITIVE) >= threshold",
        "sample_count": int(len(labels)),
        "curve": curve,
    }


def _plot_threshold_curves(payload, path):
    """Lưu hình hai panel; chỉ đánh dấu selected khi đã chọn trên validation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curve = payload["curve"]
    thresholds = [row["threshold"] for row in curve]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for key, label in [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("binary_f1", "Binary F1")]:
        axes[0].plot(thresholds, [row[key] for row in curve], marker="o", markersize=3, label=label)
    for key, label in [("precision", "Precision (+)"), ("recall", "Recall / TPR (+)"),
                       ("specificity", "Specificity"), ("FPR", "FPR (+)")]:
        axes[1].plot(thresholds, [row[key] for row in curve], marker="o", markersize=3, label=label)
    for ax in axes:
        ax.axvline(.5, color="gray", linestyle=":", label="Default threshold 0.5")
        if payload["selected_threshold"] is not None:
            selected = payload["selected_threshold"]
            ax.axvline(selected, color="black", linestyle="--", label=f"Validation-selected {selected:.2f}")
        ax.set(xlabel="Decision threshold", ylabel="Metric", xlim=(0, 1), ylim=(0, 1.04))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="best")
    suffix = "selection on validation" if payload["selection_performed"] else "descriptive only; no threshold selected"
    fig.suptitle(f"Threshold sensitivity — {payload['split']} ({suffix})")
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(fig)


def export_threshold_analysis(labels, positive_probabilities, output_dir,
                              split="validation", thresholds=None, select=True):
    """Lưu bảng JSON và hình vào output_dir do caller cung cấp.

    select=True chỉ cho validation. select=False xuất phân tích mô tả, kể cả
    trên test, nhưng không ghi ngưỡng tối ưu hay được phép dùng nó để chọn model.
    Caller quản lý run context/manifest để toàn bộ artifact có cùng nguồn gốc.
    """
    if select:
        payload = select_validation_threshold(labels, positive_probabilities, thresholds, split)
    else:
        if split not in {"train", "validation", "test"}:
            raise ValueError("Diagnostic split must be train, validation or test")
        labels, probabilities = _validate_inputs(labels, positive_probabilities)
        payload = {
            "split": split,
            "purpose": "descriptive threshold sensitivity; forbidden for model/threshold selection on test",
            "selection_performed": False,
            "selection_metric": None,
            "selected_threshold": None,
            "selected_metrics": None,
            "baseline_threshold": 0.5,
            "baseline_metrics": _metrics(labels, probabilities, 0.5),
            "tie_break": None,
            "decision_rule": "Positive if P(POSITIVE) >= threshold",
            "sample_count": int(len(labels)),
            "curve": scan_thresholds(labels, probabilities, thresholds),
        }
    output_dir = Path(output_dir)
    if not select and (output_dir / "selected_threshold.json").exists():
        raise ValueError("Diagnostic output must not reuse a directory with an existing selected threshold")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "threshold_scan.json", payload)
    if select:
        save_json(output_dir / "selected_threshold.json", {
            key: payload[key] for key in ["split", "selection_metric", "selected_threshold",
                                        "selected_metrics", "baseline_metrics", "tie_break", "decision_rule"]
        })
    _plot_threshold_curves(payload, output_dir / "threshold_curves.png")
    return payload

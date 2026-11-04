"""Đọc và áp dụng policy ngưỡng đã được chọn trên validation, không fit lại."""
from pathlib import Path
import json
import math
from .utils.runs import hash_file


def checkpoint_identity(model_path):
    for name in ["model.safetensors", "pytorch_model.bin"]:
        path = Path(model_path) / name
        if path.is_file():
            return {"weights_filename": name, "weights_sha256": hash_file(path)}
    raise FileNotFoundError("Checkpoint weights are required to bind/verify a threshold policy")


def load_decision_policy(model_path, explicit_file=None):
    if explicit_file:
        path = Path(explicit_file)
        if not path.is_file(): raise FileNotFoundError(f"Threshold file not found: {path}")
    else:
        model_path = Path(model_path)
        candidates = [model_path / "decision_threshold.json", model_path.parent / "decision_threshold.json"]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None: return None
    policy = json.loads(path.read_text(encoding="utf-8"))
    threshold = policy.get("selected_threshold")
    if policy.get("split") != "validation" or isinstance(threshold, bool) or not isinstance(threshold, (float, int)):
        raise ValueError("Only a validation-selected threshold policy is accepted")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid selected threshold")
    expected = policy.get("checkpoint_identity")
    if not isinstance(expected, dict) or expected != checkpoint_identity(model_path):
        raise ValueError("Threshold policy does not match this checkpoint weights SHA256")
    return policy


def apply_policy(predictions, policy, labels):
    if policy is None: return predictions
    threshold = policy["selected_threshold"]
    for row in predictions:
        row["argmax_label"] = row["label"]
        idx = int(row["probabilities"][1] >= threshold)
        row["label"] = labels[idx]
        row["score"] = row["probabilities"][idx]
        row["decision_threshold"] = threshold
        row["threshold_selection_split"] = "validation"
    return predictions

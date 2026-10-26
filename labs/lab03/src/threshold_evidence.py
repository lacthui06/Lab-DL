"""Kiểm toán đọc-only các run chọn ngưỡng, đánh giá checkpoint và suy luận thật."""
from pathlib import Path, PurePosixPath
import math
import re
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, log_loss
from .evidence import read_json, verify_manifest
from .training.metrics import compute_metrics
from .training.thresholds import select_validation_threshold, binary_metrics_at_threshold
from .utils.runs import ROOT


def _same(actual, expected, location, atol=1e-12):
    """So sánh cấu trúc đã lưu với phép tính lại, không làm tròn trước kiểm tra."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Evidence structure differs: {location}")
        for key in expected:
            _same(actual[key], expected[key], f"{location}.{key}", atol)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Evidence length differs: {location}")
        for index, (a, b) in enumerate(zip(actual, expected)):
            _same(a, b, f"{location}[{index}]", atol)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected or (isinstance(expected, bool) and type(actual) is not bool):
            raise ValueError(f"Evidence value differs: {location}")
    elif isinstance(expected, (int, float)):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
            raise ValueError(f"Nonfinite or invalid evidence: {location}")
        if not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=atol):
            raise ValueError(f"Recomputed evidence differs: {location}")
    elif actual != expected:
        raise ValueError(f"Evidence value differs: {location}")


def _prediction_vectors(path, sample_count=1066):
    """Kiểm tra đủ chỉ số, nhãn cân bằng và xác suất trước khi tính metrics."""
    rows = read_json(path)
    if not isinstance(rows, list) or len(rows) != sample_count:
        raise ValueError(f"Expected {sample_count} predictions: {path.name}")
    if [r.get("index") for r in rows] != list(range(sample_count)):
        raise ValueError(f"Missing, reordered or duplicate prediction indices: {path.name}")
    probs = np.asarray([r.get("probabilities") for r in rows], dtype=float)
    labels = np.asarray([r.get("true_label") for r in rows])
    predictions = np.asarray([r.get("predicted_label") for r in rows])
    if (probs.shape != (sample_count, 2) or not np.isfinite(probs).all()
        or np.any((probs < 0) | (probs > 1))
        or not np.allclose(probs.sum(axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError(f"Invalid probability vectors: {path.name}")
    if labels.dtype.kind not in "biuf" or not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"Invalid binary labels: {path.name}")
    labels = labels.astype(int)
    if np.bincount(labels, minlength=2).tolist() != [533, 533]:
        raise ValueError(f"Unexpected balanced label counts: {path.name}")
    if not np.array_equal(predictions, probs.argmax(axis=1)):
        raise ValueError(f"Stored argmax predictions differ: {path.name}")
    if not all(isinstance(r.get("text"), str) and r["text"] for r in rows):
        raise ValueError(f"Missing source texts: {path.name}")
    return rows, labels, probs


def _checkpoint_origin(selection_run, checkpoint_path, policy):
    """Đối chiếu hash weights/provenance gốc nếu bản run train có trong checkout."""
    identity = policy.get("checkpoint_identity")
    if (not isinstance(identity, dict) or set(identity) != {"weights_filename", "weights_sha256"}
        or not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("weights_sha256", "")))
        or identity.get("weights_filename") not in {"model.safetensors", "pytorch_model.bin"}):
        raise ValueError("Invalid checkpoint identity in frozen threshold policy")
    original_id = PurePosixPath(checkpoint_path).parent.name
    candidates = [Path(selection_run).parent / original_id, ROOT / "outputs/runs" / original_id]
    original = next((p for p in candidates if (p / "manifest.json").is_file()), None)
    if original is None:
        return {"run": original_id, "original_manifest_available": False,
                "weights_sha256": identity["weights_sha256"], "weights_reexecuted_locally": False}
    manifest = read_json(original / "manifest.json")
    expected = manifest.get(f"best_model/{identity['weights_filename']}")
    if expected != identity["weights_sha256"]:
        raise ValueError("Threshold checkpoint SHA256 differs from original training manifest")
    provenance = read_json(original / "provenance.json")
    fingerprint = provenance.get("dataset_fingerprints", {}).get("validation")
    if fingerprint != policy.get("validation_dataset_fingerprint"):
        raise ValueError("Validation dataset fingerprint differs from original training run")
    return {"run": original_id, "original_manifest_available": True,
            "weights_sha256": expected, "validation_fingerprint": fingerprint,
            "weights_reexecuted_locally": False}


def audit_threshold_experiment(selection_run, evaluation_run, inference_run):
    """Tính lại toàn bộ bằng chứng mới; không tải data/model, không sửa artifact.

    Ngưỡng được chọn lại CHỈ từ validation để kiểm toán policy đã lưu. Test chỉ
    đo tại ngưỡng cố định đó và argmax; không quét/chọn ngưỡng tối ưu từ test.
    """
    selection_run, evaluation_run, inference_run = map(Path, (selection_run, evaluation_run, inference_run))
    paths = {"threshold_selection": selection_run, "evaluation": evaluation_run, "inference": inference_run}
    manifest_audits = {kind: verify_manifest(path) for kind, path in paths.items()}
    if any(v["omitted_checkpoint_files"] for v in manifest_audits.values()):
        raise ValueError("Supplemental run must contain all artifacts recorded in its manifest")
    policy = read_json(selection_run / "decision_threshold.json")
    if policy.get("split") != "validation" or policy.get("selection_performed") is not True:
        raise ValueError("Threshold policy must record validation-only selection")
    for kind in ("evaluation", "inference"):
        _same(read_json(paths[kind] / "decision_threshold.json"), policy, f"{kind}.frozen_policy")

    _, validation_labels, validation_probs = _prediction_vectors(selection_run / "validation_predictions.json")
    recomputed = select_validation_threshold(validation_labels, validation_probs[:, 1], split="validation")
    for key, value in recomputed.items():
        _same(policy.get(key), value, f"validation.policy.{key}")
    _same(read_json(selection_run / "validation_threshold/threshold_scan.json"), recomputed, "validation.scan")
    selected = read_json(selection_run / "validation_threshold/selected_threshold.json")
    selected_keys = ["split", "selection_metric", "selected_threshold", "selected_metrics",
                     "baseline_metrics", "tie_break", "decision_rule"]
    _same(selected, {key: recomputed[key] for key in selected_keys}, "validation.selected_policy")
    argmax_validation = compute_metrics((validation_probs, validation_labels))
    _same(read_json(selection_run / "validation_metrics.json"), argmax_validation, "validation.argmax_metrics")
    runtime_validation = read_json(selection_run / "validation_runtime_metrics.json")
    if isinstance(runtime_validation.get("validation_loss"), bool) or not isinstance(runtime_validation.get("validation_loss"), (int, float)) or not math.isfinite(runtime_validation["validation_loss"]) or runtime_validation["validation_loss"] < 0:
        raise ValueError("Missing finite validation loss from checkpoint prediction")

    cfg_selection = read_json(selection_run / "config.json")
    cfg_evaluation = read_json(evaluation_run / "config.json")
    cfg_inference = read_json(inference_run / "config.json")
    checkpoint_path = cfg_selection.get("threshold_checkpoint")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError("Missing threshold checkpoint provenance")
    if cfg_evaluation.get("evaluated_checkpoint") != checkpoint_path or cfg_inference.get("inference", {}).get("model") != checkpoint_path:
        raise ValueError("Selection, evaluation and inference do not use the same checkpoint")
    for cfg in (cfg_evaluation, cfg_inference):
        _same(cfg.get("exercise2"), cfg_selection.get("exercise2"), "checkpoint.config")
    origin = _checkpoint_origin(selection_run, checkpoint_path, policy)

    _, labels, probs = _prediction_vectors(evaluation_run / "all_predictions.json")
    argmax_metrics = compute_metrics((probs, labels))
    _same(read_json(evaluation_run / "test_metrics.json"), argmax_metrics, "test.argmax_metrics")
    threshold = float(policy["selected_threshold"])
    selected_metrics = binary_metrics_at_threshold(labels, probs[:, 1], threshold)
    expected_test = dict(selected_metrics, selection_split="validation", selection_performed_on_test=False)
    _same(read_json(evaluation_run / "test_threshold_metrics.json"), expected_test, "test.selected_metrics")
    runtime = read_json(evaluation_run / "test_runtime_metrics.json")
    loss = runtime.get("test_loss")
    if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
        raise ValueError("Missing finite nonnegative checkpoint test loss")
    reconstructed_nll = float(log_loss(labels, probs / probs.sum(axis=1, keepdims=True), labels=[0, 1]))
    if not math.isclose(loss, reconstructed_nll, rel_tol=0, abs_tol=1e-5):
        raise ValueError("Test loss differs from normalized saved-probability NLL")

    inference = read_json(inference_run / "predictions.json")
    inference_cfg = cfg_inference.get("inference", {})
    if not isinstance(inference, list) or len(inference) != 1:
        raise ValueError("Expected one recorded CLI inference example")
    _same(inference_cfg.get("threshold"), threshold, "inference.config.threshold")
    for row in inference:
        p = np.asarray(row.get("probabilities"), dtype=float)
        if p.shape != (2,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.isclose(p.sum(), 1, atol=1e-6, rtol=0):
            raise ValueError("Invalid inference probabilities")
        idx = int(p[1] >= threshold); names = ["NEGATIVE", "POSITIVE"]
        _same(row.get("text"), inference_cfg.get("text"), "inference.text")
        _same(row.get("decision_threshold"), threshold, "inference.threshold")
        _same(row.get("threshold_selection_split"), "validation", "inference.selection_split")
        _same(row.get("argmax_label"), names[int(p.argmax())], "inference.argmax_label")
        _same(row.get("label"), names[idx], "inference.selected_label")
        _same(row.get("score"), float(p[idx]), "inference.selected_score")

    predicted = (probs[:, 1] >= threshold).astype(int)
    argmax_cm = confusion_matrix(labels, probs.argmax(axis=1), labels=[0, 1]).tolist()
    threshold_cm = confusion_matrix(labels, predicted, labels=[0, 1]).tolist()
    return {
        "source_runs": {kind: path.name for kind, path in paths.items()},
        "verified_files": sum(v["verified_files"] for v in manifest_audits.values()),
        "manifest_audits": manifest_audits,
        "policy": policy,
        "validation_baseline": recomputed["baseline_metrics"],
        "validation_selected": recomputed["selected_metrics"],
        "validation_curve": recomputed["curve"],
        "validation_argmax": argmax_validation,
        "validation_loss": float(runtime_validation["validation_loss"]),
        "test_argmax": argmax_metrics,
        "test_selected": expected_test,
        "test_loss": float(loss),
        "reconstructed_test_nll": reconstructed_nll,
        "test_loss_comparison_atol": 1e-5,
        "confusion_matrices": {"argmax": argmax_cm, "selected_threshold": threshold_cm},
        "per_class_argmax": classification_report(labels, probs.argmax(axis=1), labels=[0, 1], target_names=["NEGATIVE", "POSITIVE"], output_dict=True),
        "per_class_selected": classification_report(labels, predicted, labels=[0, 1], target_names=["NEGATIVE", "POSITIVE"], output_dict=True),
        "checkpoint_origin": origin,
        "inference": inference,
        "selection_performed_on_test": False,
        "validation_sample_count": int(len(validation_labels)),
        "test_sample_count": int(len(labels)),
        "weights_reexecuted_locally": False,
    }

"""Kiểm chứng dữ liệu Kaggle và phát hiện sửa sai dù đã băm manifest lại."""
import json
import shutil
from pathlib import Path
import numpy as np
import pytest
from labs.lab03.src.threshold_evidence import audit_threshold_experiment
from labs.lab03.src.utils.runs import ROOT, hash_file

SELECTION = ROOT / "outputs/runs/20261026T083558Z_threshold_selection_b64cf564"
EVALUATION = ROOT / "outputs/runs/20261026T083612Z_evaluation_a724c16c"
INFERENCE = ROOT / "outputs/runs/20261026T083627Z_inference_0ed240e2"


def _copied(tmp_path):
    paths = []
    for source in [SELECTION, EVALUATION, INFERENCE]:
        target = tmp_path / source.name
        shutil.copytree(source, target)
        paths.append(target)
    return paths


def _rewrite(run, name, mutate, rehash=True):
    path = run / name
    content = json.loads(path.read_text())
    mutate(content)
    path.write_text(json.dumps(content))
    if rehash:
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest[name] = hash_file(path)
        manifest_path.write_text(json.dumps(manifest))


def test_actual_threshold_artifacts_recompute_without_models_or_mutation():
    paths = [SELECTION, EVALUATION, INFERENCE]
    before = {str(p): hash_file(p) for run in paths for p in run.rglob("*") if p.is_file()}
    result = audit_threshold_experiment(*paths)
    assert result["verified_files"] == 32
    assert result["policy"]["selected_threshold"] == .8
    assert result["validation_selected"]["macro_f1"] == pytest.approx(.8526632714595723)
    assert len(result["validation_curve"]) == 19
    assert result["test_argmax"]["accuracy"] == pytest.approx(895/1066)
    assert result["test_selected"]["accuracy"] == pytest.approx(896/1066)
    assert result["confusion_matrices"] == {
        "argmax": [[444,89],[82,451]], "selected_threshold": [[466,67],[103,430]]}
    assert result["test_loss"] == pytest.approx(.5442677736282349)
    assert result["checkpoint_origin"]["original_manifest_available"] is True
    assert result["checkpoint_origin"]["validation_fingerprint"] == "4d310508c191379e"
    assert result["selection_performed_on_test"] is False
    assert result["weights_reexecuted_locally"] is False
    assert {str(p):hash_file(p) for run in paths for p in run.rglob("*") if p.is_file()} == before


def test_threshold_audit_rejects_original_hash_corruption(tmp_path):
    selection, evaluation, inference = _copied(tmp_path)
    _rewrite(evaluation, "test_threshold_metrics.json", lambda row: row.update(accuracy=1), rehash=False)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        audit_threshold_experiment(selection, evaluation, inference)


def test_threshold_audit_rejects_metric_change_after_rehash(tmp_path):
    selection, evaluation, inference = _copied(tmp_path)
    _rewrite(evaluation, "test_threshold_metrics.json", lambda row: row.update(accuracy=1))
    with pytest.raises(ValueError, match="test.selected_metrics.accuracy"):
        audit_threshold_experiment(selection, evaluation, inference)


def test_threshold_audit_rejects_non_validation_selection(tmp_path):
    selection, evaluation, inference = _copied(tmp_path)
    _rewrite(selection, "decision_threshold.json", lambda row: row.update(split="test"))
    with pytest.raises(ValueError, match="validation-only"):
        audit_threshold_experiment(selection, evaluation, inference)


def test_threshold_audit_rejects_changed_frozen_policy(tmp_path):
    selection, evaluation, inference = _copied(tmp_path)
    _rewrite(inference, "decision_threshold.json", lambda row: row.update(selected_threshold=.7))
    with pytest.raises(ValueError, match="inference.frozen_policy.selected_threshold"):
        audit_threshold_experiment(selection, evaluation, inference)


def test_threshold_audit_rejects_invalid_probabilities_after_rehash(tmp_path):
    selection, evaluation, inference = _copied(tmp_path)
    _rewrite(selection, "validation_predictions.json", lambda rows: rows[0].update(probabilities=[.7,.8]))
    with pytest.raises(ValueError, match="probability vectors"):
        audit_threshold_experiment(selection, evaluation, inference)


def test_threshold_audit_rejects_duplicate_indices_after_rehash(tmp_path):
    selection, evaluation, inference = _copied(tmp_path)
    _rewrite(evaluation, "all_predictions.json", lambda rows: rows[1].update(index=0))
    with pytest.raises(ValueError, match="duplicate prediction indices"):
        audit_threshold_experiment(selection, evaluation, inference)


def test_threshold_audit_rejects_invented_loss_after_rehash(tmp_path):
    selection, evaluation, inference = _copied(tmp_path)
    _rewrite(evaluation, "test_runtime_metrics.json", lambda row: row.update(test_loss=.123))
    with pytest.raises(ValueError, match="Test loss differs"):
        audit_threshold_experiment(selection, evaluation, inference)


def test_threshold_audit_rejects_removed_selected_policy_metadata(tmp_path):
    selection, evaluation, inference = _copied(tmp_path)
    _rewrite(selection, "validation_threshold/selected_threshold.json", lambda row: row.clear())
    with pytest.raises(ValueError, match="structure differs"):
        audit_threshold_experiment(selection, evaluation, inference)

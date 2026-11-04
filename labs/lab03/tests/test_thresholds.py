"""Kiểm thử ngưỡng nhị phân, quy tắc lựa chọn và chống chọn ngưỡng trên test."""
import json
import numpy as np
import pytest
from labs.lab03.src.training.thresholds import (
    binary_metrics_at_threshold,
    scan_thresholds,
    select_validation_threshold,
    export_threshold_analysis,
)


def test_threshold_metrics_match_hand_computed_confusion_and_boundaries():
    result = binary_metrics_at_threshold([0, 0, 1, 1], [.1, .8, .5, .9], .5)
    assert {k: result[k] for k in ["TN", "FP", "FN", "TP"]} == {
        "TN": 1, "FP": 1, "FN": 0, "TP": 2}
    assert result["accuracy"] == .75
    assert result["precision"] == pytest.approx(2/3)
    assert result["recall"] == 1
    assert result["specificity"] == .5
    assert result["FPR"] == .5
    assert result["binary_f1"] == pytest.approx(.8)
    assert result["macro_f1"] == pytest.approx((2/3 + .8)/2)
    assert binary_metrics_at_threshold([0, 1], [0, 1], 0)["FP"] == 1
    assert binary_metrics_at_threshold([0, 1], [0, 1], 1)["TP"] == 1


def test_default_grid_and_validation_selection_can_change_operating_point():
    labels = [0, 0, 1, 1]
    probabilities = [.1, .2, .3, .4]
    curve = scan_thresholds(labels, probabilities)
    assert len(curve) == 19
    assert [row["threshold"] for row in curve] == pytest.approx(np.arange(1, 20)/20)
    result = select_validation_threshold(labels, probabilities)
    assert result["split"] == "validation"
    assert result["selected_threshold"] == .3
    assert result["selected_metrics"]["macro_f1"] == 1
    assert result["baseline_metrics"]["macro_f1"] < 1
    json.dumps(result, allow_nan=False)


def test_tie_break_uses_distance_to_half_then_smaller_threshold():
    labels = [0, 1]
    probabilities = [.1, .9]
    assert select_validation_threshold(labels, probabilities)["selected_threshold"] == .5
    result = select_validation_threshold(labels, probabilities, thresholds=[.7, .3])
    assert result["selected_threshold"] == .3
    assert [row["threshold"] for row in result["curve"]] == [.3, .7]


@pytest.mark.parametrize("split", ["test", "train", "TEST", "holdout"])
def test_selection_rejects_any_nonvalidation_split(split):
    with pytest.raises(ValueError, match="validation"):
        select_validation_threshold([0, 1], [.1, .9], split=split)


@pytest.mark.parametrize("labels,probabilities", [
    ([], []), ([0], [.1, .9]), ([0, 2], [.1, .9]), ([0, np.nan], [.1, .9]),
    ([0, 1], [np.nan, .9]), ([0, 1], [np.inf, .9]), ([0, 1], [-.1, .9]),
    ([0, 1], [.1, 1.1]), ([0, 1], [[.9, .1], [.1, .9]]),
    (["0", "1"], [.1, .9]),
])
def test_invalid_input_cannot_silently_produce_metrics(labels, probabilities):
    with pytest.raises(ValueError):
        binary_metrics_at_threshold(labels, probabilities)


@pytest.mark.parametrize("thresholds", [[], [-.01, .5], [.5, 1.01], [.5, float("nan")]])
def test_invalid_threshold_grid_is_rejected(thresholds):
    with pytest.raises(ValueError):
        scan_thresholds([0, 1], [.1, .9], thresholds)


def test_single_class_metrics_stay_finite_and_macro_covers_both_classes():
    result = binary_metrics_at_threshold([0, 0], [.1, .2])
    assert result["accuracy"] == 1
    assert result["macro_f1"] == .5
    assert result["recall"] == result["binary_f1"] == 0
    assert all(np.isfinite(v) for v in result.values())


def test_export_writes_numeric_evidence_and_figure_and_test_is_diagnostic_only(tmp_path):
    labels, probabilities = [0, 0, 1, 1], [.1, .2, .3, .4]
    validation = tmp_path / "validation_threshold"
    result = export_threshold_analysis(labels, probabilities, validation)
    saved = json.loads((validation / "threshold_scan.json").read_text())
    assert saved["selected_threshold"] == result["selected_threshold"] == .3
    assert json.loads((validation / "selected_threshold.json").read_text())["selected_threshold"] == .3
    assert (validation / "threshold_curves.png").stat().st_size > 1000
    diagnostic = tmp_path / "test_threshold_diagnostic"
    result = export_threshold_analysis(labels, probabilities, diagnostic, split="test", select=False)
    assert result["selected_threshold"] is None
    assert result["selection_performed"] is False
    assert not (diagnostic / "selected_threshold.json").exists()
    assert (diagnostic / "threshold_scan.json").is_file()
    with pytest.raises(ValueError, match="validation"):
        export_threshold_analysis(labels, probabilities, tmp_path / "invalid", split="test")
    assert not (tmp_path / "invalid").exists()

"""Artifacts trực quan phải truy vết được, cache không sửa run nguồn."""
import json
from pathlib import Path
import pytest
from labs.lab03.src.utils.runs import ROOT, hash_file
from labs.lab03.src.analysis import ensure_analysis, ensure_exercise1_analysis
from labs.lab03.src.evidence import verify_manifest


def test_real_analysis_exports_visuals_and_never_selects_on_test(tmp_path):
    source = ROOT / "outputs/runs/20261025T181550Z_finetuning_e02bced7"
    before = {p.name: hash_file(p) for p in source.iterdir() if p.is_file()}
    out = ensure_analysis(source, root=tmp_path)
    expected = {"roc_pr_curves.png", "calibration_curve.png", "probability_distribution.png",
                "class_distribution.png", "review_length_distribution.png", "per_class_metrics.png", "metric_comparison.png"}
    assert expected <= {p.name for p in out.glob("*.png")}
    summary = json.loads((out / "analysis_summary.json").read_text())
    assert summary["metrics"]["accuracy"] == pytest.approx(895/1066)
    assert 0 <= summary["roc_auc"] <= 1
    assert summary["selection_performed"] is False
    assert summary["validation_probabilities_available"] is False
    diagnostic = json.loads((out / "threshold_test_diagnostic/threshold_scan.json").read_text())
    assert diagnostic["selected_threshold"] is None
    assert not (out / "threshold_test_diagnostic/selected_threshold.json").exists()
    verify_manifest(out)
    assert ensure_analysis(source, root=tmp_path) == out
    assert {p.name: hash_file(p) for p in source.iterdir() if p.is_file()} == before


def test_exercise1_visuals_persist_tables_and_comparison(tmp_path):
    source = ROOT / "outputs/runs/20261025T181532Z_exercise1_71da6f40"
    out = ensure_exercise1_analysis(source, root=tmp_path)
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["tokenization"]) == 8
    assert len(summary["comparison"]) == 3
    assert len(list(out.glob("*.png"))) == 3
    verify_manifest(out)
    assert ensure_exercise1_analysis(source, root=tmp_path) == out


def test_exercise1_visuals_support_more_than_three_sentences(tmp_path):
    import shutil
    source = tmp_path / "source"
    shutil.copytree(ROOT / "outputs/runs/20261025T181532Z_exercise1_71da6f40", source)
    path = source / "predictions.json"
    predictions = json.loads(path.read_text())
    predictions["pipeline"].append(dict(predictions["pipeline"][0]))
    fourth = dict(predictions["manual"][0]); fourth["text"] = "Additional synthetic fixture sentence"
    predictions["manual"].append(fourth)
    path.write_text(json.dumps(predictions))
    cfgpath=source / "config.json"; cfg=json.loads(cfgpath.read_text())
    cfg["exercise1"]["sample_sentences"].append(fourth["text"]);cfgpath.write_text(json.dumps(cfg))
    manifest_path=source / "manifest.json";manifest=json.loads(manifest_path.read_text())
    for name in ["predictions.json","config.json"]:manifest[name]=hash_file(source/name)
    manifest_path.write_text(json.dumps(manifest))
    out = ensure_exercise1_analysis(source,root=tmp_path / "render")
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["comparison"]) == 4
    assert (out / "pipeline_manual_comparison.png").is_file()

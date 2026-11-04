"""Artifacts trực quan phải truy vết được, cache không sửa run nguồn."""
import json
from pathlib import Path
import pytest
from src.utils.runs import ROOT, hash_file
from src.analysis import ensure_exercise1_analysis
from src.evidence import verify_manifest


def test_exercise1_visuals_persist_tables_and_comparison(tmp_path, exercise_run):
    source = exercise_run
    out = ensure_exercise1_analysis(source, root=tmp_path)
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["tokenization"]) == 8
    assert len(summary["comparison"]) == 3
    assert len(list(out.glob("*.png"))) == 3
    verify_manifest(out)
    assert ensure_exercise1_analysis(source, root=tmp_path) == out


def test_exercise1_visuals_support_more_than_three_sentences(tmp_path, exercise_run):
    import shutil
    source = tmp_path / "source"
    shutil.copytree(exercise_run, source)
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

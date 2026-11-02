"""Publication privacy, file policy and replacement safety checks."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("publish_artifacts", Path(__file__).resolve().parents[1] / "scripts/publish_artifacts.py")
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


def test_private_paths_and_credentials():
    original = {"path": r"C:\Users\Alice\Documents\Lab DL\labs\lab01\outputs\runs\example", "formula": r"\frac{x}{y}"}
    cleaned = publisher.clean_value(original)
    assert cleaned["path"] == "<LAB>/outputs/runs/example"
    assert cleaned["formula"] == original["formula"]
    with pytest.raises(ValueError, match="credential"):
        publisher.sanitize("ghp_" + "a" * 36)


def test_report_file_policy():
    for name in ["figure.png", "history.csv", "metrics.json", "test_predictions.npz", "split_indices.npz"]:
        assert publisher.exclusion(Path(name)) is None
    for name in ["best.pt", "console.log", "prepared.npz", "study.db", "train_features.csv", "internal_changes.md"]:
        assert publisher.exclusion(Path(name)) is not None


def test_publish_and_failure_preserve_previous(tmp_path):
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        publisher.write_json(path, value)
    run = tmp_path / "outputs/runs/example"
    save(tmp_path / "config/workflow.json", {"runs": {"preparation": "outputs/runs/example"}})
    save(tmp_path / "config/publication.json", {"extra_runs": {}})
    save(run / "status.json", {"status": "completed"})
    save(run / "metrics.json", {"accuracy": 0.9})
    (run / "plot.png").write_bytes(b"fixture-image")
    (tmp_path / "walkthrough_lab_01.md").write_text("![plot](outputs/runs/example/plot.png)", encoding="utf-8")
    target = publisher.publish(tmp_path)
    assert (target / "preparation/plot.png").read_bytes() == b"fixture-image"
    publisher.publish(tmp_path)
    before = (target / "index.json").read_bytes()
    save(run / "status.json", {"status": "failed"})
    with pytest.raises(ValueError, match="not completed"):
        publisher.publish(tmp_path)
    assert (target / "index.json").read_bytes() == before

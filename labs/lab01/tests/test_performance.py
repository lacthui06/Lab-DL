"""Architecture evidence and isolated batch measurement contracts."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.mlp import FashionMLP


def test_architecture_matches_configuration_without_changing_rng(tmp_path):
    from src.utils.performance import show_model_architecture
    config = dict(seed=42, models={"fixture": dict(in_features=794, hidden_dims=[16])})
    before = torch.random.get_rng_state().clone()
    path = show_model_architecture(config, "fixture", root=tmp_path)
    assert torch.equal(before, torch.random.get_rng_state())
    text = (path / "model_architecture.txt").read_text()
    assert "in_features=794, out_features=16" in text
    assert "out_features=10" in text


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_batch_measurement_preserves_checkpoint_and_reports_units(tmp_path, legacy, batch_size):
    from src.utils.performance import _measure_worker
    torch.set_num_threads(2)
    model = FashionMLP(in_features=784, hidden_dims=[8])
    checkpoint = tmp_path / "best.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    original = checkpoint.read_bytes()
    config = dict(
        name="SmallMLP", in_features=784, hidden_dims=[8], seed=42,
        optimizer="Adam", learning_rate=.001, weight_decay=0., momentum=.9)
    if legacy:
        del config["in_features"], config["hidden_dims"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    data = tmp_path / "prepared.npz"
    np.savez(data, train_pixels=np.zeros((4, 784), dtype=np.float32),
             train_labels=np.arange(4), train_features=np.zeros((4, 10), dtype=np.float32))
    settings = dict(batch_size=batch_size, warmup=1, repeats=2, threads=2, device="cpu")
    row = _measure_worker(tmp_path, data, settings)
    assert checkpoint.read_bytes() == original
    assert row["batch_size"] == batch_size and row["measured_samples"] == batch_size * 2
    assert row["ms_per_sample"] == pytest.approx(row["median_batch_ms"] / batch_size)
    assert row["ram_peak_rss_mib_per_sample"] == pytest.approx(row["ram_peak_rss_mib"] / batch_size)
    assert row["vram_peak_allocated_mib_per_sample"] is None
    assert row["ms_per_sample"] > 0 and row["ram_peak_rss_mib"] > 0


def test_invalid_measurement_budget_is_rejected_before_spawning():
    from src.utils.performance import benchmark_models
    with pytest.raises(ValueError):
        benchmark_models([], Path("missing"), dict(batch_size=0, warmup=1, repeats=2, threads=2))

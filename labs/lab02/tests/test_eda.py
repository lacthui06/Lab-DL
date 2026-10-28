"""Protect data-only imports, fixed partitions, and independently checked features."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.eda import handcrafted, locked_split, duplicate_audit


def test_eda_import_has_no_training_dependencies():
    result = subprocess.run([sys.executable, '-c',
        "import sys; import src.eda; "
        "assert not any(m in sys.modules for m in ['torch', 'torchvision', 'src.models', 'src.experiment', 'src.notebook_views'])"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_features_on_constant_and_step_images():
    images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    images[1, :, 16:] = 255
    result = handcrafted(images, np.array([0, 1]), np.array([0, 1]))
    assert result.brightness.tolist() == [0., .5]
    assert result.contrast.tolist() == [0., .5]
    assert result.zero_fraction.tolist() == [1., .5]
    assert result.edge_density.iloc[0] == 0
    assert result.edge_density.iloc[1] == pytest.approx(1/31)


def test_duplicate_audit_detects_held_out_leakage():
    pool = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    pool[1] = 255
    indices = dict(train=np.array([0]), val=np.array([1]), test=np.array([0]))
    checks, duplicates = duplicate_audit(dict(train=pool, val=pool, test=pool),
        {s: np.array([0, 1]) for s in indices}, indices)
    assert len(duplicates) == 1
    assert duplicates.iloc[0].previous_split == 'train'
    assert duplicates.iloc[0]['split'] == 'test'
    assert checks.loc[checks.check == 'constant_images', 'count'].tolist() == [1, 1, 1]


def test_split_is_reused_and_tampering_rejected(tmp_path):
    labels = np.repeat(np.arange(10), 5000)
    first = locked_split(tmp_path, labels, 42)
    second = locked_split(tmp_path, labels, 42)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    assert not np.intersect1d(first['train'], first['val']).size
    np.testing.assert_array_equal(np.bincount(labels[first['train']]), [4500]*10)
    with pytest.raises(ValueError, match='metadata'):
        locked_split(tmp_path, labels, 43)
    (tmp_path/'split_indices.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='modified'):
        locked_split(tmp_path, labels, 42)


def test_raw_eda_matches_historical_training_split():
    root = Path(__file__).resolve().parents[1]
    run = root / json.loads((root/'outputs/selected_run.json').read_text())['historical_run']
    meta = json.loads((root/'outputs/eda/split_metadata.json').read_text())
    old = json.loads((run/'resnet18/head_trial_00/config.json').read_text())
    assert meta['indices_sha256'] == old['data_sha256']
    measured = json.loads((root/'outputs/eda/preprocessing_config.json').read_text())
    original = json.loads((run/'eda/summary.json').read_text())
    for key in ['train_mean_rgb', 'train_std_rgb']:
        np.testing.assert_allclose(measured[key], original[key], rtol=0, atol=1e-12)

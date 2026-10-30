"""Synthetic expectations fixed in README before implementation."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('diagnostics', Path(__file__).with_name('run_diagnostics.py'))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def test_v02_v03_hand_computed_metrics():
    p = np.zeros((2, 10))
    p[:, :2] = [[.9, .1], [.6, .4]]
    metrics, classes, counts, _, losses = d.summarize(p, np.array([0, 1]))
    assert counts[0, 0] == counts[1, 0] == 1
    assert metrics['accuracy'] == .5
    assert metrics['loss'] == pytest.approx((-np.log(.9)-np.log(.4))/2)
    assert losses[0] == pytest.approx(.1053605157)
    assert classes.loc[1, 'precision'] == 0
    assert np.isfinite(classes.select_dtypes('number')).all().all()


@pytest.mark.parametrize('val,train', [([1, 1], [2]), ([-1, 1], [2]), ([1, 5], [2]), ([1, 2], [2]), ([1.0, 2.0], [3]), ([], [3])])
def test_v04_invalid_indices(val, train):
    with pytest.raises(ValueError):
        d.validate_indices(val, train, pool_size=5, expected=2)


def test_v01_valid_indices():
    d.validate_indices(np.array([0, 2]), np.array([1, 3]), pool_size=4, expected=2)


@pytest.mark.parametrize('p,y', [(np.full((1, 10), np.nan), [0]),
                                  (np.ones((1, 9)), [0]), (np.ones((1, 10)), [0]),
                                  (np.full((1, 10), .1), [10]),
                                  (np.full((1, 10), .1), [0.0]),
                                  (np.empty((0, 10)), [])])
def test_v05_bad_probabilities(p, y):
    with pytest.raises(ValueError):
        d.summarize(p, np.asarray(y))


def test_v06_history_comparison():
    h = pd.DataFrame(dict(epoch=[1], val_accuracy=[.8], val_loss=[.5]))
    d.check_history(dict(accuracy=.8, loss=.500001), h, 1)
    with pytest.raises(ValueError):
        d.check_history(dict(accuracy=.7, loss=.5), h, 1)
    with pytest.raises(ValueError):
        d.check_history(dict(accuracy=.8, loss=.5), h, 2)


def test_v04_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        d.torch.load(tmp_path / 'missing.pt', weights_only=True)

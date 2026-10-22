"""Regression checks for missing dependencies and fresh runtime bootstrap."""
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))


def test_missing_optuna_raises_import_error(tmp_path, monkeypatch):
    import src.training.tuning as tuning
    from src.utils.runs import Run
    monkeypatch.setattr(tuning, "Run", lambda name, config, root=None: Run(name, config, root=tmp_path / "runs"))
    p = tmp_path / "data.npz"
    np.savez(p, train_pixels=np.zeros((2, 784), dtype=np.float32))
    with patch.dict(sys.modules, {"optuna": None}):
        with pytest.raises(ImportError):
            tuning.run_tuning(p, trials_per_optimizer=1, epochs=1, batch_size=2)


def test_fresh_runtime_import_then_create_preprocessing(tmp_path):
    code = r"""
import json
from pathlib import Path
from src.utils.helpers import EDA_DIR, CONFIG_DIR, BASE_DIR, resolve_normalization_stats
assert BASE_DIR == Path(__import__('os').environ['LAB01_WORK_DIR'])
assert CONFIG_DIR.joinpath('workflow.json').is_file()
assert not EDA_DIR.joinpath('preprocessing_config.json').exists()
try:
    resolve_normalization_stats()
except FileNotFoundError:
    pass
else:
    raise AssertionError('Normalization must require measured artifact')
EDA_DIR.joinpath('preprocessing_config.json').write_text(json.dumps({'normalization': {'mean':[0.2], 'std':[0.4]}}))
mean, std, src_str = resolve_normalization_stats()
assert mean == (0.2,)
assert std == (0.4,)
assert src_str.startswith('artifact:')
"""
    env = dict(os.environ, PYTHONPATH=str(LAB), LAB01_WORK_DIR=str(tmp_path), MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

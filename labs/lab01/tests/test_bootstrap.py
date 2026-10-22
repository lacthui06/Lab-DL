"""Regression checks for missing dependencies and fresh runtime bootstrap."""
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

def test_missing_optuna_does_not_return_measurements():
    from src.training.experiments import run_optuna_optimizer_study
    with patch.dict(sys.modules, {"optuna": None}):
        with pytest.raises(ImportError, match="Optuna"):
            run_optuna_optimizer_study(None, None, num_trials=1)

def test_fresh_runtime_import_then_create_preprocessing(tmp_path):
    code = r"""
import json
from pathlib import Path
from src import run_eda_and_fe_pipeline, get_transforms
from src.utils.helpers import EDA_DIR, CONFIG_DIR, BASE_DIR
assert BASE_DIR == Path(__import__('os').environ['LAB01_WORK_DIR'])
assert CONFIG_DIR.joinpath('augmentations.yaml').is_file()
assert not EDA_DIR.joinpath('preprocessing_config.json').exists()
try:
    get_transforms()
except FileNotFoundError:
    pass
else:
    raise AssertionError('Normalization must require measured artifact')
EDA_DIR.joinpath('preprocessing_config.json').write_text(json.dumps({'normalization': {'mean':[0.2], 'std':[0.4]}}))
t = get_transforms()
assert tuple(t.transforms[-1].mean) == (0.2,)
assert tuple(t.transforms[-1].std) == (0.4,)
"""
    env = dict(os.environ, PYTHONPATH=str(LAB), LAB01_WORK_DIR=str(tmp_path), MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

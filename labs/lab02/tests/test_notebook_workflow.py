"""Chế độ xem không chạy pipeline; chế độ thực thi phải được chọn rõ ràng."""
import json
from pathlib import Path
from unittest.mock import Mock
import sys
import types

import pytest

from src.utils import notebook_workflow as workflow
from src.utils import notebook_views


def test_view_never_imports_training(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'src.training.experiment', None)
    expected = (tmp_path, tmp_path / 'run', tmp_path / 'out')
    monkeypatch.setattr(notebook_views, 'context', lambda: expected)
    assert workflow.prepare_modeling(tmp_path) == expected


@pytest.mark.parametrize('mode', ['prepare', '', None, 'TRAIN'])
def test_modeling_invalid_mode(mode, tmp_path):
    with pytest.raises(ValueError, match='MODE'):
        workflow.prepare_modeling(tmp_path, mode)


def test_train_requires_config_and_uses_returned_run(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='TRAIN_CONFIG'):
        workflow.prepare_modeling(tmp_path, 'train')
    with pytest.raises(FileNotFoundError):
        workflow.prepare_modeling(tmp_path, 'train', 'missing.json')
    config = tmp_path / 'config.json'
    config.write_text('{}')
    run = tmp_path / 'new_run'
    runner = Mock(return_value=run)
    monkeypatch.setitem(sys.modules, 'src.training.experiment', types.SimpleNamespace(run_suite=runner))
    context = Mock(return_value=(tmp_path, run, tmp_path / 'out'))
    monkeypatch.setattr(notebook_views, 'context', context)
    monkeypatch.setenv('LAB02_RUN', 'old_run')
    result = workflow.prepare_modeling(tmp_path, 'train', 'config.json', 'cpu')
    runner.assert_called_once_with(config, work_dir=tmp_path, device_override='cpu')
    context.assert_called_once_with(run_path=run)
    assert result[1] == run


def test_eda_modes_and_missing_artifacts(tmp_path, monkeypatch):
    from src.datasets import eda
    build = Mock(return_value=tmp_path / 'outputs/eda')
    monkeypatch.setattr(eda, 'build', build)
    with pytest.raises(FileNotFoundError):
        workflow.prepare_eda(tmp_path)
    build.assert_not_called()
    assert workflow.prepare_eda(tmp_path, 'prepare', tmp_path / 'data') == build.return_value
    build.assert_called_once_with(tmp_path, tmp_path / 'data')
    with pytest.raises(ValueError):
        workflow.prepare_eda(tmp_path, 'train')


def test_benchmark_for_another_run_is_not_reused(tmp_path, capsys):
    (tmp_path / 'provenance.json').write_text(json.dumps({'training_run': 'old'}))
    assert not workflow.benchmark_matches(Path('new'), tmp_path)
    assert 'bỏ qua' in capsys.readouterr().out
    assert workflow.benchmark_matches(Path('old'), tmp_path)

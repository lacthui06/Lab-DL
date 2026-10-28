"""Chặn chọn sai run và giả định winner cố định khi gộp kết quả."""
import json
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from src import merged_results as merged, notebook_views as view

ROOT = Path(__file__).resolve().parents[1]


def test_explicit_run_bypasses_aggregate_selection(tmp_path, monkeypatch):
    requested = tmp_path/'requested'
    output = tmp_path/'report'
    monkeypatch.setenv('LAB02_RUN', str(requested))
    monkeypatch.setattr(view, 'context', lambda: (tmp_path, requested, output))
    monkeypatch.setattr(merged, '_winners', lambda *a: pytest.fail('Không gộp khi yêu cầu run riêng'))
    assert merged.context() == (tmp_path, requested, output)
    calls = []
    monkeypatch.setattr(view, 'quality_cost', lambda run, out: calls.append(('quality', run)))
    monkeypatch.setattr(view, 'efficiency', lambda run, out: calls.append(('cost', run)))
    monkeypatch.setattr(view, 'fine_search', lambda run, out, root: calls.append(('grid', run)))
    merged.summary(tmp_path, output, requested)
    merged.grid_results(tmp_path, output, requested)
    assert calls == [('quality', requested), ('cost', requested), ('grid', requested)]


def test_registry_uses_exact_paths_and_rejects_duplicate_candidates(tmp_path):
    """Thư mục nhiễu không được chọn thay run đã đăng ký."""
    run = tmp_path/'runs/wanted'
    (tmp_path/'runs/noise').mkdir(parents=True)
    (run/'resnet18').mkdir(parents=True)
    (run/'config.json').write_text(json.dumps({'models':['resnet18'], 'selection_policy':{}}))
    (run/'status.json').write_text('{"status":"completed"}')
    row = dict(id='candidate', trial=4, best_val_accuracy=.9, best_val_loss=.2, train_seconds=1.)
    pd.DataFrame([row]).to_csv(run/'resnet18/finetune_search_results.csv', index=False)
    (run/'resnet18/selection.json').write_text(json.dumps({'finetune':row}))
    (tmp_path/'outputs').mkdir()
    registry = {'group_runs':{'7':'runs/wanted'}, 'completed_fine_trials':1}
    (tmp_path/'outputs/selected_run.json').write_text(json.dumps(registry))
    runs, grid = merged.collect(tmp_path)
    assert runs == [run] and grid.iloc[0]['group'] == 7
    assert merged._winners(tmp_path)['resnet18']['trial'] == 4
    registry['group_runs']['8'] = 'runs/wanted'
    registry['completed_fine_trials'] = 2
    (tmp_path/'outputs/selected_run.json').write_text(json.dumps(registry))
    with pytest.raises(ValueError, match='trùng'):
        merged.collect(tmp_path)


def test_completed_config_matches_all_measured_candidates():
    _, grid = merged.collect(ROOT)
    cfg = json.loads((ROOT/'config/completed.json').read_text())
    assert len(grid) == 18
    expected = {c['id'] for c in cfg['finetune_candidates']}
    for _, part in grid.groupby('model'):
        assert set(part.id) == expected


def test_gpu_split_members_match_eda_despite_archive_byte_difference():
    """NPZ zip timestamps không phải khác biệt membership của split."""
    registry = json.loads((ROOT/'outputs/selected_run.json').read_text())
    paths = [ROOT/'outputs/eda/split_indices.npz',
             *(ROOT/p/'selected_indices.npz' for p in registry['group_runs'].values())]
    if not all(p.exists() for p in paths):
        pytest.skip('Split NPZ gốc chỉ lưu local, không đi cùng Git')
    with np.load(paths[0]) as expected:
        for path in paths[1:]:
            with np.load(path) as actual:
                for split in ('train', 'val', 'test'):
                    np.testing.assert_array_equal(actual[split], expected[split])

"""Inline reporting for the three completed Kaggle groups; never trains models."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from .notebook_views import _save
from .tuning import select_trial


def collect(root):
    rows = []
    runs = []
    for group in range(3):
        run = next((Path(root)/f'outputs/kaggle/group_{group:02d}/extracted/runs').iterdir())
        runs.append(run)
        for model in ('resnet18', 'densenet121'):
            frame = pd.read_csv(run/model/'finetune_search_results.csv')
            frame['model'], frame['group'] = model, group
            rows.append(frame)
    grid = pd.concat(rows, ignore_index=True)
    assert len(grid) == 18 and not grid.duplicated(['model', 'id']).any()
    return runs, grid


def grid_results(root, output):
    runs, grid = collect(root)
    policy = json.loads((runs[0]/'config.json').read_text())['selection_policy']
    selected = {model: select_trial(frame.to_dict('records'), policy)['id']
                for model, frame in grid.groupby('model')}
    grid['selected'] = [selected[m] == i for m, i in zip(grid.model, grid.id)]
    grid.to_csv(output/'all_18_fine_trials.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for model, ax in zip(('resnet18', 'densenet121'), axes):
        part = grid[grid.model == model].reset_index(drop=True)
        labels = part.id.str.replace('adamw_', 'AW ', regex=False).str.replace('sgd_', 'SGD ', regex=False)
        bars = ax.barh(labels, part.best_val_accuracy*100,
                       color=['#059669' if s else '#93c5fd' for s in part.selected])
        ax.bar_label(bars, fmt='%.2f', fontsize=8)
        ax.set(title=f'{model}: validation accuracy', xlabel='Accuracy (%)', xlim=(85, 97))
        ax.tick_params(axis='y', labelsize=8)
    _save(fig, output, 'all_18_trials')
    display(grid[['model', 'id', 'optimizer', 'backbone_lr', 'head_lr', 'unfreeze',
                  'scheduler', 'best_val_accuracy', 'best_val_loss', 'epochs_completed',
                  'best_epoch', 'train_seconds', 'stop_reason', 'selected']].round(5))


def summary(root, output):
    runs, grid = collect(root)
    policy = json.loads((runs[0]/'config.json').read_text())['selection_policy']
    result = pd.read_csv(runs[2]/'comparison.csv')
    for model, frame in grid.groupby('model'):
        winner = select_trial(frame.to_dict('records'), policy)
        assert winner['group'] == 2 and winner['trial'] == 0
    display(result[['model', 'stage', 'accuracy', 'loss', 'macro_f1',
                    'macro_precision', 'macro_recall', 'best_val_accuracy']].round(5))
    fine = result[result.stage == 'finetune'].copy()
    fine['selected_fit_minutes'] = fine.train_seconds/60
    fine['all_9_fine_trials_hours'] = fine.model.map(grid.groupby('model').train_seconds.sum())/3600
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, metric, scale, label in zip(axes, ['accuracy', 'loss', 'train_seconds'],
                                       [100, 1, 1/60], ['Test accuracy (%)', 'Test loss', 'Selected fit (minutes)']):
        bars = ax.bar(fine.model, fine[metric]*scale, color=['#2563eb', '#059669'])
        ax.bar_label(bars, fmt='%.3f')
        ax.set(title=label, ylim=(0, max(fine[metric]*scale)*1.18))
    _save(fig, output, 'gpu_quality_time')
    display(fine[['model', 'selected_fit_minutes', 'all_9_fine_trials_hours']].round(3))

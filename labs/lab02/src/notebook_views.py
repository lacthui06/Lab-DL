"""Compact artifact views for both notebooks; no training/model imports.

EDA consumes only eda/ evidence. Modeling views consume saved experiment tables.
New figures are saved under outputs/review/<run>/, preserving the original run.
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display


def context() -> tuple[Path, Path, Path]:
    """Resolve the lab, explicitly selected run, and presentation destination."""
    candidates = [Path(os.environ.get('LAB02_ROOT', '.')).resolve(),
                  *Path.cwd().resolve().parents]
    root = next(p for p in candidates if (p / 'src/notebook_views.py').exists())
    selected = json.loads((root / 'outputs/selected_run.json').read_text(encoding='utf-8'))
    run = Path(os.environ.get('LAB02_RUN', selected['run']))
    run = run if run.is_absolute() else root / run
    output = root / 'outputs/review' / run.name
    output.mkdir(parents=True, exist_ok=True)
    return root, run, output


def _save(fig, output: Path, name: str) -> None:
    """Persist and display the same figure, then release its memory."""
    fig.tight_layout()
    fig.savefig(output / f'{name}.png', dpi=130, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def data_overview(run: Path) -> None:
    """Show raw data metadata; missing integrity checks stay explicit."""
    stats = json.loads((run / 'eda/summary.json').read_text(encoding='utf-8'))
    display(pd.DataFrame([
        ['Dataset', stats['dataset']], ['Raw shape (HWC)', str(stats['original_shape'])],
        ['Classes', len(stats['classes'])], ['Statistics source', 'train only'],
        ['Duplicate / corrupt image audit', 'not computed in saved evidence'],
    ], columns=['Field', 'Value']))


def class_distribution(run: Path, output: Path) -> None:
    """Plot the recorded counts for the fixed split; do not resplit data."""
    counts = pd.read_csv(run / 'eda/class_counts.csv', index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    counts.plot.bar(ax=axes[0], color=['#2563eb', '#0d9488', '#f59e0b'])
    axes[0].set(title='Recorded class counts', ylabel='Images', xlabel='')
    counts.sum().plot.pie(ax=axes[1], autopct='%1.1f%%', ylabel='', title='Split sizes')
    _save(fig, output, 'class_distribution')


def raw_examples(run: Path) -> None:
    """Display the actual saved raw RGB training grid."""
    display(Image(filename=str(run / 'eda/training_examples.png'), width=650))


def channel_features(run: Path, output: Path) -> None:
    """Visualize measured RGB mean/std; no learned backbone features."""
    stats = json.loads((run / 'eda/summary.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for ax, key, title in zip(axes, ['train_mean_rgb', 'train_std_rgb'],
                              ['Train RGB mean [0,1]', 'Train RGB standard deviation']):
        bars = ax.bar(['R', 'G', 'B'], stats[key], color=['#dc2626', '#16a34a', '#2563eb'])
        ax.bar_label(bars, fmt='%.4f')
        ax.set(title=title, ylim=(0, max(stats[key]) * 1.22))
    _save(fig, output, 'channel_features')


def preprocessing(output: Path) -> None:
    """Illustrate the input contract and normalization, not a measured histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].axis('off')
    steps = ['Raw RGB: uint8 HWC 32 x 32 x 3',
             'Resize: 64 x 64 (recorded CPU profile)',
             'Scale /255 and reorder: float32 CHW',
             'Normalize each channel: (x - mean) / std']
    for i, step in enumerate(steps):
        axes[0].text(.5, .9-i*.24, step, ha='center', va='center', fontsize=10,
                     bbox={'boxstyle': 'round,pad=.5', 'fc': '#eff6ff', 'ec': '#93c5fd'})
        if i < 3:
            axes[0].annotate('', (.5, .76-i*.24), (.5, .83-i*.24),
                             arrowprops={'arrowstyle': '->'})
    x = np.linspace(0, 1, 100)
    for channel, mean, std, color in zip('RGB', [.485, .456, .406],
                                       [.229, .224, .225], ['r', 'g', 'b']):
        axes[1].plot(x, (x-mean)/std, color=color, label=channel)
    axes[1].axhline(0, color='gray', lw=.7)
    axes[1].set(title='Normalization mapping (illustration)',
                xlabel='Scaled pixel [0,1]', ylabel='Normalized value')
    axes[1].legend()
    _save(fig, output, 'preprocessing')


def quality_cost(run: Path, output: Path) -> None:
    """Compare quality and recorded costs, retaining the execution caveat."""
    metrics = pd.read_csv(run / 'comparison.csv')
    costs = pd.read_csv(run / 'cost_comparison.csv')
    table = metrics.merge(costs[['model', 'stage', 'pipeline_training_seconds', 'execution']],
                          on=['model', 'stage'], validate='one_to_one')
    names = table.model + '\n' + table.stage
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, scale, title in zip(axes, ['accuracy', 'loss', 'pipeline_training_seconds'],
                                    [100, 1, 1], ['Test accuracy (%)', 'Test loss', 'Pipeline training (s)']):
        bars = ax.bar(names, table[col]*scale, color=['#93c5fd', '#2563eb', '#5eead4', '#0d9488'])
        ax.bar_label(bars, fmt='%.6f' if col == 'loss' else '%.2f', fontsize=8)
        ax.set(title=title, ylim=(0, max(table[col]*scale)*1.18))
        ax.tick_params(axis='x', labelsize=8)
    fig.suptitle('Recorded costs: ResNet full-image forward; DenseNet cached prefix', fontsize=10)
    _save(fig, output, 'quality_cost')
    display(table[['model', 'stage', 'accuracy', 'loss', 'macro_f1', 'best_val_accuracy',
                   'train_seconds', 'pipeline_training_seconds', 'test_seconds', 'execution']].round(4))


def learning_curves(run: Path, output: Path) -> None:
    """Plot training/validation history without printing model architecture."""
    models = pd.read_csv(run / 'comparison.csv').model.unique()
    fig, axes = plt.subplots(len(models), 2, figsize=(11, 6), squeeze=False)
    for row, name in enumerate(models):
        for stage, folder, style in [('head', 'head_selected', '--'), ('finetune', 'finetune', '-')]:
            frame = pd.read_csv(run / name / folder / 'history.csv')
            for ax, metric in zip(axes[row], ['loss', 'accuracy']):
                for split, color in [('train', '#2563eb'), ('val', '#d97706')]:
                    ax.plot(frame.epoch, frame[f'{split}_{metric}'], style, marker='o',
                            color=color, label=f'{stage}/{split}')
                ax.set(title=f'{name}: {metric}', xlabel='Epoch within stage')
                ax.legend(fontsize=7)
                ax.grid(alpha=.2)
    _save(fig, output, 'learning_curves')


def lr_search(run: Path, output: Path) -> None:
    """Plot actual LR/batch search; optimizer was fixed to AdamW in this run."""
    models = pd.read_csv(run / 'comparison.csv').model.unique()
    fig, axes = plt.subplots(1, len(models), figsize=(10, 3.5), squeeze=False)
    for name, ax in zip(models, axes.flat):
        frame = pd.read_csv(run / name / 'search_results.csv')
        for batch, rows in frame.groupby('batch_size'):
            rows = rows.sort_values('lr')
            ax.plot(rows.lr, rows.best_val_accuracy*100, '-o', label=f'batch={batch}')
        ax.set(xscale='log', title=name, xlabel='Learning rate', ylabel='Best validation accuracy (%)')
        ax.legend()
        ax.grid(alpha=.2)
    _save(fig, output, 'lr_search')


def architecture_summary(run: Path) -> None:
    """Show only counts and trainable blocks, read from saved metadata."""
    table = pd.read_csv(run / 'comparison.csv')
    table = table[['model', 'stage', 'total', 'trainable']].copy()
    table['unfrozen_blocks'] = ['classifier' if row.stage == 'head' else
        ('layer4 + fc' if row.model == 'resnet18' else 'denseblock4 + norm5 + classifier')
        for row in table.itertuples()]
    display(table)


def class_metrics(run: Path) -> pd.DataFrame:
    """Compute precision, recall/TPR, F1 and FPR from saved confusion counts."""
    frames = []
    for row in pd.read_csv(run / 'comparison.csv').itertuples():
        folder = 'head_selected' if row.stage == 'head' else 'finetune'
        counts = pd.read_csv(run / row.model / folder / 'confusion_counts.csv', index_col=0)
        values = counts.to_numpy()
        tp = np.diag(values)
        fp = values.sum(0)-tp
        actual = values.sum(1)
        precision = tp / np.maximum(values.sum(0), 1)
        recall = tp / np.maximum(actual, 1)
        frames.append(pd.DataFrame({'model': row.model, 'stage': row.stage,
            'class': counts.index, 'precision': precision, 'recall_tpr': recall,
            'f1': 2*precision*recall/np.maximum(precision+recall, 1e-15),
            'fpr': fp/np.maximum(values.sum()-actual, 1)}))
    return pd.concat(frames, ignore_index=True)


def errors(run: Path, output: Path) -> None:
    """Show per-class recall/FPR for fine-tuned models and complete metric table."""
    table = class_metrics(run)
    fine = table[table.stage == 'finetune']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in zip(axes, ['recall_tpr', 'fpr'], ['Recall / TPR (%)', 'FPR (%)']):
        fine.pivot(index='class', columns='model', values=col).mul(100).plot.bar(ax=ax)
        ax.set(title=title, xlabel='')
        ax.legend(fontsize=8)
    _save(fig, output, 'class_rates')
    table.to_csv(output / 'class_metrics.csv', index=False)
    display(fine.round(4))


def evidence(run: Path, kind: str) -> None:
    """Display comparable fine-tune confusion or prediction grids side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, name in zip(axes, ['resnet18', 'densenet121']):
        ax.imshow(plt.imread(run / name / 'finetune' / f'{kind}.png'))
        ax.axis('off')
        ax.set_title(name)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def roundtrip(run: Path) -> None:
    """Display recorded save/load verification; no checkpoint required for viewing."""
    records = []
    for row in pd.read_csv(run / 'comparison.csv').itertuples():
        folder = 'head_selected' if row.stage == 'head' else 'finetune'
        check = json.loads((run / row.model / folder / 'roundtrip.json').read_text(encoding='utf-8'))
        records.append({'model': row.model, 'stage': row.stage, **check})
    display(pd.DataFrame(records))


def tuning_plan(root: Path) -> None:
    """Display proposals as unexecuted, never as new experimental results."""
    plan = json.loads((root / 'config/tuning_plan.json').read_text(encoding='utf-8'))
    display(pd.DataFrame(plan['experiments']))

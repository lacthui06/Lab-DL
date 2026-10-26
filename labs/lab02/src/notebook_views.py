"""Compact modeling artifact views; no training/model imports.

EDA uses the separate src.eda module. These views consume saved experiment tables.
New figures are saved under outputs/review/<run>/, preserving the original run.
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, Image, display


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


def quality_cost(run: Path, output: Path) -> None:
    """Compare quality and recorded costs, retaining the execution caveat."""
    metrics = pd.read_csv(run / 'comparison.csv')
    from .costs import cost_table
    costs = cost_table(run, metrics)
    table = metrics.drop(columns=['pipeline_training_seconds', 'execution'], errors='ignore').merge(
                          costs[['model', 'stage', 'pipeline_training_seconds', 'execution']],
                          on=['model', 'stage'], validate='one_to_one')
    names = table.model + '\n' + table.stage
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, scale, title in zip(axes, ['accuracy', 'loss', 'pipeline_training_seconds'],
                                    [100, 1, 1], ['Test accuracy (%)', 'Test loss', 'Pipeline training (s)']):
        bars = ax.bar(names, table[col]*scale, color=['#93c5fd', '#2563eb', '#5eead4', '#0d9488'])
        ax.bar_label(bars, fmt='%.6f' if col == 'loss' else '%.2f', fontsize=8)
        ax.set(title=title, ylim=(0, max(table[col]*scale)*1.18))
        ax.tick_params(axis='x', labelsize=8)
    policies = costs[costs.stage=='finetune'].execution.unique()
    fig.suptitle('Same execution policy' if len(policies)==1 else 'Different execution policies: raw time does not rank architectures', fontsize=10)
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
    depths = dict(zip(table.model+'_'+table.stage, table.get('unfreeze', pd.Series('last_block',index=table.index))))
    table = table[['model', 'stage', 'total', 'trainable']].copy()
    table['unfrozen_blocks'] = ['classifier' if row.stage == 'head' else
        ('layer4 + fc' if row.model == 'resnet18' else 'denseblock4 + norm5 + classifier')
        for row in table.itertuples()]
    for i, row in table.iterrows():
        if row.stage=='finetune' and depths[row.model+'_'+row.stage]=='last_two_blocks':
            table.loc[i,'unfrozen_blocks'] = ('layer3 + layer4 + fc' if row.model=='resnet18' else
                'denseblock3 + transition3 + denseblock4 + norm5 + classifier')
    display(table)


def architecture_visual(run: Path, output: Path) -> None:
    """Summarize parameter allocation visually; keep full saved structures collapsed."""
    import html
    table = pd.read_csv(run / 'comparison.csv')
    names = table.model + '\n' + table.stage
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.barh(names, (table.total-table.trainable)/1e6, label='Frozen', color='#cbd5e1')
    ax.barh(names, table.trainable/1e6, left=(table.total-table.trainable)/1e6,
            label='Trainable', color='#2563eb')
    ax.set(xlabel='Parameters (millions)', title='Classifier replacement and configured fine-tuning')
    ax.legend(loc='lower right')
    _save(fig, output, 'parameter_allocation')
    architecture_summary(run)
    for name in table.model.unique():
        content = html.escape((run / name / 'architecture.txt').read_text(encoding='utf-8'))
        display(HTML(f'<details><summary>{name}: full saved architecture</summary>'
                     f'<pre style="max-height:260px;overflow:auto">{content}</pre></details>'))


def tensorboard_evidence(run: Path) -> None:
    """Show actual saved monitoring evidence; distinguish from a live UI screenshot."""
    if (run/'tensorboard_evidence.png').exists():
        display(Image(filename=str(run / 'tensorboard_evidence.png'), width=1000))
    else:
        display(HTML('Monitoring figure not exported yet. Run scripts/finalize.py or open TensorBoard events for this run.'))


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
    """Visualize complete class metrics; keep only the weakest rows inline."""
    table = class_metrics(run)
    fine = table[table.stage == 'finetune']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in zip(axes, ['recall_tpr', 'fpr'], ['Recall / TPR (%)', 'FPR (%)']):
        fine.pivot(index='class', columns='model', values=col).mul(100).plot.bar(ax=ax)
        ax.set(title=title, xlabel='')
        ax.legend(fontsize=8)
    _save(fig, output, 'class_rates')
    table.to_csv(output / 'class_metrics.csv', index=False)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.8))
    for ax, column in zip(axes, ['precision', 'recall_tpr', 'f1', 'fpr']):
        matrix = fine.pivot(index='class', columns='model', values=column)
        im = ax.imshow(matrix.to_numpy()*100, vmin=0, vmax=100 if column != 'fpr' else 5, cmap='Blues')
        ax.set(xticks=range(2), xticklabels=matrix.columns, yticks=range(10),
               yticklabels=matrix.index, title=f'{column} (%)')
        ax.tick_params(axis='x', rotation=35, labelsize=8)
        for i in range(10):
            for j in range(2):
                value = matrix.iloc[i,j]*100
                ax.text(j, i, f'{value:.1f}', ha='center', va='center', fontsize=8,
                        color='white' if value > (55 if column != 'fpr' else 2.75) else 'black')
    _save(fig, output, 'class_metric_heatmaps')
    display(fine.sort_values('f1').groupby('model', sort=False).head(2).round(4))


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


def efficiency(run: Path, output: Path) -> None:
    """Separate search budget from the cost of retraining the selected recipe."""
    from .costs import cost_table, validation_recommendation
    table = cost_table(run, pd.read_csv(run/'comparison.csv'))
    fine = table[table.stage=='finetune'].copy()
    fig, axes = plt.subplots(1,2,figsize=(11,3.7))
    fine.set_index('model')[['feature_extraction_seconds','all_head_trials_fit_seconds',
        'all_finetune_trials_fit_seconds']].plot.barh(stacked=True, ax=axes[0])
    axes[0].set(title='All-search training cost',xlabel='Seconds',ylabel='')
    axes[0].legend(fontsize=7)
    fine.set_index('model')[['selected_recipe_training_seconds','test_seconds']].plot.barh(ax=axes[1])
    axes[1].set(title='Selected recipe / full test pass',xlabel='Seconds',ylabel='')
    axes[1].legend(fontsize=7)
    _save(fig,output,'training_cost_breakdown')
    display(fine[['model','execution','pipeline_training_seconds','selected_recipe_training_seconds','test_seconds']].round(2))
    cfg = json.loads((run/'config.json').read_text(encoding='utf-8'))
    choice = validation_recommendation(run,pd.read_csv(run/'comparison.csv'),cfg)
    display(pd.DataFrame([dict(model=choice['model'],val_accuracy=choice['best_val_accuracy'],
        val_loss=choice['best_val_loss'],time_comparable=choice['time_comparable'])]))


def fine_search(run: Path, output: Path, root: Path) -> None:
    """Show actual fine-search evidence when available; otherwise label the planned grid."""
    paths = list(run.glob('*/finetune_search_results.csv'))
    if paths:
        table = pd.concat([pd.read_csv(p).assign(model=p.parent.name) for p in paths],ignore_index=True)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for name, rows in table.groupby('model'):
            axes[0].scatter(rows.train_seconds, rows.best_val_accuracy*100,label=name)
            axes[1].scatter(rows.backbone_lr,rows.best_val_loss,label=name)
        axes[0].set(xlabel='Fit seconds / candidate',ylabel='Validation accuracy (%)',title='Measured candidates')
        axes[1].set(xscale='log',xlabel='Backbone LR',ylabel='Validation loss at chosen epoch')
        for ax in axes: ax.legend(); ax.grid(alpha=.2)
    else:
        cfg=json.loads((root/'config/tuned.json').read_text(encoding='utf-8'))
        table=pd.DataFrame(cfg['finetune_candidates'])
        fig,ax=plt.subplots(figsize=(10,3.5))
        for name, rows in table.groupby('optimizer'):
            for scheduler, part in rows.groupby('scheduler'):
                ax.scatter(part.backbone_lr, part.unfreeze, label=f'{name}/{scheduler}',
                    marker='o' if scheduler=='none' else 'x',s=90)
        ax.set(xscale='log',xlabel='Backbone LR (head LR = 10 x)',title='Planned candidate grid: NOT EXECUTED')
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    _save(fig,output,'finetune_search')

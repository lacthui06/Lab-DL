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
    from ..utils.plots import save_figure
    fig.tight_layout()
    save_figure(fig, output / f'{name}.png', dpi=130, bbox_inches='tight')
    display(fig)
    plt.close(fig)


def quality_cost(run: Path, output: Path) -> None:
    """Compare quality and recorded costs, retaining the execution caveat."""
    metrics = pd.read_csv(run / 'comparison.csv')
    from ..utils.costs import cost_table
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
    """Display the exact per-stage plotting implementation used by training reports."""
    from ..utils.plots import plot_learning_curves
    plot_learning_curves(run, output, show=True)


def epoch_budget(run: Path, root: Path) -> None:
    """Contrast this run's recorded epochs with the separate tuning budget."""
    recorded = json.loads((run/'config.json').read_text(encoding='utf-8'))
    active = json.loads((root/'config/tuned.json').read_text(encoding='utf-8'))
    rows = []
    for name in pd.read_csv(run/'comparison.csv').model.unique():
        for stage, folder, key in [('Head', 'head_selected', 'head_epochs'), ('Fine-tune', 'finetune', 'finetune_epochs')]:
            history = pd.read_csv(run/name/folder/'history.csv')
            summary_path = run/name/folder/'training_summary.json'
            summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
            rows.append({'Model':name, 'Stage':stage, 'Recorded epochs':len(history),
                'Original configured budget':recorded[key],
                'Stop evidence':summary.get('stop_reason', 'configured budget completed'
                    if len(history)==recorded[key] else 'not logged'),
                'Separate tuning budget':str(active[key]) if stage=='Head' else
                    f"min {active['early_stopping']['min_epochs']}, max {active[key]}"})
    display(pd.DataFrame(rows))
    trends = []
    for name in pd.read_csv(run/'comparison.csv').model.unique():
        h = pd.read_csv(run/name/'finetune/history.csv')
        trends.append({'Model':name, 'Fine epochs':len(h),
            'Val loss: first -> last':f'{h.val_loss.iloc[0]:.4f} -> {h.val_loss.iloc[-1]:.4f}',
            'Last train-val gap (pp)':round((h.train_accuracy.iloc[-1]-h.val_accuracy.iloc[-1])*100, 2),
            'Convergence':'not established; more validation history needed'})
    display(pd.DataFrame(trends))


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
    fig, axes = plt.subplots(2, 1, figsize=(13, 4.2))
    diagrams = [
        ('ResNet18', ['Stem +\nlayer1 / layer2', 'layer3', 'layer4', 'avgpool + fc'],
         ['#cbd5e1', '#fbbf24', '#93c5fd', '#93c5fd']),
        ('DenseNet121', ['Stem + blocks1/2\n+ transitions1/2', 'denseblock3', 'transition3',
                         'denseblock4\n+ norm5', 'pool + classifier'],
         ['#cbd5e1', '#fbbf24', '#fbbf24', '#93c5fd', '#93c5fd']),
    ]
    for ax, (name, blocks, colors) in zip(axes, diagrams):
        ax.set(xlim=(0, len(blocks)), ylim=(0, 1), title=name)
        ax.axis('off')
        for i, (block, color) in enumerate(zip(blocks, colors)):
            ax.text(i+.5, .5, block, ha='center', va='center', fontsize=10,
                    bbox={'boxstyle':'round,pad=.6', 'fc':color, 'ec':'#64748b'})
            if i < len(blocks)-1:
                ax.annotate('', (i+1.18, .5), (i+.84, .5), arrowprops={'arrowstyle':'->'})
    fig.suptitle('Gray: frozen | Blue: last-block recipe | Yellow: additionally open in two-block trials', fontsize=11)
    _save(fig, output, 'unfreeze_block_map')
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
        ax.imshow(matrix.to_numpy()*100, vmin=0, vmax=100 if column != 'fpr' else 5, cmap='Blues')
        ax.set(xticks=range(2), xticklabels=matrix.columns, yticks=range(10),
               yticklabels=matrix.index, title=f'{column} (%)')
        ax.tick_params(axis='x', rotation=35, labelsize=8)
        for i in range(10):
            for j in range(2):
                value = matrix.iloc[i,j]*100
                ax.text(j, i, f'{value:.1f}', ha='center', va='center', fontsize=8,
                        color='white' if value > (55 if column != 'fpr' else 2.75) else 'black')
    _save(fig, output, 'class_metric_heatmaps')
    display(table[['model', 'stage', 'class', 'precision', 'recall_tpr', 'f1', 'fpr']].round(4))


def evidence(run: Path, kind: str) -> None:
    """Display comparable fine-tune confusion or prediction grids side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, name in zip(axes, ['resnet18', 'densenet121']):
        ax.imshow(plt.imread(run / name / 'finetune' / f'{kind}.png'))
        ax.axis('off')
        ax.set_title(name)
    plt.tight_layout()
    display(fig)
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
    from ..utils.costs import cost_table, validation_recommendation
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
        columns = ['model', 'trial', 'optimizer', 'backbone_lr', 'head_lr', 'unfreeze',
                   'scheduler', 'epochs_completed', 'best_val_accuracy', 'best_val_loss',
                   'train_seconds', 'stop_reason']
        measured = table[[c for c in columns if c in table.columns]].copy()
        if 'best_val_accuracy' in measured:
            measured['best_val_accuracy'] = measured['best_val_accuracy'].mul(100).round(2)
        display(measured.rename(columns={'best_val_accuracy':'Validation accuracy (%)',
            'best_val_loss':'Validation loss', 'train_seconds':'Fit seconds',
            'epochs_completed':'Actual epochs', 'trial':'Candidate'}))
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for name, rows in table.groupby('model'):
            axes[0].scatter(rows.train_seconds, rows.best_val_accuracy*100,label=name)
            axes[1].scatter(rows.backbone_lr,rows.best_val_loss,label=name)
            for row in rows.itertuples():
                label = f'{name}: C{row.trial}'
                axes[0].annotate(label, (row.train_seconds, row.best_val_accuracy*100),
                                 xytext=(4, 4), textcoords='offset points', fontsize=7)
                axes[1].annotate(label, (row.backbone_lr, row.best_val_loss),
                                 xytext=(4, 4), textcoords='offset points', fontsize=7)
        axes[0].set(xlabel='Fit seconds / candidate',ylabel='Validation accuracy (%)',title='Measured candidates')
        axes[1].set(xscale='log',xlabel='Backbone LR',ylabel='Validation loss at chosen epoch')
        for ax in axes: ax.legend(); ax.grid(alpha=.2)
    else:
        cfg=json.loads((root/'config/tuned.json').read_text(encoding='utf-8'))
        table=pd.DataFrame(cfg['finetune_candidates'])
        rows=[]
        for (optimizer, scheduler, depth), part in table.groupby(['optimizer','scheduler','unfreeze'],sort=False):
            rows.append({'Optimizer':optimizer, 'Backbone LR':', '.join(f'{v:g}' for v in sorted(part.backbone_lr.unique())),
                'Head LR': '10 x backbone LR',
                'Opened blocks':'last block' if depth=='last_block' else 'last two blocks',
                  'Scheduler':scheduler, 'Candidates':len(part), 'Status':'NO RESULTS IN THIS RUN'})
        display(pd.DataFrame(rows))
        display(HTML(
            '<p><strong>Cách đọc một candidate:</strong> AdamW, backbone LR '
            '<code>0.00001</code>, classifier LR <code>0.0001</code>, mở block cuối, '
            'cosine giảm LR theo epoch. Đây là <strong>một cấu hình cần train</strong>, '
            'chưa phải một điểm accuracy/loss.</p>'
            '<p>Grid gồm 1 control LR cố định + 6 AdamW + 6 SGD = '
            '<strong>13 candidate mỗi model</strong>. Hai số LR dùng bước cập nhật '
            'khác nhau cho backbone và classifier. Depth là phạm vi trọng số được học; '
            'scheduler là cách thay đổi LR trong quá trình train.</p>'))
        display(pd.DataFrame([{'Head search batches':str(cfg['batch_sizes']),
            'Head epochs':cfg['head_epochs'], 'Fine-tune max epochs':cfg['finetune_epochs'],
            'Fine-tune minimum':cfg['early_stopping']['min_epochs'],
            'Stop rule':'validation loss: patience 7, min_delta 0.0001'}]))
        return
    _save(fig,output,'finetune_search')

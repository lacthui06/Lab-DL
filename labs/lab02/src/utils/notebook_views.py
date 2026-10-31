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
from IPython.display import HTML, display


def context() -> tuple[Path, Path, Path]:
    """Resolve the lab, explicitly selected run, and presentation destination."""
    configured = os.environ.get('LAB02_ROOT')
    try:
        root = Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[1]
    except OSError as exc:
        raise RuntimeError('Cannot resolve Lab02. Set LAB02_ROOT to an existing absolute Lab02 directory.') from exc
    if not (root / 'src/notebook_views.py').is_file():
        raise RuntimeError(f'Invalid Lab02 directory: {root}. Set LAB02_ROOT to the directory containing src/notebook_views.py.')
    selected = json.loads((root / 'outputs/selected_run.json').read_text(encoding='utf-8'))
    run = Path(os.environ.get('LAB02_RUN', selected['run']))
    run = run if run.is_absolute() else root / run
    output = root / 'outputs/review' / run.name
    output.mkdir(parents=True, exist_ok=True)
    return root, run, output


def _save(fig, output: Path, name: str) -> None:
    """Persist and display the same figure, then release its memory."""
    from .reporting import save_figure
    fig.tight_layout()
    save_figure(fig, output / f'{name}.png', dpi=130, bbox_inches='tight')
    display(fig)
    plt.close(fig)


def quality_cost(run: Path, output: Path) -> None:
    """Compare quality and recorded costs, retaining the execution caveat."""
    metrics = pd.read_csv(run / 'comparison.csv')
    from ..training.costs import cost_table
    costs = cost_table(run, metrics)
    table = metrics.drop(columns=['pipeline_training_seconds', 'execution'], errors='ignore').merge(
                          costs[['model', 'stage', 'pipeline_training_seconds', 'execution']],
                          on=['model', 'stage'], validate='one_to_one')
    names = table.model + '\n' + table.stage
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col, scale, title in zip(axes.flat, ['loss', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1'],
                                    [1, 100, 100, 100, 100], ['Test loss (lower is better)', 'Test accuracy (%)',
                                     'Test Macro precision (%)', 'Test Macro recall / TPR (%)', 'Test Macro F1 (%)']):
        bars = ax.bar(names, table[col]*scale, color=['#93c5fd', '#2563eb', '#5eead4', '#0d9488'])
        ax.bar_label(bars, fmt='%.4f' if col == 'loss' else '%.2f', fontsize=8)
        ax.set(title=title, ylim=(0, max(table[col]*scale)*1.18))
        ax.tick_params(axis='x', labelsize=8)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.05, .85, 'Macro: equal weight for each class.\n\n'
        'Recall and TPR are the same metric.\n\n'
        'Macro F1 = mean of per-class F1.\n\n'
        'Loss is not an error percentage.\n\n'
        'Higher is better for accuracy/P/R/F1.', va='top', fontsize=11)
    fig.suptitle('Recorded test quality; model selection uses validation, not test', fontsize=10)
    _save(fig, output, 'quality_cost')
    summary = table[['model', 'stage', 'loss', 'accuracy', 'macro_precision', 'macro_recall',
                     'macro_f1', 'execution']].copy()
    percentages = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
    summary[percentages] *= 100
    summary.rename(columns={key: key+' (%)' for key in percentages}, inplace=True)
    display(summary.round(4))
    fine = metrics[metrics.stage == 'finetune'].sort_values('accuracy', ascending=False)
    if len(fine) == 2:
        gap = (fine.accuracy.iloc[0]-fine.accuracy.iloc[1])*100
        display(HTML(f'<p>Test accuracy gap: {gap:.2f} percentage points. '
                     'One seed does not establish an architecture advantage; use validation for selection.</p>'))


def learning_curves(run: Path, output: Path) -> None:
    """Display the exact per-stage plotting implementation used by training reports."""
    from .reporting import plot_learning_curves
    plot_learning_curves(run, output, show=True)


def epoch_budget(run: Path, root: Path) -> None:
    """Contrast measured run history with the not-yet-executed active config."""
    recorded = json.loads((run/'config.json').read_text(encoding='utf-8'))
    active = json.loads((root/'config/tuned.json').read_text(encoding='utf-8'))
    rows = []
    for name in pd.read_csv(run/'comparison.csv').model.unique():
        for stage, folder, key in [('Head', 'head_selected', 'head_epochs'), ('Fine-tune', 'finetune', 'finetune_epochs')]:
            history = pd.read_csv(run/name/folder/'history.csv')
            rows.append({'Model':name, 'Stage':stage, 'Recorded epochs':len(history),
                'Original configured budget':recorded[key],
                'New budget (NOT RUN)':str(active[key]) if stage=='Head' else
                    f"min {active['early_stopping']['min_epochs']}, max {active[key]}"})
    display(pd.DataFrame(rows))


def _search_table(frame: pd.DataFrame, caption: str) -> None:
    """Display styled HTML and a readable plain-text fallback from measured rows."""
    formats = {column: '{:.0e}' for column in frame if column.endswith('LR')}
    formats.update({'Val accuracy (%)': '{:.2f}', 'Val loss': '{:.4f}',
                    'Fit (min)': '{:.2f}'})
    style = (frame.style.hide(axis='index').format(formats, escape='html')
             .set_caption(caption)
             .apply(lambda row: ['background-color: #e6f4ea; color: #153d23; font-weight: 600'
                                if row['Selected'] == 'Yes' else '' for _ in row], axis=1)
             .set_table_styles([
                 {'selector': 'caption', 'props': [('text-align', 'left'), ('font-weight', 'bold'), ('padding', '10px 0')]},
                 {'selector': 'th', 'props': [('text-align', 'left'), ('padding', '8px 12px'), ('border-bottom', '2px solid #94a3b8')]},
                 {'selector': 'td', 'props': [('padding', '8px 12px'), ('border-bottom', '1px solid #cbd5e1')]},
             ]))
    display({'text/html': style.to_html(), 'text/plain': frame.to_string(index=False, formatters={
        column: fmt.format for column, fmt in formats.items() if column in frame})}, raw=True)


def lr_search(run: Path, output: Path) -> None:
    """Compare measured head trials, highlighting saved validation selections."""
    records = []
    for name in pd.read_csv(run / 'comparison.csv').model.unique():
        frame = pd.read_csv(run/name/'search_results.csv')
        selected = json.loads((run/name/'selection.json').read_text(encoding='utf-8'))['head']
        for row in frame.sort_values(['lr', 'batch_size']).itertuples():
            records.append({'Model': name, 'Trial': row.trial, 'Head LR': row.lr,
                            'Batch': row.batch_size, 'Best epoch': row.best_epoch,
                            'Val accuracy (%)': row.best_val_accuracy * 100,
                            'Val loss': row.best_val_loss, 'Fit (min)': row.train_seconds / 60,
                            'Selected': 'Yes' if row.trial == selected['trial'] else 'No'})
    _search_table(pd.DataFrame(records), 'Head search: measured trials; green = selected within each model')
    display(HTML('<p>Selection uses the saved validation policy, not test. Loss is measured at the chosen checkpoint, '
                 'not necessarily the minimum observed loss. Fit time excludes feature extraction.</p>'))


def architecture_summary(run: Path) -> None:
    """Show only counts and trainable blocks, read from saved metadata."""
    table = pd.read_csv(run / 'comparison.csv')
    depths = dict(zip(table.model+'_'+table.stage, table.get('unfreeze', pd.Series('last_block',index=table.index))))
    table = table[['model', 'stage', 'total', 'trainable']].copy()
    table['unfrozen_blocks'] = ['classifier' if row.stage == 'head' else
        ('layer4 + fc' if row.model == 'resnet18' else 'denseblock4 + norm5 + classifier')
        for row in table.itertuples()]
    for i, row in table.iterrows():
        if row.stage=='finetune' and depths[row.model+'_'+row.stage]=='scheduled_tail':
            table.loc[i,'unfrozen_blocks'] = ('layer4.1 + fc' if row.model=='resnet18' else
                'block3:11–12,18–24 + transition3 + block4:1–4,13–16 + norm5 + classifier')
        if row.stage=='finetune' and depths[row.model+'_'+row.stage]=='last_units':
            table.loc[i,'unfrozen_blocks'] = ('layer4.1 + fc' if row.model=='resnet18' else
                'denseblock4.denselayer13–16 + norm5 + classifier')
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
    for i, row in enumerate(table.itertuples()):
        ax.text(row.total/1e6 + .12, i,
                f'{row.trainable:,} trainable ({100*row.trainable/row.total:.2f}%)', va='center', fontsize=8)
    ax.set(xlabel='Parameters (millions)', title='Parameter allocation; bar segments do not represent layer order',
           xlim=(0, table.total.max()/1e6*1.6))
    ax.legend(loc='upper right')
    _save(fig, output, 'parameter_allocation')
    architecture_summary(run)
    display(HTML('<p>Image → backbone blocks → pooling → classifier → 10 logits. '
        'Head-only updates the classifier; fine-tuning also updates the configured final blocks. '
        'Frozen blocks still compute features. BatchNorm running statistics stay fixed. '
        'Parameter counts do not determine accuracy or runtime.</p>'))
    for name in table.model.unique():
        content = html.escape((run / name / 'architecture.txt').read_text(encoding='utf-8'))
        display(HTML(f'<details><summary>{name}: full saved architecture</summary>'
                     f'<pre style="max-height:260px;overflow:auto">{content}</pre></details>'))


def tensorboard_evidence(run: Path) -> None:
    """Render saved scalars without overlapping stage epochs or rewriting old evidence."""
    from matplotlib.ticker import MaxNLocator, PercentFormatter
    path = run/'tensorboard_scalars.csv'
    if not path.exists():
        display(HTML('No exported scalars. Open live TensorBoard using the event files for this run.'))
        return
    data = pd.read_csv(path)
    models = pd.read_csv(run/'comparison.csv').model.unique()
    fig, axes = plt.subplots(4, len(models), figsize=(12, 12), squeeze=False)
    for j, name in enumerate(models):
        for stage_index, (stage, key) in enumerate([('Head', 'head'), ('Fine-tune', 'finetune')]):
            for metric_index, metric in enumerate(['loss', 'accuracy']):
                ax = axes[stage_index*2+metric_index, j]
                for split, color, style in [('train', '#2563eb', '-o'), ('val', '#d97706', '--s')]:
                    rows = data[(data['model'] == name) & (data['stage'] == key) &
                                (data['tag'] == f'{split}_{metric}')].sort_values('epoch')
                    ax.plot(rows.epoch, rows.value, style, color=color, label=split)
                ax.set(title=f'{name} | {stage} | {metric}', xlabel='Epoch within stage', ylabel=metric)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                if metric == 'accuracy':
                    ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.legend(fontsize=8)
                ax.grid(alpha=.2)
    fig.suptitle('Saved TensorBoard scalars; separate stages; not a live dashboard')
    plt.tight_layout(rect=(0, 0, 1, .97))
    display(fig)
    plt.close(fig)


def monitoring_status(run: Path) -> None:
    """Inspect local artifacts, without implying a live server is running."""
    events = list(run.rglob('events.out.tfevents.*'))
    checkpoints = list(run.rglob('best.pt'))
    display(pd.DataFrame([{'run': run.name, 'event_files': len(events),
        'best_checkpoints': len(checkpoints), 'live_server': 'not probed',
        'historical_server_record': (run/'tensorboard_server_verification.json').exists()}]))
    display(HTML('<p>Live TensorBoard needs event files: <code>tensorboard --logdir outputs/runs --port 6006</code>. '
                 'Run this from labs/lab02 in a second terminal while training is running. '
                 'Notebook figures do not start TensorBoard or generate training events.</p>'))


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
    table.to_csv(output / 'class_metrics.csv', index=False)
    fig, axes = plt.subplots(1, 4, figsize=(13, 5))
    for ax, column, label in zip(axes, ['precision', 'recall_tpr', 'f1', 'fpr'],
                                 ['Precision', 'Recall / TPR', 'F1', 'FPR']):
        matrix = fine.pivot(index='class', columns='model', values=column)
        upper = 100 if column != 'fpr' else max(5, float(matrix.max().max()*100))
        im = ax.imshow(matrix.to_numpy()*100, vmin=0, vmax=upper,
                       cmap='Blues' if column != 'fpr' else 'Reds', aspect='auto')
        ax.set(xticks=range(2), xticklabels=matrix.columns, yticks=range(10),
               yticklabels=matrix.index,
               title=f'{label} (%)\n' +
                   ('Lower is better' if column == 'fpr' else 'Higher is better'))
        ax.tick_params(axis='x', rotation=35, labelsize=8)
        for i in range(10):
            for j in range(2):
                value = matrix.iloc[i,j]*100
                ax.text(j, i, f'{value:.1f}', ha='center', va='center', fontsize=8,
                        color='white' if value > upper*.55 else 'black')
        fig.colorbar(im, ax=ax, fraction=.05, pad=.04)
    _save(fig, output, 'class_metric_heatmaps')
    display(fine.sort_values('f1').groupby('model', sort=False).head(2).round(4))
    display(HTML('FPR uses false positives / actual negatives, not 1 - precision. '
                 'Weak class metrics identify where to investigate, not the visual cause of errors.'))


def evidence(run: Path, kind: str) -> None:
    """Display count-only confusion labels or one prediction grid per row."""
    if kind == 'prediction_examples':
        fig, axes = plt.subplots(2, 1, figsize=(13, 16))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, name in zip(axes, ['resnet18', 'densenet121']):
        if kind == 'confusion_matrix':
            counts = pd.read_csv(run/name/'finetune/confusion_counts.csv', index_col=0)
            values = counts.to_numpy()
            fractions = values/np.maximum(values.sum(1, keepdims=True), 1)
            ax.imshow(fractions, vmin=0, vmax=1, cmap='Blues')
            for i in range(len(counts)):
                for j in range(len(counts)):
                    ax.text(j, i, str(values[i,j]), ha='center', va='center',
                            fontsize=8, color='white' if fractions[i,j] > .55 else 'black')
            ax.set(xticks=range(len(counts)), xticklabels=counts.columns,
                   yticks=range(len(counts)), yticklabels=counts.index, xlabel='Predicted', ylabel='Actual',
                   title=f'{name}: image counts')
            ax.tick_params(axis='x', rotation=45, labelsize=8)
        else:
            ax.imshow(plt.imread(run / name / 'finetune' / f'{kind}.png'))
            ax.axis('off')
            ax.set_title(name)
    plt.tight_layout()
    display(fig)
    plt.close(fig)
    if kind == 'confusion_matrix':
        records = []
        for name in ['resnet18', 'densenet121']:
            counts = pd.read_csv(run/name/'finetune/confusion_counts.csv', index_col=0)
            wrong = counts.to_numpy().copy()
            np.fill_diagonal(wrong, 0)
            for position in np.argsort(wrong, axis=None)[-3:][::-1]:
                i, j = np.unravel_index(position, wrong.shape)
                records.append({'model': name, 'actual': counts.index[i], 'predicted': counts.columns[j],
                                'count': wrong[i,j], 'within_actual_class_pct': 100*wrong[i,j]/counts.iloc[i].sum()})
        display(pd.DataFrame(records))
    else:
        display(HTML('Up to 8 correct and 8 incorrect samples per model; this selected grid does not estimate accuracy.'))


def roc_pr(run: Path) -> None:
    """Present existing fine-tune curve evidence at readable width, without predictions."""
    from IPython.display import Image
    rows = []
    for name in pd.read_csv(run/'comparison.csv').model.unique():
        folder = run/name/'finetune'
        picture, scores = folder/'roc_pr.png', folder/'roc_pr_metrics.csv'
        if not picture.exists() or not scores.exists():
            display(HTML(f'<p>{name}: saved ROC/PR evidence unavailable; no curves reconstructed.</p>'))
            continue
        display(Image(filename=str(picture), width=1200))
        frame = pd.read_csv(scores)
        rows.append({'model': name, 'classes': len(frame),
                     'Macro ROC AUC (OvR)': frame.roc_auc_ovr.mean(),
                     'Macro average precision (OvR)': frame.average_precision_ovr.mean()})
    if rows:
        display(pd.DataFrame(rows).round(4))
    display(HTML('<p>ROC: X = FPR, Y = TPR. PR: X = recall, Y = precision. '
                 'Each curve varies the score threshold for one class versus the rest. '
                 'ROC legend numbers are AUC; PR legend numbers are average precision (AP), not F1. '
                 'Macro scores above average the saved per-class scores; no new aggregate curve is inferred. '
                 'The argmax metrics above are not a fixed 0.5 threshold on these curves. '
                 'Saved figures do not establish fresh inference or calibration.</p>'))


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
    from ..training.costs import cost_table, validation_recommendation
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
    fig.suptitle('Timing comparison requires matching execution policies', fontsize=11)
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
        records = []
        for row in table.itertuples():
            selection = json.loads((run/row.model/'selection.json').read_text(encoding='utf-8'))
            records.append({'Model': row.model, 'Trial': row.trial,
                            'Backbone LR': row.backbone_lr, 'Head LR': row.head_lr,
                            'Batch': selection['head']['batch_size'], 'Unfreeze': row.unfreeze,
                            'Best epoch': row.best_epoch, 'Val accuracy (%)': row.best_val_accuracy * 100,
                            'Val loss': row.best_val_loss, 'Fit (min)': row.train_seconds / 60,
                            'Selected': 'Yes' if row.trial == selection['finetune']['trial'] else 'No'})
        _search_table(pd.DataFrame(records), 'Fine-tune: measured candidates; selection is within each model')
        display(HTML('<p>Batch is inherited from the selected head. Loss is measured at the selected checkpoint. '
                     'Fit time excludes head search and feature extraction. A single candidate per model does not establish an LR trend.</p>'))
    else:
        cfg=json.loads((root/'config/tuned.json').read_text(encoding='utf-8'))
        table=pd.DataFrame(cfg['finetune_candidates'])
        rows=[]
        for (optimizer, scheduler, depth), part in table.groupby(['optimizer','scheduler','unfreeze'],sort=False):
            rows.append({'Optimizer':optimizer, 'Backbone LR':', '.join(f'{v:g}' for v in sorted(part.backbone_lr.unique())),
                'Head LR': '10 x backbone LR',
                'Opened blocks':'last block' if depth=='last_block' else 'last two blocks',
                'Scheduler':scheduler, 'Candidates':len(part), 'Status':'NOT RUN'})
        display(pd.DataFrame(rows))
        display(pd.DataFrame([{'Head search batches':str(cfg['batch_sizes']),
            'Head epochs':cfg['head_epochs'], 'Fine-tune max epochs':cfg['finetune_epochs'],
            'Fine-tune minimum':cfg['early_stopping']['min_epochs'],
            'Stop rule':'validation loss: patience 7, min_delta 0.0001'}]))
        return

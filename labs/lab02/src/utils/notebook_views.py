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
        root = Path(configured).expanduser() if configured else Path(__file__).resolve().parents[2]
        if not root.is_absolute():
            raise RuntimeError('Set LAB02_ROOT to an existing absolute Lab02 directory.')
        root = root.resolve()
    except OSError as exc:
        raise RuntimeError('Cannot resolve Lab02. Set LAB02_ROOT to an existing absolute Lab02 directory.') from exc
    if not (root / 'src/utils/notebook_views.py').is_file():
        raise RuntimeError(f'Invalid Lab02 directory: {root}. Set LAB02_ROOT to an existing absolute Lab02 directory.')
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
    from .costs import cost_table
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
    """Display the measured epoch budget for each recorded stage."""
    recorded = json.loads((run/'config.json').read_text(encoding='utf-8'))
    rows = []
    for name in pd.read_csv(run/'comparison.csv').model.unique():
        for stage, folder, key in [('Head', 'head_selected', 'head_epochs'), ('Fine-tune', 'finetune', 'finetune_epochs')]:
            history = pd.read_csv(run/name/folder/'history.csv')
            rows.append({'Model':name, 'Stage':stage, 'Recorded epochs':len(history),
                'Configured budget':recorded[key]})
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
    display(HTML('<p>ResNet18 uses residual addition to learn corrections to a block input; '
        'DenseNet121 concatenates earlier feature maps, encouraging feature reuse. '
        'They provide two connectivity patterns for this transfer-learning exercise; '
        'fewer parameters do not guarantee lower latency. Both use explicit ImageNet weights '
        'and a replacement 10-class head. These choices are experimental, not proof of optimality.</p>'))
    for name in table.model.unique():
        arch_path = run / name / 'architecture.txt'
        if arch_path.exists():
            content = html.escape(arch_path.read_text(encoding='utf-8'))
            display(HTML(f'<details><summary>{name}: full saved architecture</summary>'
                         f'<pre style="max-height:260px;overflow:auto">{content}</pre></details>'))


def training_benchmark(run: Path, evidence: Path, output: Path, verify=True) -> None:
    """Read explicit training evidence without importing torch or running training."""
    from .delivery import file_hash
    evidence = Path(evidence)
    if verify:
        status = json.loads((evidence/'status.json').read_text())
        if status.get('status') != 'completed':
            raise ValueError('Training benchmark evidence is incomplete')
        manifest = json.loads((evidence/'manifest.json').read_text())
        required = {'config.json', 'methodology.json', 'training_performance.csv', 'provenance.json',
                    'resnet18/measurement.json', 'densenet121/measurement.json', 'status.json'}
        if not required <= manifest.keys():
            raise ValueError('Training benchmark manifest is incomplete')
        for name, checksum in manifest.items():
            path = (evidence/name).resolve()
            if not path.is_relative_to(evidence.resolve()) or file_hash(path) != checksum:
                raise ValueError('Training benchmark checksum mismatch: '+name)
        provenance = json.loads((evidence/'provenance.json').read_text())
        if provenance['training_run'] != run.name:
            raise ValueError('Training benchmark belongs to another training run')
        for name, checksum in provenance['checkpoints'].items():
            path = run/name/'finetune/best.pt'
            if path.exists() and file_hash(path) != checksum:
                raise ValueError('Training checkpoint differs from benchmark source')
    table = pd.read_csv(evidence/'training_performance.csv')
    measurements = [json.loads((evidence/name/'measurement.json').read_text()) for name in table.model]
    table.insert(5, 'baseline_rss_mib', [v['memory']['baseline_rss_bytes']/2**20 for v in measurements])
    table['peak_reserved_mib'] = [None if v['memory']['peak_reserved_bytes'] is None else
                                 v['memory']['peak_reserved_bytes']/2**20 for v in measurements]
    display(table.style.hide(axis='index').format(precision=4, na_rep='Unavailable'))
    fields = [('median_ms', 'Training step (ms/sample)'), ('peak_rss_mib', 'Process peak RSS (MiB)'),
              ('rss_increase_mib', 'RSS increase (MiB)')]
    if table.peak_allocated_mib.notna().any():
        fields.append(('peak_allocated_mib', 'CUDA allocated peak (MiB)'))
    fig, axes = plt.subplots(1, len(fields), figsize=(5*len(fields), 4), squeeze=False)
    for ax, (field, title) in zip(axes[0], fields):
        bars = ax.bar(table.model, table[field], color=['#2563eb', '#d97706'])
        ax.bar_label(bars, fmt='%.3f')
        ax.set(title=title, ylim=(0, max(table[field].max()*1.2, .001)))
    fig.suptitle('Training only | batch 1 | fixed train image | fresh optimizer | sampled process RSS')
    fig.tight_layout()
    _save(fig, output, 'training_resources')
    display(HTML('<p>Separate from inference and historical fit time. CPU VRAM is unavailable. '
                 'RSS includes Python/runtime; sampled peaks may miss short allocations. '
                 'No augmentation; freeze/BatchNorm follow checkpoint metadata.</p>'))


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


def single_image_benchmark(run: Path, benchmark: Path, output: Path) -> dict:
    """Display an explicit, verified loaded-model benchmark without running inference."""
    from .delivery import file_hash
    provenance = json.loads((benchmark/'provenance.json').read_text(encoding='utf-8'))
    if provenance['training_run'] != run.name:
        raise ValueError('Benchmark training run differs from the selected modeling run')
    models = ('resnet18', 'densenet121')
    if set(provenance['benchmark_runs']) != set(models):
        raise ValueError('Expected one benchmark for each supported model')
    if file_hash(benchmark/'input.png') != provenance['input_sha256']:
        raise ValueError('Benchmark input checksum mismatch')
    records = []
    for name in models:
        folder = benchmark/name
        manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        required = {'summary.json', 'config.json', 'environment_snapshot.json',
                    'status.json', 'latency_samples.csv', 'stage_samples.csv'}
        if not required <= manifest.keys():
            raise ValueError('Incomplete benchmark manifest')
        for filename, digest in manifest.items():
            if Path(filename).name != filename:
                raise ValueError('Unsafe benchmark manifest path')
            if file_hash(folder/filename) != digest:
                raise ValueError('Benchmark evidence checksum mismatch: '+filename)
        read = lambda filename: json.loads((folder/filename).read_text(encoding='utf-8'))
        status, cfg, result, env = (read(p) for p in
            ('status.json', 'config.json', 'summary.json', 'environment_snapshot.json'))
        if status['status'] != 'completed' or status['kind'] != 'single_image_benchmark':
            raise ValueError('Only completed single-image benchmarks can be displayed')
        if (result['model'] != name or cfg['model'] != name or
                cfg['image_sha256'] != provenance['input_sha256'] or
                cfg['model_load_in_timing'] or result['batch_size'] != 1):
            raise ValueError('Benchmark input or loaded-model protocol mismatch')
        records.append((name, cfg, result, env))
    latency, first, ram, vram, stages, environment = [], [], [], [], [], []
    mib = lambda value: None if value is None else value/2**20
    for name, cfg, result, env in records:
        memory, timing = result['memory'], result['latency_ms']
        latency.append({'Model':name, 'Device':env['device_name'], 'Requests':timing['count'],
                        'Mean (ms/image)':timing['mean'], 'Median (ms/image)':timing['median'],
                        'P95 (ms/image)':timing['p95']})
        first.append({'Model':name, 'First request (ms)':result['first_request_ms']})
        ram.append({'Model':name, 'RSS after load (MiB)':mib(memory['after_load']['rss_bytes']),
                    'RSS before memory pass (MiB)':mib(memory['before_pass']['rss_bytes']),
                    'RSS peak (MiB)':mib(memory['peak_rss_bytes']),
                    'RSS increase (MiB)':mib(memory['rss_increase_bytes'])})
        vram.append({'Model':name, 'Allocated peak (MiB)':mib(memory['peak_allocated_bytes']),
                     'Reserved peak (MiB)':mib(memory['peak_reserved_bytes']),
                     'Allocated increase (MiB)':mib(memory['allocated_increase_bytes'])})
        stages.extend({'Model':name, 'Stage':stage, 'Mean (ms)':values['mean'],
                       'Median (ms)':values['median'], 'P95 (ms)':values['p95']}
                      for stage, values in result['stages_ms'].items())
        environment.append({'Model':name, 'Training run':provenance['training_run'],
            'Benchmark run':provenance['benchmark_runs'][name], 'Device':env['device_name'],
            'Torch':env['torch'], 'CUDA':env['cuda'], 'Image size':result['image_size'],
            'Batch':result['batch_size'], 'Precision':result['precision'], 'Warm-up':cfg['warmup']})
    tables = {key:pd.DataFrame(rows) for key, rows in
              [('latency',latency), ('first',first), ('ram',ram), ('vram',vram),
               ('stages',stages), ('environment',environment)]}
    captions = {'latency':'Warmed end-to-end latency', 'first':'First request after model load',
                'ram':'Process RAM (RSS)', 'vram':'PyTorch CUDA allocator memory',
                'stages':'Separate diagnostic stage pass', 'environment':'Recorded execution context'}
    for key, table in tables.items():
        display(HTML('<h4>'+captions[key]+'</h4>'))
        display(table.round(4))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    tables['latency'].set_index('Model')[['Median (ms/image)', 'P95 (ms/image)']].plot.barh(ax=axes[0])
    tables['ram'].set_index('Model')[['RSS after load (MiB)', 'RSS peak (MiB)']].plot.barh(ax=axes[1])
    gpu = tables['vram'].set_index('Model')[['Allocated peak (MiB)', 'Reserved peak (MiB)']]
    if gpu.notna().any().any():
        gpu.plot.barh(ax=axes[2])
    else:
        axes[2].text(.1, .5, 'CUDA memory: not applicable', transform=axes[2].transAxes)
    for ax, title, unit in zip(axes, ['Warmed request', 'Process RAM', 'CUDA allocator'],
                               ['ms/image', 'MiB', 'MiB']):
        ax.set(title=title, xlabel=unit, ylabel='')
        if ax.get_legend() is not None:
            ax.legend(fontsize=7)
    _save(fig, output, 'single_image_benchmark')
    display(HTML('<p>Model load is excluded. First requests are separate from warmed statistics. '
        'Stages and memory use separate passes; do not sum stage timing to infer total latency. '
        'RSS is sampled process memory; CUDA allocated/reserved exclude context and other allocators. '
        'These measurements do not establish minimum hardware requirements.</p>'))
    return tables


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

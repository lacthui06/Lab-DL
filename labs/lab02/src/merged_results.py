"""Inline reporting for the three completed Kaggle groups; never trains models."""
import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

from .notebook_views import _save
from .tuning import select_trial
from .delivery import resolve_run, write_json


def collect(root):
    """Đọc đúng các run đã đăng ký; từ chối nhóm thiếu hoặc policy khác nhau."""
    root = Path(root)
    registry = json.loads((root/'outputs/selected_run.json').read_text(encoding='utf-8'))
    rows = []
    runs = []
    policies = []
    for group, path in registry['group_runs'].items():
        run = resolve_run(root, path)
        cfg = json.loads((run/'config.json').read_text(encoding='utf-8'))
        if json.loads((run/'status.json').read_text())['status'] != 'completed':
            raise ValueError(f'Run chưa hoàn tất: {run}')
        policies.append(cfg.get('selection_policy', {}))
        runs.append(run)
        for model in cfg['models']:
            frame = pd.read_csv(run/model/'finetune_search_results.csv')
            frame['model'], frame['group'], frame['run'] = model, int(group), str(run)
            rows.append(frame)
    grid = pd.concat(rows, ignore_index=True)
    if any(p != policies[0] for p in policies) or grid.duplicated(['model', 'id']).any():
        raise ValueError('Policy giữa nhóm khác nhau hoặc candidate ID bị trùng')
    if len(grid) != registry['completed_fine_trials']:
        raise ValueError('Số trial không khớp registry')
    return runs, grid


def _winners(root):
    runs, grid = collect(root)
    policy = json.loads((runs[0]/'config.json').read_text())['selection_policy']
    winners = {model: select_trial(frame.to_dict('records'), policy)
               for model, frame in grid.groupby('model', sort=False)}
    for model, winner in winners.items():
        selection = json.loads((Path(winner['run'])/model/'selection.json').read_text())
        if selection['finetune']['id'] != winner['id']:
            raise ValueError(f'{model}: winner chưa có test export; cần đánh giá checkpoint đã chọn')
    return winners


def context():
    """Tôn trọng LAB02_RUN; chỉ gộp mặc định khi không yêu cầu run riêng."""
    from . import notebook_views as view
    root, run, output = view.context()
    if os.getenv('LAB02_RUN'):
        return root, run, output
    winners = _winners(root)
    write_json(output/'global_selection.json', winners)
    paths = {Path(w['run']) for w in winners.values()}
    if len(paths) == 1:
        return root, paths.pop(), output
    # Nếu hai model thắng ở hai nhóm khác nhau, tạo view nhẹ có provenance rõ ràng.
    combined = output/'selected_report'
    combined.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, winner in winners.items():
        source = Path(winner['run'])
        rows.append(pd.read_csv(source/'comparison.csv').query('model == @model'))
        for file in (source/model).rglob('*'):
            if file.is_file() and file.suffix in {'.csv', '.json', '.txt', '.png'}:
                destination = combined/model/file.relative_to(source/model)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, destination)
    pd.concat(rows).to_csv(combined/'comparison.csv', index=False)
    shutil.copy2(Path(next(iter(winners.values()))['run'])/'config.json', combined/'config.json')
    write_json(combined/'provenance.json', winners)
    return root, combined, output


def grid_results(root, output, run=None):
    if os.getenv('LAB02_RUN'):
        from .notebook_views import fine_search
        return fine_search(run, output, root)
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


def summary(root, output, run=None):
    if os.getenv('LAB02_RUN'):
        from .notebook_views import quality_cost, efficiency
        quality_cost(run, output)
        efficiency(run, output)
        return
    _, grid = collect(root)
    result = pd.concat([pd.read_csv(Path(w['run'])/'comparison.csv').query('model == @model')
                        for model, w in _winners(root).items()], ignore_index=True)
    display(result[['model', 'stage', 'accuracy', 'loss', 'macro_f1',
                    'macro_precision', 'macro_recall', 'best_val_accuracy']].round(5))
    fine = result[result.stage == 'finetune'].copy()
    fine['selected_fit_minutes'] = fine.train_seconds/60
    fine['all_fine_trials_hours'] = fine.model.map(grid.groupby('model').train_seconds.sum())/3600
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, metric, scale, label in zip(axes, ['accuracy', 'loss', 'train_seconds'],
                                       [100, 1, 1/60], ['Test accuracy (%)', 'Test loss', 'Selected fit (minutes)']):
        bars = ax.bar(fine.model, fine[metric]*scale, color=['#2563eb', '#059669'])
        ax.bar_label(bars, fmt='%.3f')
        ax.set(title=label, ylim=(0, max(fine[metric]*scale)*1.18))
    _save(fig, output, 'gpu_quality_time')
    display(fine[['model', 'selected_fit_minutes', 'all_fine_trials_hours']].round(3))


def explain(run, section):
    """Nhận xét lấy từ run đang xem, không tái dùng số liệu viết cứng của GPU."""
    run = Path(run)
    cfg = json.loads((run/'config.json').read_text(encoding='utf-8'))
    table = pd.read_csv(run/'comparison.csv')
    messages = []
    if section == 'intro':
        registry = json.loads((Path(__file__).resolve().parents[1]/'outputs/selected_run.json').read_text())
        messages = [f'Run đang xem: `{run.name}`. Ảnh {cfg["image_size"]}×{cfg["image_size"]}, '
                    f'seed {cfg["seed"]}. ' + ('Phạm vi riêng theo LAB02_RUN.' if os.getenv('LAB02_RUN')
                    else f'Gộp các nhóm trong registry; {registry["completed_fine_trials"]} fine-trial.'),
                    'Chọn bằng validation accuracy/loss/time; test chỉ để báo cáo sau lựa chọn. '
                    'Thời gian fit trial được chọn khác tổng chi phí search.']
    elif section == 'quality':
        for model, part in table.groupby('model', sort=False):
            head = part[part.stage=='head'].iloc[0]
            fine = part[part.stage=='finetune'].iloc[0]
            messages.append(f'**{model}**: test accuracy {fine.accuracy:.2%}, loss {fine.loss:.4f}, '
                f'Macro F1 {fine.macro_f1:.4f}; tăng {(fine.accuracy-head.accuracy)*100:.2f} điểm % '
                f'so với head; fit fine-tune {fine.train_seconds/60:.2f} phút.')
        messages.append('Một seed chưa đủ khẳng định ưu thế thống kê. Không so throughput head cache với fine-tune ảnh đầy đủ.')
    elif section == 'training':
        for model in table.model.unique():
            h = pd.read_csv(run/model/'finetune/history.csv')
            s = json.loads((run/model/'finetune/training_summary.json').read_text())
            messages.append(f'**{model}**: chạy {len(h)}/{cfg["finetune_epochs"]} epoch, '
                f'checkpoint epoch {s["best_epoch"]}; stop: `{s.get("stop_reason", "not logged")}`. '
                f'Train/validation accuracy cuối {h.train_accuracy.iloc[-1]:.2%}/{h.val_accuracy.iloc[-1]:.2%}; '
                f'val loss {h.val_loss.iloc[0]:.4f} → {h.val_loss.iloc[-1]:.4f}. '
                + ('Có dấu hiệu overfit: train loss giảm nhưng val loss tăng.'
                   if h.train_loss.iloc[-1]<h.train_loss.iloc[0] and h.val_loss.iloc[-1]>h.val_loss.iloc[0]
                   else 'Số epoch riêng lẻ chưa chứng minh hội tụ; cần đọc cùng đường cong.'))
    elif section == 'errors':
        for model in table.model.unique():
            counts = pd.read_csv(run/model/'finetune/confusion_counts.csv', index_col=0)
            messages.append(f'**{model}**: cat → dog {counts.loc["cat","dog"]} ảnh; '
                            f'dog → cat {counts.loc["dog","cat"]} ảnh. '
                            f'Recall cat {counts.loc["cat","cat"]/counts.loc["cat"].sum():.2%}.')
    elif section == 'predictions':
        for model in table.model.unique():
            path = run/model/'finetune/prediction_examples.csv'
            if path.exists():
                from .eda import CLASSES
                wrong = pd.read_csv(path).query('true != predicted').nlargest(2, 'probability')
                messages.extend(f'{model}: {CLASSES[int(r.true)]} → {CLASSES[int(r.predicted)]}, '
                                f'p={r.probability:.3f}.' for r in wrong.itertuples())
        messages.append('Lưới tối đa 8 đúng và 8 sai không đại diện accuracy; confidence cao không bảo đảm đúng.')
    elif section == 'conclusion':
        from .costs import validation_recommendation
        choice = validation_recommendation(run, table, cfg)
        messages = [f'Khuyến nghị theo validation: **{choice["model"]}**. '
                    'Đọc chất lượng cùng thời gian, bộ nhớ và execution policy. '
                    'Augmentation/dropout là hướng thử tiếp, chưa được đo trong các kết quả này.']
    else:
        raise ValueError(f'Unknown section: {section}')
    display(Markdown('\n\n'.join(messages)))

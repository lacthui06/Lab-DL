"""Thí nghiệm 10 epoch: chọn cấu hình trên validation rồi mới đánh giá test."""
from copy import deepcopy
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, brier_score_loss
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from .workflow import read_config, seed_all, finetune, evaluate_model
from .datasets.loader import load_data, tokenize_data
from .models.classifier import load_model
from .training.metrics import compute_metrics
from .training.thresholds import select_validation_threshold, binary_metrics_at_threshold
from .utils.runs import ROOT, run_context, save_json, hash_file
from .utils.plots import finish


def choose_candidate(rows):
    """Quy tắc công bố trước; không nhận hoặc đọc bất cứ metric test nào."""
    if not rows or any(r.get('selection_split') != 'validation' for r in rows):
        raise ValueError('Candidate selection requires validation-only evidence')
    for row in rows:
        if not np.isfinite([row['macro_f1'], row['loss'], row['learning_rate']]).all():
            raise ValueError('Nonfinite candidate selection metric')
    return min(rows, key=lambda r: (-r['macro_f1'], r['loss'], r['learning_rate']))


def fit_text_baseline(train, validation, settings, seed):
    """Fit từ vựng/IDF/classifier chỉ trên train, chọn C bằng validation."""
    candidates = []
    best_model = None
    best_key = None
    for C in settings['C_values']:
        model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=tuple(settings['ngram_range']),
                min_df=settings['min_df'], max_features=settings['max_features'], sublinear_tf=True)),
            ('classifier', LogisticRegression(C=C, max_iter=2000, solver='liblinear', random_state=seed)),
        ])
        model.fit(train['text'], train['label'])
        probabilities = model.predict_proba(validation['text'])
        metrics = compute_metrics((np.log(np.clip(probabilities, 1e-12, 1)), validation['label']))
        row = {'C': C, **metrics, 'loss': float(log_loss(validation['label'], probabilities)), 'selection_split': 'validation',
               'validation_probabilities': probabilities.tolist(), 'validation_labels': list(map(int, validation['label']))}
        candidates.append(row)
        key = (-row['macro_f1'], row['loss'], C)
        if best_key is None or key < best_key:
            best_key, best_model = key, model
    policy = select_validation_threshold(validation['label'], best_model.predict_proba(validation['text'])[:, 1])
    return best_model, candidates, policy


def audit_split_texts(raw):
    """Đếm trùng nội bộ/giao nhau theo exact và casefold-whitespace; không đổi split."""
    result = {}
    for mode, normalize in [('exact', lambda s: s), ('casefold_whitespace', lambda s: ' '.join(s.casefold().split()))]:
        sets = {split: set(map(normalize, raw[split]['text'])) for split in ('train', 'validation', 'test')}
        result[mode] = {'within_split_duplicates': {s: len(raw[s]) - len(sets[s]) for s in sets},
                       'cross_split_overlaps': {f'{a}__{b}': len(sets[a] & sets[b])
                           for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]}}
    return result


def prediction_diagnostics(labels, probabilities, out, threshold, seed=42, repeats=2000):
    """Chẩn đoán mô tả sau khi khóa policy, không tìm threshold tối ưu trên test."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    from .analysis import _reliability
    labels = np.asarray(labels); probabilities = np.asarray(probabilities)
    p = probabilities[:, 1]
    pred = (p >= threshold).astype(int)
    bins, ece = _reliability(labels, probabilities)
    metrics = {'roc_auc': float(roc_auc_score(labels, p)),
               'average_precision': float(average_precision_score(labels, p)),
               'brier_score': float(brier_score_loss(labels, p)), 'confidence_ece': ece,
               'nll': float(log_loss(labels, probabilities / probabilities.sum(axis=1, keepdims=True))),
               'threshold': threshold, 'selection_split': 'validation'}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repeats):
        # Bootstrap phân tầng giữ support hai lớp, không dùng CI để đổi policy.
        idx = np.concatenate([rng.choice(np.flatnonzero(labels == c), np.sum(labels == c), replace=True) for c in (0, 1)])
        m = binary_metrics_at_threshold(labels[idx], p[idx], threshold)
        samples.append([m['accuracy'], m['macro_f1']])
    metrics['bootstrap_95_percentile_ci'] = {k: np.quantile(np.array(samples)[:, i], [.025, .975]).tolist()
                                            for i, k in enumerate(['accuracy', 'macro_f1'])}
    metrics['bootstrap_repeats'] = repeats
    metrics['ci_scope'] = 'fixed model, fixed threshold, stratified test sampling; excludes training-seed uncertainty'
    save_json(out / 'probability_diagnostics.json', metrics)
    save_json(out / 'reliability_bins.json', bins)
    fpr, tpr, _ = roc_curve(labels, p)
    precision, recall, _ = precision_recall_curve(labels, p)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(fpr, tpr, label=f"ROC AUC={metrics['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], '--', color='gray'); axes[0].set(xlabel='False positive rate', ylabel='True positive rate')
    axes[1].plot(recall, precision, label=f"Average precision={metrics['average_precision']:.3f}")
    axes[1].axhline(labels.mean(), ls='--', color='gray', label='Class prevalence')
    axes[1].set(xlabel='Recall', ylabel='Precision')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    finish(fig, out / 'roc_pr_curves.png')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    nonempty = [b for b in bins if b.get('count', 0)]
    # Tên trường lấy từ hợp đồng chung của analysis._reliability.
    axes[0].plot([b['mean_confidence'] for b in nonempty], [b['accuracy'] for b in nonempty], 'o-', label=f'ECE={ece:.3f}')
    axes[0].plot([.5, 1], [.5, 1], '--', color='gray'); axes[0].set(xlabel='Mean confidence', ylabel='Observed accuracy')
    for c in (0, 1): axes[1].hist(p[labels == c], bins=np.linspace(0, 1, 21), alpha=.55, label=f'True class {c}')
    axes[1].axvline(threshold, ls='--', color='black', label=f'Frozen threshold={threshold:g}')
    axes[1].set(xlabel='P(POSITIVE)', ylabel='Test examples')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    finish(fig, out / 'calibration_confidence.png')
    errors = [{'index': int(i), 'true_label': int(labels[i]), 'predicted_label': int(pred[i]),
               'positive_probability': float(p[i]), 'decision_confidence': float(p[i] if pred[i] else 1-p[i])}
              for i in np.flatnonzero(pred != labels)]
    save_json(out / 'threshold_errors.json', errors)
    return metrics


def _study_figures(rows, comparison, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for row in rows:
        history = json.loads((ROOT / 'outputs/runs' / row['run_id'] / 'trainer_history.json').read_text())
        points = [r for r in history if 'eval_macro_f1' in r]
        axes[0].plot([r['step'] for r in points], [r['eval_macro_f1'] for r in points], label=f"LR={row['learning_rate']:g}")
        axes[1].plot([r['step'] for r in points], [r['eval_loss'] for r in points], label=f"LR={row['learning_rate']:g}")
    for ax in axes: ax.set_xlabel('Optimizer step'); ax.legend(); ax.grid(alpha=.2)
    axes[0].set_ylabel('Validation macro F1'); axes[1].set_ylabel('Validation cross-entropy')
    finish(fig, out / 'candidate_validation_comparison.png')
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(comparison))
    for offset, key in [(-.24, 'accuracy'), (0, 'macro_f1'), (.24, 'binary_f1')]:
        bars = ax.bar(x + offset, [r[key] for r in comparison], width=.24, label=key)
        ax.bar_label(bars, fmt='%.3f', fontsize=8)
    ax.set_xticks(x, [r['model'] for r in comparison], rotation=10)
    ax.set(ylim=(0, 1.1), ylabel='Test metric (descriptive only)'); ax.legend()
    finish(fig, out / 'baseline_model_comparison.png')
    stats = json.loads((ROOT / 'outputs/runs' / rows[0]['run_id'] / 'data_statistics.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (split, counts) in zip(axes, stats['label_counts'].items()):
        bars = ax.bar(['Negative', 'Positive'], [counts['0'], counts['1']])
        ax.bar_label(bars); ax.set(title=split, ylabel='Examples')
    finish(fig, out / 'class_distribution.png')


def require_study_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError('Enable a Kaggle GPU before launching the 10-epoch study')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Use CUDA_VISIBLE_DEVICES=0: this study is designed for one GPU')


def run_study(config=None, settings=None):
    """Mỗi candidate đủ 10 epoch; test chỉ chạy sau khi khóa winner và threshold."""
    config = config or read_config()
    settings = settings or json.loads((ROOT / 'config/study.json').read_text())
    rates = settings['learning_rates']
    if not rates or len(rates) != len(set(rates)) or not all(np.isfinite(r) and r > 0 for r in rates):
        raise ValueError('Learning rates must be unique finite positive values')
    if settings.get('selection_metric', 'macro_f1') != 'macro_f1':
        raise ValueError('This study predeclares validation macro_f1 as selection metric')
    if settings['epochs_per_candidate'] != 10:
        raise ValueError('This study requires exactly 10 epochs per candidate')
    if settings['seed'] != config['seed']:
        raise ValueError('Study seed and training seed must agree')
    if config['exercise2'].get('metric_for_best_model') != 'macro_f1':
        raise ValueError('Study checkpoints must be selected by validation macro_f1')
    if not config['exercise2'].get('threshold_selection', {}).get('enabled', True):
        raise ValueError('Study requires validation threshold selection')
    if not isinstance(settings['bootstrap_repeats'], int) or settings['bootstrap_repeats'] < 1:
        raise ValueError('bootstrap_repeats must be a positive integer')
    require_study_gpu()
    seed_all(config['seed'])
    with run_context('study', {'training': config, 'study': settings}) as out:
        start = time.perf_counter()
        raw = load_data(config['exercise2'])
        save_json(out / 'split_overlap_audit.json', audit_split_texts(raw))
        baseline_train = raw['train']
        if config['exercise2'].get('augmentation', {}).get('enabled', False):
            from .datasets.augmentation import augment_train, export_augmentation
            augmented_raw, augmentation = augment_train(raw, config['exercise2']['augmentation'], config['seed'])
            export_augmentation(augmentation, out)
            baseline_train = augmented_raw['train']
        baseline, baseline_rows, baseline_policy = fit_text_baseline(baseline_train, raw['validation'], settings['baseline'], config['seed'])
        save_json(out / 'baseline_validation_candidates.json', baseline_rows)
        save_json(out / 'baseline_validation_policy.json', baseline_policy)
        candidates = []
        for rate in settings['learning_rates']:
            cfg = deepcopy(config)
            cfg['exercise2'].update(learning_rate=rate, num_train_epochs=10)
            candidate = finetune(cfg, evaluate_test=False)
            summary = json.loads((candidate / 'selection_summary.json').read_text())
            validations = [r for r in json.loads((candidate / 'trainer_history.json').read_text()) if 'eval_macro_f1' in r]
            best_step = int(Path(summary['best_checkpoint']).name.split('-')[-1])
            point = next(r for r in validations if r['step'] == best_step)
            candidates.append({'run_id': candidate.name, 'learning_rate': rate, 'selection_split': 'validation',
                'macro_f1': point['eval_macro_f1'], 'loss': point['eval_loss'],
                'completed_epochs': summary['completed_epochs'], 'best_step': best_step})
            save_json(out / 'candidate_comparison.json', candidates)
            gc.collect(); torch.cuda.empty_cache()
        winner = choose_candidate(candidates)
        source = ROOT / 'outputs/runs' / winner['run_id']
        from .decision import load_decision_policy
        policy = load_decision_policy(source / 'best_model', source / 'decision_threshold.json')
        # Ghi quyết định bất biến trước khi tạo dự đoán test.
        save_json(out / 'frozen_selection.json', {'winner': winner, 'threshold': policy['selected_threshold'],
                  'selection_split': 'validation', 'test_used_for_selection': False,
                  'candidate_manifest_sha256': hash_file(source / 'manifest.json'),
                  'prior_test_exposure': 'The historical 3-epoch test report was already inspected; this is not a new untouched benchmark.'})
        save_json(out / 'decision_threshold.json', policy)
        save_json(out / 'checkpoint_reference.json', {'run_id': source.name, 'relative_path': f'outputs/runs/{source.name}/best_model'})
        # Chỉ tải checkpoint đã chọn, không tải ngẫu nhiên backbone lần nữa.
        selected_config = json.loads((source / 'config.json').read_text())
        model, tokenizer = load_model(str(source / 'best_model'))
        tokenized = tokenize_data(raw, tokenizer, selected_config['exercise2']['max_length'])
        trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(out / 'temporary'),
            per_device_eval_batch_size=config['exercise2']['eval_batch_size'], report_to='none'),
            processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer))
        metrics, samples = evaluate_model(trainer, tokenized['test'], raw['test'], out, policy['selected_threshold'])
        from .workflow import report_run
        report_run(out, metrics, samples)
        test_rows = json.loads((out / 'all_predictions.json').read_text())
        probabilities = np.array([r['probabilities'] for r in test_rows])
        diagnostics = prediction_diagnostics(raw['test']['label'], probabilities, out, policy['selected_threshold'],
                               config['seed'], settings['bootstrap_repeats'])
        # Gắn text vào error records để report không phải lookup lại dữ liệu gốc.
        error_path = out / 'threshold_errors.json'
        errors = json.loads(error_path.read_text())
        for row in errors: row['text'] = raw['test'][row['index']]['text']
        save_json(error_path, errors)
        from .models.classifier import manual_predict
        inference_texts = ['I love this wonderful movie!', 'This movie was terrible and boring.',
                           'The acting was excellent, but the story was disappointing.']
        inference = manual_predict(trainer.model, tokenizer, inference_texts, selected_config['exercise2']['max_length'])
        for row in inference:
            p = row['probabilities'][1]
            row['argmax_label'] = row.pop('label')
            decision = int(p >= policy['selected_threshold'])
            row['label'] = 'POSITIVE' if decision else 'NEGATIVE'
            row['score'] = row['probabilities'][decision]
            row['decision_threshold'] = policy['selected_threshold']
            row['selection_split'] = 'validation'
        save_json(out / 'inference_predictions.json', inference)
        from .analysis import _table_image
        _table_image([{'Text': r['text'], 'Argmax': r['argmax_label'], 'Frozen policy': r['label'],
                       'P(POSITIVE)': f"{r['probabilities'][1]:.5f}"} for r in inference],
            ['Text', 'Argmax', 'Frozen policy', 'P(POSITIVE)'], out / 'inference_predictions.png', 'Inference: selected checkpoint and frozen validation policy')
        baseline_out = out / 'baseline'; baseline_out.mkdir()
        bp = baseline.predict_proba(raw['test']['text'])
        save_json(baseline_out / 'all_predictions.json', [{'index': i, 'true_label': int(y), 'probabilities': p.tolist()}
                  for i, (y, p) in enumerate(zip(raw['test']['label'], bp))])
        comparison = []
        for name, prob, threshold in [('TF-IDF + logistic regression (0.5)', bp, .5),
            ('TF-IDF + validation threshold', bp, baseline_policy['selected_threshold']),
            ('DistilBERT (0.5)', probabilities, .5), ('DistilBERT + validation threshold', probabilities, policy['selected_threshold'])]:
            comparison.append({'model': name, **binary_metrics_at_threshold(raw['test']['label'], prob[:, 1], threshold)})
        save_json(out / 'test_comparison.json', comparison)
        _study_figures(candidates, comparison, out)
        write_study_report(out, candidates, winner, comparison, diagnostics)
        save_json(out / 'study_summary.json', {'winner': winner, 'candidate_count': len(candidates),
            'epochs_per_candidate': 10, 'test_comparison': comparison, 'runtime_seconds': time.perf_counter()-start,
            'training_seeds': [config['seed']], 'seed_variability_measured': False,
            'statistical_significance_established': False,
            'note': 'Compare with historical evidence; no guaranteed gain from longer training or validation search.'})
    return out


def write_study_report(out, candidates, winner, comparison, diagnostics):
    """Xuất report có ảnh/bảng từ số liệu đã chạy, không cần dựng lại bằng agent."""
    lines = ['# Lab03 — Báo cáo thí nghiệm 10 epoch', '', f'Run: `{out.name}`', '',
        '## Thiết kế và lựa chọn', '',
        'Full official splits: 8.530 train / 1.066 validation / 1.066 test. Seed 42.',
        'Mỗi DistilBERT candidate chạy đủ 10 epoch, không early stopping. Checkpoint chọn bằng validation Macro-F1.',
        'TF-IDF và Logistic Regression fit chỉ trên train; C chọn bằng validation.',
        'Learning rate chọn bằng validation Macro-F1; hòa ưu tiên loss thấp hơn rồi LR nhỏ hơn.',
        'Threshold chốt từ 19 điểm validation trước test; không tối ưu threshold trên test.', '',
        '| Candidate | LR | Epoch hoàn tất | Best step | Val Macro-F1 | Val loss |',
        '|---|---:|---:|---:|---:|---:|']
    for r in candidates:
        lines.append(f"| {r['run_id']} | {r['learning_rate']:g} | {r['completed_epochs']:g} | {r['best_step']} | {r['macro_f1']:.5f} | {r['loss']:.5f} |")
    lines += ['', f"Winner: `{winner['run_id']}`.", '',
        '![Validation candidates](candidate_validation_comparison.png)', '',
        '## Test với các policy đã chốt', '',
        '| Model / policy | Accuracy | Precision (+) | Recall (+) | Macro-F1 | Binary-F1 | FP | FN |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in comparison:
        vals = ' | '.join(f"{r[k]:.4f}" for k in ['accuracy','precision','recall','macro_f1','binary_f1'])
        lines.append(f"| {r['model']} | {vals} | {r['FP']} | {r['FN']} |")
    lines += ['', '![Baseline comparison](baseline_model_comparison.png)', '',
        '## Xếp hạng, confidence và bất định', '',
        f"ROC-AUC: {diagnostics['roc_auc']:.5f}; AP: {diagnostics['average_precision']:.5f}; Brier: {diagnostics['brier_score']:.5f}; argmax-confidence ECE: {diagnostics['confidence_ece']:.5f}.",
        f"95% percentile bootstrap CI: `{diagnostics['bootstrap_95_percentile_ci']}`.",
        'CI phân tầng trên test cho model/policy cố định; không bao gồm bất định do seed train.', '',
        '![ROC/PR](roc_pr_curves.png)', '', '![Calibration](calibration_confidence.png)', '',
        '## Ma trận nhầm lẫn và mẫu lỗi', '',
        '![Argmax confusion](confusion_matrix.png)', '',
        '![Frozen threshold confusion](test_threshold/confusion_matrix.png)', '',
        '![Sample predictions](sample_predictions.png)', '',
        'Xem report.md cho mẫu lỗi argmax và threshold_errors.json cho toàn bộ lỗi của policy chốt.',
        'Giả thuyết ngôn ngữ không phải causal attribution. Không sửa policy dựa trên ví dụ test.', '',
        '## Giới hạn', '',
        'Test lịch sử 3 epoch từng được xem. Đây là mở rộng thí nghiệm, không phải benchmark hoàn toàn chưa đụng tới.',
        'Một seed không chứng minh cấu hình ổn định qua các lần khởi tạo. Ba learning rate không chứng minh tối ưu toàn cục.',
        'Chạy lâu hơn không đảm bảo test accuracy/F1 cao hơn. Báo cáo cả cải thiện và suy giảm.',
        'Nếu overlap audit có giao nhau giữa split, phải nêu rõ trong kết luận; không tự đổi official splits.', '',
        '## Artifact theo từng candidate', '']
    for row in candidates:
        rid = row['run_id']
        for name in ['learning_curves.png','step_learning_curves.png','validation_threshold/threshold_curves.png']:
            lines += [f'### {rid}: {name}', '', f'![{name}](../{rid}/{name})', '']
    (out / 'study_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

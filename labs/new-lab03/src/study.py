"""Baseline train-only và diagnostics xác suất cho phân loại nhị phân."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, brier_score_loss
from .data_preparation import make_tfidf
from .training.metrics import compute_metrics
from .training.thresholds import select_validation_threshold, binary_metrics_at_threshold
from .utils.runs import save_json
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
            ('tfidf', make_tfidf(settings)),
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

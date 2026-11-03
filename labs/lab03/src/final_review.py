"""Kiểm toán frozen trial và dựng báo cáo so sánh từ artifact đã đo, không train."""
from pathlib import Path
import numpy as np
from sklearn.metrics import log_loss
from .evidence import read_json, verify_manifest
from .training.thresholds import binary_metrics_at_threshold, select_validation_threshold


def audit_frozen_trial(run):
    run = Path(run)
    manifest = verify_manifest(run)
    cfg = read_json(run/'config.json')['exercise2']
    rows = read_json(run/'epoch_history.json')
    selection = read_json(run/'selection_summary.json')
    frozen = read_json(run/'frozen_selection.json')
    budget, completed = cfg['num_train_epochs'], len(rows)
    enabled = bool(cfg.get('early_stopping',{}).get('enabled',False))
    if (not 1 <= completed <= budget or [r['epoch'] for r in rows] != list(range(1,completed+1))
            or cfg['metric_for_best_model'] != 'loss' or selection['early_stopping'] != enabled
            or selection['completed_epochs'] != completed or selection.get('requested_epochs',budget) != budget
            or (not enabled and completed != budget)):
        raise ValueError('Epoch history differs from declared budget/early-stopping policy')
    best = min(rows, key=lambda r: r['validation_loss'])
    if best['epoch'] != selection['selected_epoch'] or best['step'] != selection['best_step']:
        raise ValueError('Checkpoint differs from minimum validation loss')
    if frozen['selected_epoch'] != best['epoch'] or frozen['test_used_for_selection']:
        raise ValueError('Invalid frozen selection')
    if frozen['weights_sha256'] != read_json(run/'manifest.json')['best_model/model.safetensors']:
        raise ValueError('Checkpoint identity mismatch')
    stats = read_json(run/'data_statistics.json')
    if stats['official_sizes'] != {'train':8530, 'validation':1066, 'test':1066}:
        raise ValueError('Official split contract mismatch')
    aug = read_json(run/'augmentation.json')
    if (stats['effective_sizes']['train'] != 8530 + aug['added_count']
            or stats['effective_sizes']['validation'] != 1066 or stats['effective_sizes']['test'] != 1066
            or aug['validation_and_test_modified'] or stats['validation_and_test_augmented']):
        raise ValueError('Invalid effective dataset sizes')
    import hashlib, json
    digest = hashlib.sha256(json.dumps(aug['records'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if digest != aug['records_sha256'] or len(aug['records']) != aug['added_count']:
        raise ValueError('Invalid augmentation evidence')
    freezing = read_json(run/'freezing_and_optimizer.json')
    if not freezing['frozen_embeddings'] or freezing['frozen_encoder_layers'] != cfg['freeze_layers']:
        raise ValueError('Freezing configuration mismatch')
    names = [n for g in freezing['optimizer_groups'] for n in g['parameter_names']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate optimizer parameters')
    for group in freezing['optimizer_groups']:
        for name in group['parameter_names']:
            if name.startswith('distilbert.embeddings.') or any(name.startswith(f'distilbert.transformer.layer.{i}.') for i in range(cfg['freeze_layers'])):
                raise ValueError('Frozen parameter present in optimizer')
            expected_lr = cfg['learning_rate'] if name.startswith('distilbert.') else cfg['head_learning_rate']
            if group['lr'] != expected_lr:
                raise ValueError('Optimizer learning-rate mismatch')
    validation = read_json(run/'validation_predictions.json')
    test = read_json(run/'test_predictions.json')
    for records in (validation, test):
        if [r['index'] for r in records] != list(range(1066)):
            raise ValueError('Prediction cohort missing or reordered')
        probabilities = np.array([r['probabilities'] for r in records])
        if (probabilities.shape != (1066, 2) or not np.isfinite(probabilities).all()
                or np.any((probabilities < 0) | (probabilities > 1))
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)):
            raise ValueError('Invalid prediction probabilities')
    yv = [r['true_label'] for r in validation]; pv = [r['probabilities'][1] for r in validation]
    reconstructed = select_validation_threshold(yv, pv)
    policy = read_json(run/'decision_threshold.json')
    comparable_policy = {k:v for k,v in policy.items() if k != 'checkpoint_identity'}
    if comparable_policy != reconstructed or frozen['threshold'] != policy['selected_threshold']:
        raise ValueError('Threshold differs from validation-only selection')
    if 'checkpoint_identity' in policy and policy['checkpoint_identity']['weights_sha256'] != frozen['weights_sha256']:
        raise ValueError('Policy checkpoint identity mismatch')
    vp = np.array([r['probabilities'] for r in validation]); vp /= vp.sum(axis=1, keepdims=True)
    if not np.isclose(log_loss(yv, vp), best['validation_loss'], atol=1e-4):
        raise ValueError('Validation predictions do not match selected checkpoint loss')
    comparison = read_json(run/'test_comparison.json')
    baseline = read_json(run/'baseline_test_predictions.json')
    if [(r['index'],r['text'],r['true_label']) for r in baseline] != [(r['index'],r['text'],r['true_label']) for r in test]:
        raise ValueError('Baseline and transformer test cohorts differ')
    for row, records in zip(comparison, [baseline, test, test]):
        calculated = binary_metrics_at_threshold([r['true_label'] for r in records],
            [r['probabilities'][1] for r in records], row['threshold'])
        if any(not np.isclose(row[k], v, atol=1e-10) for k,v in calculated.items()):
            raise ValueError('Test metrics differ from saved predictions')
    tp = np.array([r['probabilities'] for r in test]); tp /= tp.sum(axis=1, keepdims=True)
    nll = log_loss([r['true_label'] for r in test], tp)
    if not np.isclose(nll, read_json(run/'test_runtime_metrics.json')['test_loss'], atol=1e-4):
        raise ValueError('Test loss differs from probabilities')
    return {'status':'passed', 'run_id':run.name, 'manifest':manifest, 'selected_epoch':best['epoch'],
        'threshold':policy['selected_threshold'], 'test_samples':len(test),
        'weights_loaded_locally':False, 'threshold_reselected_on_test':False,
        'test_used_for_selection':False}

"""Kiểm toán study chỉ từ artifacts; không tải model, dataset hoặc chạy training."""
from pathlib import Path
import numpy as np
from .utils.runs import ROOT, hash_file
from .evidence import read_json, verify_manifest
from .study import choose_candidate
from .training.metrics import compute_metrics
from .training.thresholds import select_validation_threshold, binary_metrics_at_threshold


def _same_metrics(actual, expected):
    for key, value in expected.items():
        if key not in actual or not np.isclose(actual[key], value, atol=1e-10, rtol=1e-8):
            raise ValueError(f'Metric mismatch: {key}')


def audit_study(study, root=ROOT, expected_sizes=None):
    root = Path(root).resolve(); study = Path(study).resolve()
    expected_sizes = expected_sizes or {'train':8530, 'validation':1066, 'test':1066}
    runs = root / 'outputs/runs'
    if study.parent != runs: raise ValueError('Invalid study path')
    manifests = {study.name: verify_manifest(study)}
    config = read_json(study / 'config.json')
    settings = config['study']
    rows = read_json(study / 'candidate_comparison.json')
    if len(rows) != len(settings['learning_rates']) or sorted(r['learning_rate'] for r in rows) != sorted(settings['learning_rates']):
        raise ValueError('Candidate learning-rate budget differs from declared study')
    reconstructed = []
    for row in rows:
        run_id = row['run_id']
        if Path(run_id).name != run_id or run_id in {'.','..'}: raise ValueError('Unsafe candidate path')
        run = runs / run_id
        manifests[run_id] = verify_manifest(run)
        stats = read_json(run / 'data_statistics.json')
        if stats['split_sizes'] != expected_sizes: raise ValueError('Dataset contract differs')
        summary = read_json(run / 'selection_summary.json')
        if summary['requested_epochs'] != 10 or summary['completed_epochs'] != 10 or summary['early_stopping']:
            raise ValueError('Candidate did not complete exactly 10 epochs')
        if summary['test_evaluated'] or (run / 'all_predictions.json').exists():
            raise ValueError('Candidate must not evaluate test during selection')
        history = read_json(run / 'trainer_history.json')
        step = int(Path(summary['best_checkpoint']).name.split('-')[-1])
        point = next(p for p in history if p.get('step') == step and 'eval_macro_f1' in p)
        local_config = read_json(run / 'config.json')
        if local_config['seed'] != config['training']['seed']: raise ValueError('Candidate seed mismatch')
        reconstructed.append({'run_id':run_id, 'learning_rate':local_config['exercise2']['learning_rate'],
            'selection_split':'validation', 'macro_f1':point['eval_macro_f1'], 'loss':point['eval_loss'],
            'completed_epochs':summary['completed_epochs'], 'best_step':step})
        validation = read_json(run / 'validation_predictions.json')
        if [r['index'] for r in validation] != list(range(expected_sizes['validation'])):
            raise ValueError('Validation prediction indices missing/reordered')
        probabilities = np.array([r['probabilities'] for r in validation])
        labels = np.array([r['true_label'] for r in validation])
        selected = select_validation_threshold(labels, probabilities[:, 1])
        policy = read_json(run / 'decision_threshold.json')
        if selected['selected_threshold'] != policy['selected_threshold']:
            raise ValueError('Threshold differs from validation selection')
        _same_metrics(policy['selected_metrics'], selected['selected_metrics'])
        identity = policy['checkpoint_identity']
        expected_hash = read_json(run / 'manifest.json')['best_model/' + identity['weights_filename']]
        if expected_hash != identity['weights_sha256']: raise ValueError('Policy weights hash mismatch')
    if rows != reconstructed: raise ValueError('Candidate comparison differs from checkpoint history')
    winner = choose_candidate(reconstructed)
    frozen = read_json(study / 'frozen_selection.json')
    source = runs / winner['run_id']
    if frozen['winner'] != winner or frozen['test_used_for_selection'] or frozen['selection_split'] != 'validation':
        raise ValueError('Frozen selection differs from validation evidence')
    if frozen['candidate_manifest_sha256'] != hash_file(source / 'manifest.json'):
        raise ValueError('Frozen source manifest mismatch')
    policy = read_json(study / 'decision_threshold.json')
    if policy != read_json(source / 'decision_threshold.json') or frozen['threshold'] != policy['selected_threshold']:
        raise ValueError('Test policy differs from selected candidate policy')
    predictions = read_json(study / 'all_predictions.json')
    if [r['index'] for r in predictions] != list(range(expected_sizes['test'])):
        raise ValueError('Test prediction indices missing/reordered')
    p = np.array([r['probabilities'] for r in predictions]); y = np.array([r['true_label'] for r in predictions])
    if p.shape != (expected_sizes['test'], 2) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.allclose(p.sum(1), 1, atol=1e-6):
        raise ValueError('Invalid probability vectors')
    _same_metrics(read_json(study / 'test_metrics.json'), compute_metrics((np.log(np.clip(p, 1e-12, 1)), y)))
    _same_metrics(read_json(study / 'test_threshold_metrics.json'), binary_metrics_at_threshold(y, p[:,1], policy['selected_threshold']))
    baseline_rows = read_json(study / 'baseline_validation_candidates.json')
    for row in baseline_rows:
        bp = np.array(row['validation_probabilities']); by = np.array(row['validation_labels'])
        if len(by) != expected_sizes['validation']: raise ValueError('Baseline validation support differs')
        _same_metrics(row, compute_metrics((np.log(np.clip(bp, 1e-12, 1)), by)))
    baseline_best = min(baseline_rows, key=lambda r: (-r['macro_f1'], r['loss'], r['C']))
    baseline_policy = read_json(study / 'baseline_validation_policy.json')
    recomputed_policy = select_validation_threshold(baseline_best['validation_labels'], np.array(baseline_best['validation_probabilities'])[:,1])
    if baseline_policy != recomputed_policy: raise ValueError('Baseline threshold policy differs')
    baseline_test = read_json(study / 'baseline/all_predictions.json')
    if [r['index'] for r in baseline_test] != list(range(expected_sizes['test'])) or [r['true_label'] for r in baseline_test] != y.tolist():
        raise ValueError('Baseline test cohort differs')
    bp = np.array([r['probabilities'] for r in baseline_test])
    comparisons = read_json(study / 'test_comparison.json')
    expected = [(bp, .5), (bp, baseline_policy['selected_threshold']), (p, .5), (p, policy['selected_threshold'])]
    if len(comparisons) != len(expected): raise ValueError('Missing test comparison rows')
    for row, (prob, threshold) in zip(comparisons, expected):
        _same_metrics(row, binary_metrics_at_threshold(y, prob[:,1], threshold))
    return {'status':'passed', 'study_run':study.name, 'winner':winner, 'candidate_count':len(rows),
            'epochs_per_candidate':10, 'verified_manifests':manifests,
            'test_samples':len(predictions), 'test_used_for_selection':False,
            'threshold':policy['selected_threshold'], 'training_seeds':[config['training']['seed']],
            'weights_loaded_for_audit':False, 'historical_test_previously_inspected':True}

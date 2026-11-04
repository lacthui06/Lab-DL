"""Kiểm toán frozen trial và dựng báo cáo so sánh từ artifact đã đo, không train."""
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, log_loss
from .evidence import read_json, verify_manifest
from .utils.runs import ROOT, hash_file, save_json, run_context
from .training.thresholds import binary_metrics_at_threshold, select_validation_threshold


def audit_frozen_trial(run):
    run = Path(run)
    manifest = verify_manifest(run)
    cfg = read_json(run/'config.json')['exercise2']
    rows = read_json(run/'epoch_history.json')
    selection = read_json(run/'selection_summary.json')
    frozen = read_json(run/'frozen_selection.json')
    if ([r['epoch'] for r in rows] != list(range(1, 11)) or cfg['num_train_epochs'] != 10
            or cfg['metric_for_best_model'] != 'loss' or selection['early_stopping']):
        raise ValueError('Expected ten epochs and validation-loss checkpoint selection')
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
        'historical_test_previously_inspected':True}


def ensure_final_review(root=ROOT):
    root = Path(root)
    configuration = read_json(root/'config/report.json')
    roles = configuration['runs']
    trial = root/'outputs/runs'/roles['frozen_trial']
    augmented = root/'outputs/runs'/roles['augmented_study']
    audit = audit_frozen_trial(trial)
    from .study_evidence import audit_study
    audit_study(augmented, root=root)
    sources = {'frozen_trial':trial, 'augmented_study':augmented,
        'unregularized_study':root/'outputs/runs'/roles['study']}
    key = {'source_manifests':{k:hash_file(p/'manifest.json') for k,p in sources.items()},
           'analysis_code_sha256':hash_file(Path(__file__))}
    registry = read_json(root/'config/workflow.json')
    for record in reversed(registry['runs']):
        if record['kind'] == 'final_review' and record['status'] == 'completed':
            folder = root/'outputs/runs'/record['id']
            if read_json(folder/'review_source.json') == key:
                verify_manifest(folder)
                return folder
    trial_rows = read_json(trial/'epoch_history.json')
    selected = read_json(trial/'selection_summary.json')
    augmented_winner = read_json(augmented/'study_summary.json')['winner']['run_id']
    augmented_epochs = read_json(root/'outputs/runs'/augmented_winner/'epoch_history.json')
    comparison = []
    for label, path in [('Unregularized 10 epoch',sources['unregularized_study']),
                        ('Augmented full fine-tuning',augmented), ('Frozen 4 layers',trial)]:
        metrics = read_json(path/'test_comparison.json')[-1]
        diagnostic = read_json(path/'probability_diagnostics.json')
        comparison.append({'experiment':label,'run_id':path.name,
            **{k:metrics[k] for k in ('threshold','accuracy','precision','recall','binary_f1','macro_f1','TN','FP','FN','TP')},
            'test_loss':read_json(path/'test_runtime_metrics.json')['test_loss'],
            'confidence_ece':diagnostic['confidence_ece'], 'brier_score':diagnostic['brier_score']})
    for source in (augmented, sources['unregularized_study']):
        records = read_json(source/'all_predictions.json')
        if [(r['index'],r['text'],r['true_label']) for r in records] != [(r['index'],r['text'],r['true_label']) for r in read_json(trial/'test_predictions.json')]:
            raise ValueError('Historical test cohorts differ')
    records = read_json(trial/'test_predictions.json')
    threshold = audit['threshold']
    y = np.array([r['true_label'] for r in records]); p = np.array([r['probabilities'][1] for r in records])
    predicted = (p >= threshold).astype(int)
    report = classification_report(y, predicted, labels=[0,1], target_names=['NEGATIVE','POSITIVE'], output_dict=True)
    per_class = []
    for c, name in enumerate(('NEGATIVE','POSITIVE')):
        fp = int(np.sum((predicted==c)&(y!=c))); negative_support = int(np.sum(y!=c))
        per_class.append({'class':name,**report[name], 'TPR':report[name]['recall'],'FPR':fp/negative_support})
    examples = [{**r, 'predicted_label':int(predicted[i]), 'confidence':float(max(r['probabilities'])),
        'correct':bool(predicted[i]==y[i])} for i,r in enumerate(records)]
    shown = []
    for correct in (True, False):
        for label in (0,1):
            candidates = [r for r in examples if r['correct']==correct and r['true_label']==label]
            shown.extend(sorted(candidates,key=lambda r:-r['confidence'])[:2])
    final = trial_rows[-1]; previous = augmented_epochs[-1]
    gap = {'frozen_final':{'accuracy_gap':final['train_accuracy']-final['validation_accuracy'],
        'loss_gap':final['validation_loss']-final['train_loss_eval']},
        'augmented_final':{'accuracy_gap':previous['train_accuracy']-previous['validation_accuracy'],
        'loss_gap':previous['validation_loss']-previous['train_loss_eval']}}
    with run_context('final_review',key,root=root) as out:
        save_json(out/'review_source.json',key);save_json(out/'audit.json',audit)
        save_json(out/'comparison.json',comparison);save_json(out/'per_class_metrics.json',per_class)
        save_json(out/'sample_predictions.json',shown);save_json(out/'generalization_gaps.json',gap)
        save_json(out/'review_summary.json',{'primary_result':'frozen_trial','run_id':trial.name,
            'selected_epoch':selected['selected_epoch'],'threshold':threshold,
            'purpose':'reduce observed overfitting, not maximize historical test accuracy',
            'one_training_seed':True,'causal_ablation_performed':False,
            'checkpoint_loaded_locally':False,'test_used_for_new_selection':False,
            'test_previously_inspected':True,'test_comparison':comparison})
        from .analysis import _table_image
        from .utils.plots import finish, plot_samples
        import matplotlib.pyplot as plt
        plot_samples(shown, out/'sample_predictions.png')
        _table_image(per_class,['class','precision','recall','f1-score','TPR','FPR','support'],
            out/'per_class_metrics.png','Class metrics at validation-selected threshold',widths=[.2,.14,.14,.14,.13,.13,.12])
        fig, axes = plt.subplots(1,2,figsize=(12,4))
        for name, rows in [('Augmented full fine-tuning',augmented_epochs),('Frozen 4 layers',trial_rows)]:
            axes[0].plot([r['epoch'] for r in rows],[r['validation_loss']-r['train_loss_eval'] for r in rows],marker='o',label=name)
            axes[1].plot([r['epoch'] for r in rows],[100*(r['train_accuracy']-r['validation_accuracy']) for r in rows],marker='o',label=name)
        for ax in axes:ax.set(xlabel='Epoch',xticks=list(range(1,11)));ax.grid(alpha=.2);ax.legend()
        axes[0].set_ylabel('Validation loss minus train loss (eval mode)')
        axes[1].set_ylabel('Train minus validation accuracy (percentage points)')
        finish(fig,out/'generalization_gaps.png')
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        for ax,key,label in zip(axes,['accuracy','test_loss','confidence_ece'],['Test accuracy','Test loss','Confidence ECE']):
            values=[r[key] for r in comparison]
            bars=ax.bar(['Original','Augmented','Frozen'],values);ax.bar_label(bars,fmt='%.3f');ax.set_title(label)
            ax.set_ylim(0, max(values)*1.2)
        finish(fig,out/'experiment_comparison.png')
    return out

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


def write_results_analysis(run, output):
    """Dẫn xuất báo cáo từ predictions/history đã audit; không sửa nguồn hoặc load model."""
    from .utils.runs import save_json,hash_file
    from .utils.plots import finish
    import matplotlib.pyplot as plt
    run,output=Path(run),Path(output)
    audit_frozen_trial(run)
    output.mkdir(parents=True,exist_ok=True)
    settings=read_json(run/'config.json')['exercise2']
    policy=read_json(run/'decision_threshold.json')
    records=read_json(run/'test_predictions.json')
    labels=np.array([r['true_label'] for r in records])
    probabilities=np.array([r['probabilities'] for r in records])
    predicted=probabilities[:,1]>=policy['selected_threshold']
    metrics=[]
    for label,name in [(0,'Negative'),(1,'Positive')]:
        truth,decision=labels==label,predicted==label
        tp=int(np.sum(truth & decision));fp=int(np.sum(~truth & decision))
        fn=int(np.sum(truth & ~decision));tn=int(np.sum(~truth & ~decision))
        ratio=lambda a,b:a/b if b else 0.
        precision,recall=ratio(tp,tp+fp),ratio(tp,tp+fn)
        metrics.append({'label':label,'class':name,'precision':precision,'recall':recall,
            'f1':ratio(2*precision*recall,precision+recall),'true_positive_rate':recall,
            'false_positive_rate':ratio(fp,fp+tn),'support':int(truth.sum()),
            'TP':tp,'FP':fp,'FN':fn,'TN':tn})
    history=read_json(run/'epoch_history.json')
    stop=settings.get('early_stopping',{})
    best=None;counter=0;replay=[]
    for row in history:
        loss=row['validation_loss']
        improved=best is None or (loss<best and abs(loss-best)>stop.get('min_delta',.001))
        counter=0 if improved else counter+1
        replay.append({'epoch':row['epoch'],'validation_loss':loss,'previous_best_loss':best,
            'improved_beyond_min_delta':improved,'patience_counter':counter})
        best=loss if best is None else min(best,loss)
    stopping={'source':'replay of Transformers 4.48.3 callback using config and epoch_history; not a recorded callback counter',
        'counter_was_logged':False,'epochs':replay,'enabled':stop.get('enabled',False),
        'patience_reached_at_budget':bool(stop.get('enabled') and counter>=stop.get('patience',2)
            and len(history)==settings['num_train_epochs'])}
    baseline=read_json(run/'baseline_test_predictions.json')
    baseline_probs=np.array([r['probabilities'] for r in baseline]);baseline_probs/=baseline_probs.sum(axis=1,keepdims=True)
    summary={'source_run':run.name,'source_manifest_sha256':hash_file(run/'manifest.json'),
        'analysis_code_sha256':hash_file(Path(__file__)),'test_samples':len(records),
        'test_loss':read_json(run/'test_runtime_metrics.json')['test_loss'],
        'baseline_log_loss':float(log_loss(labels,baseline_probs,labels=[0,1])),
        'baseline_loss_scope':'recomputed sklearn log_loss from full baseline test probabilities; not neural cross-entropy GPU timing',
        'final_accuracy_gap_pp':100*(history[-1]['train_accuracy']-history[-1]['validation_accuracy']),
        'final_loss_gap':history[-1]['validation_loss']-history[-1]['train_loss_eval'],
        'gpu_weights_loaded_for_analysis':False}
    save_json(output/'results_summary.json',summary)
    save_json(output/'per_class_metrics.json',metrics)
    save_json(output/'early_stopping_replay.json',stopping)
    detail=[{'epoch':row['epoch'],'train_accuracy_percent':100*row['train_accuracy'],
        'validation_accuracy_percent':100*row['validation_accuracy'],
        'accuracy_gap_pp':100*(row['train_accuracy']-row['validation_accuracy']),
        'loss_gap':row['validation_loss']-row['train_loss_eval']} for row in history]
    save_json(output/'learning_detail.json',{'source_run':run.name,
        'source_manifest_sha256':hash_file(run/'manifest.json'),'epochs':detail,
        'accuracy_scope':'both cohorts evaluated in eval mode; train includes augmentation'})
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    epochs=[row['epoch'] for row in detail]
    values=[row[key] for row in detail for key in ('train_accuracy_percent','validation_accuracy_percent')]
    for key,label in [('train_accuracy_percent','Train (eval mode)'),('validation_accuracy_percent','Validation')]:
        axes[0].plot(epochs,[row[key] for row in detail],marker='o',label=label)
    axes[0].set(xlabel='Epoch',ylabel='Accuracy (%)',title='Accuracy detail (zoomed axis)',
        xticks=epochs,ylim=(max(0,min(values)-.5),min(100,max(values)+.5)))
    axes[0].legend()
    axes[1].plot(epochs,[row['accuracy_gap_pp'] for row in detail],marker='o',label='Train - validation')
    axes[1].axhline(0,color='gray',linestyle='--')
    axes[1].set(xlabel='Epoch',ylabel='Accuracy gap (percentage points)',xticks=epochs,
        title='Gap alone does not prove absence of overfitting')
    axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    finish(fig,output/'learning_detail.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4));x=np.arange(2)
    for i,key in enumerate(('precision','recall','f1')):
        axes[0].bar(x+(i-1)*.24,[100*r[key] for r in metrics],width=.24,label=key)
    axes[0].set(xticks=x,xticklabels=['Negative','Positive'],ylim=(0,100),ylabel='Test (%)',title='Per-class metrics at validation-selected threshold')
    axes[0].legend()
    bars=axes[1].bar(['Negative','Positive'],[100*r['false_positive_rate'] for r in metrics])
    axes[1].bar_label(bars,fmt='%.2f%%')
    axes[1].set(ylim=(0,100),ylabel='One-vs-rest FPR (%)',title='False positive rate: other class predicted as this class')
    finish(fig,output/'per_class_metrics.png')
    return summary

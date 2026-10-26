"""Phân tích kết quả study đã đo; giữ nguyên mọi artifact và quyết định nguồn."""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import classification_report
from .utils.runs import ROOT, hash_file, run_context, save_json
from .evidence import read_json, verify_manifest
from .study_evidence import audit_study
from .utils.plots import finish


def ensure_study_review(study, root=ROOT):
    root = Path(root); study = Path(study)
    audit = audit_study(study, root=root)
    historical = root / 'outputs/runs/20261026T155822Z_finetuning_2c2f0401'
    key = {'study_run':study.name,'source_manifest_sha256':hash_file(study/'manifest.json'),
           'historical_run':historical.name,'historical_manifest_sha256':hash_file(historical/'manifest.json'),
           'analysis_code_sha256':hash_file(Path(__file__))}
    registry = read_json(root/'config/workflow.json')
    for row in reversed(registry['runs']):
        if row['kind'] != 'study_review' or row['status'] != 'completed': continue
        path = root/'outputs/runs'/row['id']
        if (path/'review_source.json').exists() and read_json(path/'review_source.json') == key:
            verify_manifest(path);return path
    candidates = read_json(study/'candidate_comparison.json')
    candidate_rows=[]; epoch_rows=[]
    for candidate in candidates:
        source=root/'outputs/runs'/candidate['run_id']
        summary=read_json(source/'selection_summary.json');history=read_json(source/'trainer_history.json')
        validations=[r for r in history if 'eval_macro_f1' in r]
        best=next(r for r in validations if r['step']==candidate['best_step'])
        lowest_loss=min(validations,key=lambda r:r['eval_loss'])
        last=validations[-1];train=[r for r in history if 'train_accuracy'in r][-1]
        candidate_rows.append({**candidate,'best_epoch':best['epoch'],'lowest_validation_loss':lowest_loss['eval_loss'],
          'lowest_loss_epoch':lowest_loss['epoch'],'lowest_loss_macro_f1':lowest_loss['eval_macro_f1'],
          'final_train_accuracy':train['train_accuracy'],'final_train_eval_loss':train['train_loss'],
          'final_validation_macro_f1':last['eval_macro_f1'],'final_validation_loss':last['eval_loss'],
          'global_steps':summary['global_step'],'successful_optimizer_updates':summary['successful_optimizer_updates'],
          'amp_skipped_optimizer_steps':summary['amp_skipped_optimizer_steps'],
          'validation_milestones':len(validations),'train_runtime_seconds':read_json(source/'train_metrics.json')['train_runtime']})
        for train_record in [r for r in history if 'train_accuracy'in r]:
            val=next(r for r in validations if r['step']==train_record['step'])
            epoch_rows.append({'run_id':source.name,'learning_rate':candidate['learning_rate'], 'epoch':train_record['epoch'],
                'step':train_record['step'],'train_eval_loss':train_record['train_loss'],'train_accuracy':train_record['train_accuracy'],
                'validation_loss':val['eval_loss'],'validation_accuracy':val['eval_accuracy'], 'validation_macro_f1':val['eval_macro_f1']})
    current=read_json(study/'all_predictions.json');previous=read_json(historical/'all_predictions.json')
    if [(r['index'],r['text'],r['true_label']) for r in current] != [(r['index'],r['text'],r['true_label']) for r in previous]:
        raise ValueError('Historical and current test cohorts differ')
    y=np.array([r['true_label'] for r in current]);new=np.array([r['predicted_label'] for r in current])
    old=np.array([r['predicted_label'] for r in previous]);delta=(new==y).astype(float)-(old==y).astype(float)
    rng=np.random.default_rng(42);bootstrap=[]
    for _ in range(2000):
        indices=np.concatenate([rng.choice(np.flatnonzero(y==c),sum(y==c),replace=True) for c in (0,1)])
        bootstrap.append(float(delta[indices].mean()))
    paired={'same_test_cohort_verified':True,'historical_correct':int(sum(old==y)),'study_correct':int(sum(new==y)),
      'both_correct':int(sum((old==y)&(new==y))),'both_wrong':int(sum((old!=y)&(new!=y))),
      'corrected_by_study':int(sum((old!=y)&(new==y))),'new_errors_in_study':int(sum((old==y)&(new!=y))),
      'accuracy_difference_study_minus_historical':float(delta.mean()),
      'paired_stratified_bootstrap_95_ci':np.quantile(bootstrap,[.025,.975]).tolist(),
      'bootstrap_repeats':2000,'bootstrap_seed':42,'selection_performed':False,
      'scope':'descriptive comparison of two fixed predictions on previously inspected test; not a controlled epoch-only ablation'}
    report=classification_report(y,new,labels=[0,1],target_names=['NEGATIVE','POSITIVE'],output_dict=True)
    per_class=[{'label':name,**report[name]} for name in ['NEGATIVE','POSITIVE']]
    diagnostics=read_json(study/'probability_diagnostics.json')
    with run_context('study_review',key,root=root) as out:
        save_json(out/'review_source.json',key);save_json(out/'audit.json',audit)
        save_json(out/'tables.json',{'candidates':candidate_rows,'epochs':epoch_rows,'per_class':per_class,'paired_comparison':paired})
        save_json(out/'review_summary.json',{'study_run':study.name,'winner':audit['winner'],
            'test_accuracy':float(np.mean(new==y)),'test_macro_f1':report['macro avg']['f1-score'],
            'threshold':audit['threshold'],'measured_test_loss':read_json(study/'test_runtime_metrics.json')['test_loss'],
            'probability_diagnostics':diagnostics,'paired_comparison':paired,'checkpoint_weights_loaded_locally':False,
            'full_gpu_training_repeated_locally':False,'threshold_reselected_on_test':False})
        from .analysis import _table_image
        _table_image(candidate_rows,['learning_rate','completed_epochs','best_step','best_epoch','macro_f1','loss','final_train_accuracy','final_validation_loss'],
                     out/'candidate_summary.png','All candidates: 10 epochs completed; selection by validation Macro-F1')
        _table_image(per_class,['label','precision','recall','f1-score','support'],out/'per_class_metrics.png','Study test: per-class metrics at frozen threshold 0.5')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for row in candidate_rows:
            history=[r for r in epoch_rows if r['run_id']==row['run_id']]
            axes[0].plot([r['epoch'] for r in history],[r['train_accuracy']-r['validation_accuracy'] for r in history],marker='o',label=f"LR={row['learning_rate']:g}")
            axes[1].plot([r['epoch'] for r in history],[r['validation_loss'] for r in history],marker='o',label=f"LR={row['learning_rate']:g}")
        axes[0].set(xlabel='Epoch',ylabel='Train-eval accuracy minus validation accuracy',title='Generalization gap (percentage points / 100)')
        axes[1].set(xlabel='Epoch',ylabel='Validation cross-entropy',title='Late confidence degradation')
        for ax in axes:ax.grid(alpha=.2);ax.legend()
        finish(fig,out/'overfitting_diagnostics.png')
        fig,ax=plt.subplots(figsize=(9,4))
        keys=['both_correct','both_wrong','corrected_by_study','new_errors_in_study']
        bars=ax.bar(['Both correct','Both wrong','Corrected by study','New errors'],[paired[k] for k in keys]);ax.bar_label(bars)
        ax.set(ylabel='Test examples',title='Historical 3-epoch vs selected 10-epoch-study checkpoint (argmax)')
        finish(fig,out/'historical_paired_comparison.png')
        print('Study review exported; no training or test-based selection.')
    return out

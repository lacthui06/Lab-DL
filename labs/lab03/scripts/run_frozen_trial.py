"""Fine-tuning regularized với ngân sách epoch và early stopping theo cấu hình."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from copy import deepcopy
import numpy as np
import torch
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs.lab03.src.utils.runs import ROOT, run_context, save_json, hash_file
from labs.lab03.src.datasets.loader import load_data, tokenize_data
from labs.lab03.src.datasets.augmentation import augment_train, export_augmentation
from labs.lab03.src.models.classifier import load_model, verify_saved_model
from labs.lab03.src.training.trainer import make_trainer
from labs.lab03.src.training.frozen import configure_frozen_optimizer
from labs.lab03.src.training.thresholds import select_validation_threshold, binary_metrics_at_threshold, scan_thresholds
from labs.lab03.src.utils.plots import plot_epoch_learning, plot_confusion, plot_samples, plot_metrics
from labs.lab03.src.study import fit_text_baseline, audit_split_texts, prediction_diagnostics
from labs.lab03.src.evidence import verify_manifest
from labs.lab03.src.performance import ResourceMonitor, write_performance


def probabilities(logits):
    return torch.tensor(logits, dtype=torch.float32).softmax(-1).numpy()


def plot_threshold(policy, curve, output):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    for key in ('macro_f1', 'precision', 'recall'):
        ax.plot([r['threshold'] for r in curve], [r[key] for r in curve], label=key, marker='o')
    ax.axvline(policy['selected_threshold'], color='black', linestyle='--', label='Validation selected')
    ax.set(xlabel='Threshold', ylabel='Validation metric', ylim=(0, 1.05))
    ax.legend(); fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def records(raw_split, probs):
    return [{'index': i, 'text': text, 'true_label': int(label),
             'probabilities': list(map(float, prob))}
            for i, (text, label, prob) in enumerate(zip(raw_split['text'], raw_split['label'], probs))]


def archive_destination(root, allow_cpu_fixture=False, kaggle_working=Path('/kaggle/working')):
    """Fixture luôn ghi trong tmp_path kể cả khi pytest đang chạy trên Kaggle."""
    if allow_cpu_fixture or not kaggle_working.is_dir():
        return Path(root)/'outputs'
    return kaggle_working


def resolve_device(device):
    if device not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('device must be auto, cpu or cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; enable GPU or choose device=cpu')
    return ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device


def publish_completed_trial(out,root=ROOT):
    from labs.lab03.scripts.publish_artifacts import publish_artifacts
    selection=Path(root)/'outputs/latest_report.json'
    runs={'frozen_trial':Path(out).name}
    session= json.loads((Path(out)/'config.json').read_text()).get('session_id')
    workflow=Path(root)/'config/workflow.json'
    if session and workflow.is_file():
        active=json.loads(workflow.read_text()).get('active_runs',{})
        for role in ('exercise1','exercise1_analysis'):
            run_id=active.get(role)
            if not run_id or Path(run_id).name != run_id:continue
            folder=Path(root)/'outputs/runs'/run_id
            if (folder/'config.json').is_file() and json.loads((folder/'config.json').read_text()).get('session_id')==session:
                runs[role]=run_id
    save_json(selection,{'primary_result':'frozen_trial','runs':runs})
    published=publish_artifacts(selection,output_dir=Path(root)/'outputs/published',source_root=root)
    save_json(Path(root)/'outputs/experiment_results.json',json.loads((published/'metrics_summary.json').read_text()))
    return published


def run(config, root=ROOT, allow_cpu_fixture=False, device='auto'):
    config = deepcopy(config)
    config['exercise2']['device'] = 'cpu' if allow_cpu_fixture else resolve_device(device)
    if config['exercise2']['device'] == 'cpu' and not allow_cpu_fixture:
        print('Device: CPU. Full training is supported but can be slow; CUDA is recommended.', flush=True)
    e = config['exercise2']
    budget = e['num_train_epochs']
    if isinstance(budget,bool) or not isinstance(budget,int) or budget < 1 or e['metric_for_best_model'] != 'loss':
        raise ValueError('This trial requires a positive epoch budget and validation-loss selection')
    set_seed(config['seed'])
    with run_context('frozen_trial', config, root=root) as out:
        sources = {p.relative_to(ROOT).as_posix(): hash_file(p)
                   for folder in ('src', 'scripts') for p in (ROOT/folder).rglob('*.py')}
        save_json(out/'source_code_sha256.json', sources)
        save_json(out/'source_commit.json', {'commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
        raw = load_data(e)
        save_json(out/'split_overlap_audit.json', audit_split_texts(raw))
        effective, aug = augment_train(raw, e['augmentation'], config['seed'])
        export_augmentation(aug, out)
        save_json(out/'data_statistics.json', {
            'official_sizes': {s: len(raw[s]) for s in raw},
            'effective_sizes': {s: len(effective[s]) for s in effective},
            'validation_and_test_augmented': False,
            'official_fingerprints': {s: raw[s]._fingerprint for s in raw}})
        baseline_settings = {'C_values': [.1, 1., 10.], 'ngram_range': [1, 2],
                             'min_df': 2, 'max_features': 50000}
        baseline, candidates, baseline_policy = fit_text_baseline(
            effective['train'], raw['validation'], baseline_settings, config['seed'])
        save_json(out/'baseline_validation_candidates.json', candidates)
        save_json(out/'baseline_policy.json', baseline_policy)
        (out/'baseline_architecture.txt').write_text(str(baseline), encoding='utf-8')
        model, tokenizer = load_model(e['model_name'], e['model_revision'],
            finetuning=True, overrides=e['model_overrides'])
        from labs.lab03.src.data_preparation import export_data_review
        export_data_review(raw,tokenizer,e,out)
        optimizer, freezing = configure_frozen_optimizer(model, e)
        save_json(out/'freezing_and_optimizer.json', freezing)
        save_json(out/'model_architecture.json', {'architecture':type(model).__name__,
            'model_name':e['model_name'],'revision':e['model_revision'],
            'layers':model.config.n_layers,'hidden_size':model.config.dim,
            'attention_heads':model.config.n_heads,'intermediate_size':model.config.hidden_dim,
            'num_labels':model.config.num_labels,'freeze_layers':e['freeze_layers'],
            'dropout':{k:getattr(model.config,k) for k in ['dropout','attention_dropout','seq_classif_dropout']},
            'source':'actual loaded model.config; recorded before training'})
        (out/'model_architecture.txt').write_text(str(model) + '\n\n' + json.dumps(freezing, indent=2), encoding='utf-8')
        print('Trainable parameters:', freezing['trainable_parameters'], '/', freezing['total_parameters'])
        ds = tokenize_data(effective, tokenizer, e['max_length'])
        trainer = make_trainer(model, tokenizer, ds, {**e, 'seed': config['seed'], 'record_resources':True}, out)
        trainer.optimizer = optimizer
        try:
            trainer.train()  # New model from pretrained base; no resume_from_checkpoint.
        finally:
            trainer.resource_callback.close()
        history = json.loads((out/'epoch_history.json').read_text())
        completed = len(history)
        assert [r['epoch'] for r in history] == list(range(1, completed+1)) and 1 <= completed <= budget
        best = min(history, key=lambda r: r['validation_loss'])
        assert best['step'] == int(Path(trainer.state.best_model_checkpoint).name.split('-')[-1])
        save_json(out/'trainer_history.json', trainer.state.log_history)
        save_json(out/'selection_summary.json', {
            'requested_epochs':budget, 'completed_epochs': float(trainer.state.epoch),
            'early_stopping':bool(e.get('early_stopping',{}).get('enabled',False)),
            'stopped_early':completed < budget, 'early_stopping_settings':e.get('early_stopping',{}),
            'checkpoint_selection_metric': 'validation_loss', 'selected_epoch': best['epoch'],
            'best_step': best['step'], 'best_validation_loss': best['validation_loss'],
            'global_step': trainer.state.global_step,
            'amp_skipped_optimizer_steps': getattr(trainer, 'amp_skipped_optimizer_steps', 0),
            'test_used_for_selection': False})
        plot_epoch_learning(trainer.state.log_history, out/'learning_curves.png')
        trainer.save_model(str(out/'best_model')); tokenizer.save_pretrained(out/'best_model')
        save_json(out/'reload_check.json', verify_saved_model(
            trainer.model, out/'best_model', list(raw['validation']['text'][:4]), e['max_length']))
        validation = trainer.predict(ds['validation'], metric_key_prefix='validation')
        vp = probabilities(validation.predictions)
        save_json(out/'validation_predictions.json', records(raw['validation'], vp))
        policy = select_validation_threshold(raw['validation']['label'], vp[:, 1])
        from labs.lab03.src.decision import checkpoint_identity
        policy['checkpoint_identity'] = checkpoint_identity(out/'best_model')
        save_json(out/'decision_threshold.json', policy)
        save_json(out/'best_model/decision_threshold.json', policy)
        save_json(out/'best_model/training_config.json', config)
        curve = scan_thresholds(raw['validation']['label'], vp[:, 1])
        save_json(out/'validation_threshold_scan.json', curve)
        plot_threshold(policy, curve, out/'validation_threshold_curves.png')
        save_json(out/'frozen_selection.json', {
            'checkpoint_selection_metric': 'validation_loss', 'selected_epoch': best['epoch'],
            'threshold': policy['selected_threshold'], 'selection_split': 'validation',
            'weights_sha256': hash_file(out/'best_model/model.safetensors'),
            'test_used_for_selection': False,
            'selection_split': 'validation'})
        # Test được đo sau khi đã ghi cố định checkpoint và threshold.
        monitor = ResourceMonitor(trainer.args.device, len(raw['test']),
                                  trainer.args.eval_batch_size, trainer.args.per_device_eval_batch_size).start()
        try:
            prediction = trainer.predict(ds['test'], metric_key_prefix='test')
        finally:
            test_resources = monitor.finish()
        save_json(out/'test_resource_metrics.json',test_resources)
        tp = probabilities(prediction.predictions); labels = raw['test']['label']
        save_json(out/'test_predictions.json', records(raw['test'], tp))
        save_json(out/'test_runtime_metrics.json', prediction.metrics)
        bp = baseline.predict_proba(raw['test']['text'])
        save_json(out/'baseline_test_predictions.json', records(raw['test'], bp))
        comparison = []
        for name, probs, threshold in [
            ('TF-IDF', bp, baseline_policy['selected_threshold']),
            ('Frozen DistilBERT threshold 0.5', tp, .5),
            ('Frozen DistilBERT validation threshold', tp, policy['selected_threshold'])]:
            comparison.append({'model': name, **binary_metrics_at_threshold(labels, probs[:, 1], threshold)})
        save_json(out/'test_comparison.json', comparison)
        chosen=comparison[-1]
        plot_metrics({'accuracy':chosen['accuracy'],'binary_precision':chosen['precision'],
            'binary_recall':chosen['recall'],'binary_f1':chosen['binary_f1']},out/'metrics_bar.png')
        plot_confusion(labels, tp[:, 1] >= policy['selected_threshold'], out/'confusion_matrix.png')
        examples = []
        for row in records(raw['test'], tp):
            predicted = int(row['probabilities'][1] >= policy['selected_threshold'])
            examples.append({**row, 'predicted_label':predicted,
                'confidence':row['probabilities'][predicted], 'correct':predicted == row['true_label']})
        shown = []
        for correct in (True, False):
            for label in (0, 1):
                shown.extend(sorted([r for r in examples if r['correct']==correct and r['true_label']==label],
                                    key=lambda r:-r['confidence'])[:2])
        save_json(out/'sample_predictions.json', shown)
        plot_samples(shown, out/'sample_predictions.png')
        diagnostics = prediction_diagnostics(labels, tp, out, policy['selected_threshold'], config['seed'])
        save_json(out/'probability_diagnostics.json', diagnostics)
        errors=json.loads((out/'threshold_errors.json').read_text())
        for row in errors:row['text']=raw['test'][row['index']]['text']
        save_json(out/'threshold_errors.json',errors)
        summary = f"""# Frozen DistilBERT trial\n\nRun: `{out.name}`\n\nTrain completed {completed}/{budget} epochs. Best checkpoint: epoch {best['epoch']} (validation loss).\n\nEmbedding and first {e['freeze_layers']} encoder layers frozen.\nEncoder LR {e['learning_rate']}; head LR {e['head_learning_rate']}.\n\nThreshold: {policy['selected_threshold']} (validation Macro-F1).\n\nThis configuration is a hypothesis, not a guaranteed improvement.\nOne seed only; no causal claim about individual regularization techniques.\n\n```json\n{json.dumps(comparison, indent=2)}\n```\n"""
        (out/'report.md').write_text(summary, encoding='utf-8')
        write_performance(out,out)
        print(summary)
    audit = verify_manifest(out)
    published = publish_completed_trial(out,root) if not allow_cpu_fixture else None
    save_json(out/'package_audit.json', audit)
    # package_audit is a derived packaging record, not part of the original run manifest.
    destination = archive_destination(root, allow_cpu_fixture)
    bundle = destination/f'lab03_review_{out.name}.zip'
    files = [p for p in out.rglob('*') if p.is_file() and not
             {'best_model', 'checkpoints'}.intersection(p.relative_to(out).parts)]
    if published:
        files.extend(p for p in published.rglob('*') if p.is_file())
        files.append(Path(root)/'outputs/latest_report.json')
    workflow = root/'config/workflow.json'
    if workflow.is_file():
        active = json.loads(workflow.read_text()).get('active_runs', {})
        for kind in ('exercise1', 'exercise1_analysis', 'dataset_preview', 'augmentation_preview'):
            run_id = active.get(kind)
            if not run_id or Path(run_id).name != run_id:
                continue
            folder = root/'outputs/runs'/run_id
            if not folder.is_dir():
                continue
            session=json.loads((out/'config.json').read_text()).get('session_id')
            folder_config=json.loads((folder/'config.json').read_text())
            if not session or folder_config.get('session_id') != session:
                continue
            if kind in ('dataset_preview', 'augmentation_preview'):
                previous = json.loads((folder/'config.json').read_text()).get('exercise2', {})
                if any(previous.get(k) != e.get(k) for k in (
                    'dataset_name', 'dataset_revision', 'model_name', 'model_revision', 'max_length')):
                    continue
            verify_manifest(folder)
            files.extend(p for p in folder.rglob('*') if p.is_file())
    for name in ('model_architectures.txt','summary.json','manifest.json'):
        architecture=root/'outputs/architecture'/name
        if architecture.is_file():files.append(architecture)
    preflight = sorted((root/'outputs/preflight').glob('*/pytest.*'))
    if preflight:
        latest = max(p.parent for p in preflight)
        files.extend(p for p in latest.iterdir() if p.is_file())
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
        manifest = {}
        for p in files:
            name = p.relative_to(root).as_posix(); z.write(p, name); manifest[name] = hash_file(p)
        z.writestr('bundle_manifest.json', json.dumps(manifest, indent=2))
        z.writestr('bundle_info.json', json.dumps({'run_id': out.name, 'weights_included': False,
            'checkpoint_selection': 'validation_loss', 'source_commit': json.loads((out/'source_commit.json').read_text())['commit']}, indent=2))
    with zipfile.ZipFile(destination/f'lab03_checkpoint_{out.name}.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in (out/'best_model').rglob('*'):
            if p.is_file(): z.write(p, p.relative_to(out).as_posix())
    print('Review ZIP:', bundle)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'config/frozen_trial.json')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), device=args.device)

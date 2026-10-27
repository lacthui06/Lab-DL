"""Thử regularization có kiểm soát; đủ 10 epoch, chọn checkpoint bằng val loss."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
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
from labs.lab03.src.utils.plots import plot_epoch_learning, plot_confusion
from labs.lab03.src.study import fit_text_baseline, audit_split_texts, prediction_diagnostics
from labs.lab03.src.evidence import verify_manifest


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


def run(config, root=ROOT, allow_cpu_fixture=False):
    if not torch.cuda.is_available() and not allow_cpu_fixture:
        raise RuntimeError('GPU is required; enable Kaggle Accelerator before training')
    e = config['exercise2']
    if e['num_train_epochs'] != 10 or e['metric_for_best_model'] != 'loss':
        raise ValueError('This trial requires 10 epochs and validation-loss checkpoint selection')
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
        model, tokenizer = load_model(e['model_name'], e['model_revision'],
            finetuning=True, overrides=e['model_overrides'])
        optimizer, freezing = configure_frozen_optimizer(model, e)
        save_json(out/'freezing_and_optimizer.json', freezing)
        print('Trainable parameters:', freezing['trainable_parameters'], '/', freezing['total_parameters'])
        ds = tokenize_data(effective, tokenizer, e['max_length'])
        trainer = make_trainer(model, tokenizer, ds, {**e, 'seed': config['seed']}, out)
        trainer.optimizer = optimizer
        trainer.train()  # New model from pretrained base; no resume_from_checkpoint.
        history = json.loads((out/'epoch_history.json').read_text())
        assert [r['epoch'] for r in history] == list(range(1, 11))
        best = min(history, key=lambda r: r['validation_loss'])
        assert best['step'] == int(Path(trainer.state.best_model_checkpoint).name.split('-')[-1])
        save_json(out/'trainer_history.json', trainer.state.log_history)
        save_json(out/'selection_summary.json', {
            'completed_epochs': float(trainer.state.epoch), 'early_stopping': False,
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
            'prior_test_exposure': 'Historical test results have already been inspected.'})
        # Test được đo sau khi đã ghi cố định checkpoint và threshold.
        prediction = trainer.predict(ds['test'], metric_key_prefix='test')
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
        plot_confusion(labels, tp[:, 1] >= policy['selected_threshold'], out/'confusion_matrix.png')
        diagnostics = prediction_diagnostics(labels, tp, out, policy['selected_threshold'], config['seed'])
        save_json(out/'probability_diagnostics.json', diagnostics)
        summary = f"""# Frozen DistilBERT trial\n\nRun: `{out.name}`\n\nTrain completed 10 epochs. Best checkpoint: epoch {best['epoch']} (validation loss).\n\nEmbedding and first {e['freeze_layers']} encoder layers frozen.\nEncoder LR {e['learning_rate']}; head LR {e['head_learning_rate']}.\n\nThreshold: {policy['selected_threshold']} (validation Macro-F1).\n\nThis configuration is a hypothesis, not a guaranteed improvement.\nOne seed only; historical test exposure; no causal claim about individual changes.\n\n```json\n{json.dumps(comparison, indent=2)}\n```\n"""
        (out/'report.md').write_text(summary, encoding='utf-8')
        print(summary)
    audit = verify_manifest(out)
    save_json(out/'package_audit.json', audit)
    # package_audit is a derived packaging record, not part of the original run manifest.
    destination = archive_destination(root, allow_cpu_fixture)
    bundle = destination/f'lab03_review_{out.name}.zip'
    files = [p for p in out.rglob('*') if p.is_file() and not
             {'best_model', 'checkpoints'}.intersection(p.relative_to(out).parts)]
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
            if kind in ('dataset_preview', 'augmentation_preview'):
                previous = json.loads((folder/'config.json').read_text()).get('exercise2', {})
                if any(previous.get(k) != e.get(k) for k in (
                    'dataset_name', 'dataset_revision', 'model_name', 'model_revision', 'max_length')):
                    continue
            verify_manifest(folder)
            files.extend(p for p in folder.rglob('*') if p.is_file())
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
            if p.is_file(): z.write(p, p.relative_to(out))
    print('Review ZIP:', bundle)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'config/frozen_trial.json')
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))

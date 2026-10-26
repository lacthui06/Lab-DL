"""Kiểm tra lựa chọn validation, ngăn dự đoán test sớm và artifact 10 epoch."""
import json
import numpy as np
import pytest
from datasets import Dataset, DatasetDict
from labs.lab03.src.study import choose_candidate, fit_text_baseline, prediction_diagnostics, audit_split_texts
from labs.lab03.src.workflow import read_config
from labs.lab03.src.training.trainer import make_trainer


def test_candidate_selection_rejects_test_and_uses_declared_ties():
    rows = [dict(selection_split='validation', learning_rate=rate, macro_f1=f1, loss=loss)
            for rate, f1, loss in [(1e-5, .8, .4), (2e-5, .81, .5), (3e-5, .81, .45)]]
    assert choose_candidate(rows)['learning_rate'] == 3e-5
    with pytest.raises(ValueError): choose_candidate([dict(rows[0], selection_split='test')])
    with pytest.raises(ValueError): choose_candidate([dict(rows[0], loss=float('nan'))])


def test_baseline_vocabulary_fitted_on_train_only():
    train = {'text': ['wonderful excellent', 'terrible boring', 'excellent lovely', 'boring awful'], 'label': [1, 0, 1, 0]}
    val = {'text': ['validationonlytoken excellent', 'validationonlytoken awful'], 'label': [1, 0]}
    settings = {'C_values': [.1, 1], 'ngram_range': [1, 2], 'min_df': 1, 'max_features': 50}
    model, rows, policy = fit_text_baseline(train, val, settings, 42)
    assert 'validationonlytoken' not in model['tfidf'].vocabulary_
    assert len(rows) == 2 and policy['split'] == 'validation'
    assert model.predict_proba(val['text']).shape == (2, 2)


def test_candidate_completes_ten_epochs_and_never_predicts_test(tiny, tmp_path, monkeypatch):
    from labs.lab03.src import workflow
    from labs.lab03.src.utils.runs import run_context
    from labs.lab03.src.datasets.loader import tokenize_data
    model, tok = tiny
    initial = tmp_path / 'initial'; model.save_pretrained(initial); tok.save_pretrained(initial)
    raw = Dataset.from_dict({'text': ['excellent movie', 'terrible film'] * 2, 'label': [1, 0] * 2})
    # Test không tồn tại: candidate phải dùng duy nhất train/validation cho predictions.
    ds = DatasetDict(train=raw, validation=raw, test=raw)
    monkeypatch.setattr(workflow, 'prepare_data', lambda config: (ds, tok, tokenize_data(ds, tok, 16)))
    monkeypatch.setattr(workflow, 'run_context', lambda kind, cfg: run_context(kind, cfg, root=tmp_path))
    original = workflow.predict_split
    def validation_only(trainer, tokens, source, out, split):
        assert split == 'validation'
        return original(trainer, tokens, source, out, split)
    monkeypatch.setattr(workflow, 'predict_split', validation_only)
    cfg = read_config(); cfg['exercise2'].update(model_name=str(initial), model_revision=None,
        num_train_epochs=10, batch_size=4, eval_batch_size=4, eval_steps=3, logging_steps=1, fp16=False)
    out = workflow.finetune(cfg, allow_cpu=True, evaluate_test=False)
    summary = json.loads((out / 'selection_summary.json').read_text())
    assert summary['completed_epochs'] == 10 and summary['global_step'] == 10
    assert summary['test_evaluated'] is False and not (out / 'all_predictions.json').exists()
    history = json.loads((out / 'trainer_history.json').read_text())
    assert sum('train_accuracy' in r for r in history) == 10
    assert sum('loss' in r for r in history) == 10
    assert sum('eval_macro_f1' in r for r in history) >= 10
    assert (out / 'step_learning_curves.png').is_file()


def test_probability_diagnostics_and_split_overlap(tmp_path):
    labels = [0, 0, 1, 1]; probs = np.array([[.8,.2],[.4,.6],[.1,.9],[.6,.4]])
    result = prediction_diagnostics(labels, probs, tmp_path, .7, repeats=30)
    assert result['roc_auc'] == .75
    assert len(result['bootstrap_95_percentile_ci']['accuracy']) == 2
    assert (tmp_path / 'roc_pr_curves.png').is_file()
    assert (tmp_path / 'calibration_confidence.png').is_file()
    ds = {k: Dataset.from_dict({'text': ['Same text', ' OTHER '], 'label': [0,1]}) for k in ['train','validation','test']}
    assert audit_split_texts(ds)['exact']['cross_split_overlaps']['train__test'] == 2


def test_study_end_to_end_selects_before_test_and_packages_complete_evidence(tiny, tmp_path, monkeypatch):
    from labs.lab03.src import workflow, study
    from labs.lab03.src.utils.runs import run_context
    from labs.lab03.src.datasets.loader import tokenize_data
    model, tok = tiny
    initial = tmp_path / 'initial'; model.save_pretrained(initial); tok.save_pretrained(initial)
    raw = Dataset.from_dict({'text': ['excellent movie', 'terrible film'] * 2, 'label': [1, 0] * 2})
    ds = DatasetDict(train=raw, validation=raw, test=raw)
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    monkeypatch.setattr(study, 'require_study_gpu', lambda: None)
    monkeypatch.setattr(study, 'load_data', lambda config: ds)
    monkeypatch.setattr(study, 'run_context', lambda kind, cfg: run_context(kind, cfg, root=tmp_path))
    monkeypatch.setattr(workflow, 'prepare_data', lambda config: (ds, tok, tokenize_data(ds, tok, 16)))
    monkeypatch.setattr(workflow, 'run_context', lambda kind, cfg: run_context(kind, cfg, root=tmp_path))
    original_fit = workflow.finetune
    monkeypatch.setattr(study, 'finetune', lambda cfg, evaluate_test: original_fit(cfg, allow_cpu=True, evaluate_test=evaluate_test))
    original_evaluate = study.evaluate_model
    def frozen_before_test(trainer, tokens, raw_test, out, threshold):
        frozen = json.loads((out / 'frozen_selection.json').read_text())
        assert frozen['test_used_for_selection'] is False
        assert frozen['threshold'] == threshold
        return original_evaluate(trainer, tokens, raw_test, out, threshold)
    monkeypatch.setattr(study, 'evaluate_model', frozen_before_test)
    cfg = read_config(); cfg['exercise2'].update(model_name=str(initial), model_revision=None,
        batch_size=4, eval_batch_size=4, eval_steps=3, logging_steps=1, fp16=False)
    settings = {'seed':42, 'epochs_per_candidate':10, 'learning_rates':[1e-5, 2e-5],
        'baseline':{'C_values':[.1,1], 'ngram_range':[1,2], 'min_df':1, 'max_features':50}, 'bootstrap_repeats':20}
    out = study.run_study(cfg, settings)
    from labs.lab03.src.study_evidence import audit_study
    audit = audit_study(out, root=tmp_path, expected_sizes={'train':4, 'validation':4, 'test':4})
    assert audit['status'] == 'passed'
    assert (out / 'study_report.md').is_file()
    assert len(json.loads((out / 'test_comparison.json').read_text())) == 4
    assert json.loads((out / 'status.json').read_text())['status'] == 'completed'
    import importlib.util
    from labs.lab03.src.utils.runs import ROOT as source
    spec = importlib.util.spec_from_file_location('package_study_test', source / 'scripts/package_study.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    review, checkpoint = module.package_study(out, root=tmp_path)
    from zipfile import ZipFile
    with ZipFile(review) as archive:
        assert not any('best_model/' in name or 'checkpoints/' in name for name in archive.namelist())
        assert any(name.endswith('console.log') for name in archive.namelist())
        assert any(name.endswith('study_report.md') for name in archive.namelist())
    with ZipFile(checkpoint) as archive:
        assert any(name.endswith('best_model/model.safetensors') for name in archive.namelist())
    spec = importlib.util.spec_from_file_location('display_study_test', source / 'scripts/display_study.py')
    viewer = importlib.util.module_from_spec(spec); spec.loader.exec_module(viewer)
    import IPython.display
    displayed = []
    monkeypatch.setattr(IPython.display, 'display', displayed.append)
    shown = viewer.display_study(out, root=tmp_path)
    assert len(shown) >= 10
    assert sum(isinstance(x, IPython.display.Image) for x in displayed) == len(shown)
    # Rehash không che được metric bị sửa: audit còn tính lại từ predictions.
    from labs.lab03.src.utils.runs import hash_file
    metric_path = out / 'test_metrics.json'
    altered = json.loads(metric_path.read_text()); altered['accuracy'] += .01
    metric_path.write_text(json.dumps(altered))
    manifest = json.loads((out / 'manifest.json').read_text()); manifest['test_metrics.json'] = hash_file(metric_path)
    (out / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Metric mismatch'): audit_study(out, root=tmp_path, expected_sizes={'train':4,'validation':4,'test':4})



def test_existing_audit_accepts_step_records_but_requires_every_epoch_end(tmp_path):
    import shutil
    from labs.lab03.src.evidence import audit_finetuning
    from labs.lab03.src.utils.runs import ROOT, hash_file
    original = ROOT / 'outputs/runs/20261025T181550Z_finetuning_e02bced7'
    source = tmp_path / 'source'; shutil.copytree(original, source)
    history = json.loads((source / 'trainer_history.json').read_text())
    intermediate = dict(next(r for r in history if 'eval_accuracy' in r))
    intermediate.update(epoch=.5, step=100)
    history.insert(0, intermediate)
    path = source / 'trainer_history.json'; path.write_text(json.dumps(history))
    manifest = json.loads((source / 'manifest.json').read_text())
    manifest['trainer_history.json'] = hash_file(path)
    (source / 'manifest.json').write_text(json.dumps(manifest))
    assert audit_finetuning(source)['correct'] == 895
    history = [r for r in history if not ('eval_accuracy' in r and r['epoch'] == 2)]
    path.write_text(json.dumps(history)); manifest['trainer_history.json'] = hash_file(path)
    (source / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Incomplete training history'): audit_finetuning(source)



def test_nonfinite_loss_fails_but_amp_overflow_is_explicitly_recorded(tiny, tmp_path):
    from types import SimpleNamespace
    model, tok = tiny
    tokens = tok(['excellent movie', 'terrible film'])
    data = Dataset.from_dict({**tokens, 'label':[1,0]})
    cfg = dict(read_config()['exercise2'], seed=42, fp16=False)
    trainer = make_trainer(model, tok, DatasetDict(train=data, validation=data), cfg, tmp_path)
    callback = next(c for c in trainer.callback_handler.callbacks if type(c).__name__ == 'TrainingMetrics')
    state = SimpleNamespace(global_step=1, log_history=[{'grad_norm':float('inf')}])
    with pytest.raises(FloatingPointError, match='Nonfinite loss'):
        callback.on_log(SimpleNamespace(fp16=True), state, None, logs={'loss':float('nan')})
    logs = {'grad_norm':float('inf')}
    with pytest.warns(RuntimeWarning, match='AMP nonfinite'):
        callback.on_log(SimpleNamespace(fp16=True), state, None, logs=logs)
    assert logs['grad_norm'] is None and logs['amp_nonfinite_grad_norm'] is True
    assert state.log_history[-1]['amp_nonfinite_grad_norm'] is True

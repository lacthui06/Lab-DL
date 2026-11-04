"""Kiểm tra lựa chọn validation, ngăn dự đoán test sớm và artifact của profile hiện hành."""
import json
import numpy as np
import pytest
from datasets import Dataset, DatasetDict
from src.study import choose_candidate, fit_text_baseline, prediction_diagnostics, audit_split_texts
from src.workflow import read_config
from src.training.trainer import make_trainer


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


def test_probability_diagnostics_and_split_overlap(tmp_path):
    labels = [0, 0, 1, 1]; probs = np.array([[.8,.2],[.4,.6],[.1,.9],[.6,.4]])
    result = prediction_diagnostics(labels, probs, tmp_path, .7, repeats=30)
    assert result['roc_auc'] == .75
    assert len(result['bootstrap_95_percentile_ci']['accuracy']) == 2
    assert (tmp_path / 'roc_pr_curves.png').is_file()
    assert (tmp_path / 'calibration_confidence.png').is_file()
    ds = {k: Dataset.from_dict({'text': ['Same text', ' OTHER '], 'label': [0,1]}) for k in ['train','validation','test']}
    assert audit_split_texts(ds)['exact']['cross_split_overlaps']['train__test'] == 2


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

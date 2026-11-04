"""Epoch logs và curves phải đúng số epoch đã chạy, không biến đổi evidence gốc."""
import json
import shutil
import numpy as np
import pytest
from datasets import Dataset, DatasetDict
from src.utils.runs import ROOT, hash_file, save_json, run_context
from src.utils.plots import epoch_metrics, plot_epoch_learning


def test_epoch_metrics_rejects_missing_end_of_epoch():
    with pytest.raises(ValueError,match='Missing end-of-epoch'):
        epoch_metrics([{'epoch':1,'step':4,'train_accuracy':.8,'train_loss':.4},
                       {'epoch':.5,'step':2,'eval_loss':.5,'eval_accuracy':.7}])


def test_regularized_training_runs_three_complete_epochs_and_prints_three_summaries(tiny,tmp_path,monkeypatch,capsys):
    from src import workflow
    from src.datasets.loader import tokenize_data
    model,tok=tiny;initial=tmp_path/'initial';model.save_pretrained(initial);tok.save_pretrained(initial)
    raw=Dataset.from_dict({'text':['excellent movie','terrible film']*2,'label':[1,0]*2})
    ds=DatasetDict(train=raw,validation=raw,test=raw)
    tokens=tokenize_data(ds,tok,16)
    monkeypatch.setattr(workflow,'prepare_data',lambda config:(ds,tok,tokens))
    monkeypatch.setattr(workflow,'run_context',lambda kind,cfg:run_context(kind,cfg,root=tmp_path))
    cfg=workflow.read_config(ROOT/'config/frozen_trial.json')
    cfg['exercise2']['early_stopping']={'enabled':False}
    cfg['exercise2']['augmentation']={'enabled':False}
    cfg['exercise2'].update(model_name=str(initial),model_revision=None,batch_size=4,eval_batch_size=4,fp16=False)
    out=workflow.finetune(cfg,allow_cpu=True,evaluate_test=False)
    rows=json.loads((out/'epoch_history.json').read_text())
    assert [r['epoch'] for r in rows]==list(range(1,4))
    assert all(np.isfinite([r['train_loss_eval'],r['validation_loss']]).all() for r in rows)
    history=json.loads((out/'trainer_history.json').read_text())
    assert sum('eval_accuracy'in r for r in history)==3
    output=capsys.readouterr().out
    assert output.count('Train Loss (eval):')==3
    assert 'Epoch 01/03' in output and 'Epoch 03/03' in output
    config=json.loads((out/'best_model/config.json').read_text())
    assert config['dropout']==.2 and config['attention_dropout']==.2 and config['seq_classif_dropout']==.4
    provenance=json.loads((out/'provenance.json').read_text())
    assert provenance['effective_dropout']['seq_classif_dropout']==.4
    assert not (out/'all_predictions.json').exists()


@pytest.mark.parametrize('overrides',[{'dropout':float('nan')},{'dropout':1.0},{'dropout':-.1},{'hidden_size':32}])
def test_dropout_override_rejected_before_loading_model(overrides):
    from src.models.classifier import load_model
    with pytest.raises(ValueError):load_model('must-not-download',finetuning=True,overrides=overrides)

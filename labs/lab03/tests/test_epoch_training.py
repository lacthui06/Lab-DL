"""Epoch logs và curves phải đúng 10 mốc, không biến đổi evidence gốc."""
import json
import shutil
import numpy as np
import pytest
from datasets import Dataset, DatasetDict
from labs.lab03.src.utils.runs import ROOT, hash_file, save_json, run_context
from labs.lab03.src.utils.plots import epoch_metrics, plot_epoch_learning


def test_epoch_plot_has_exactly_ten_integer_points_from_dense_real_history(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from labs.lab03.src.utils import plots
    source=ROOT/'outputs/runs/20261026T185804Z_candidate_8308037d/trainer_history.json'
    original_hash=hash_file(source);history=json.loads(source.read_text())
    assert sum('eval_accuracy' in r for r in history)==63
    captured=[]
    monkeypatch.setattr(plots,'finish',lambda fig,path:captured.append(fig))
    rows=plot_epoch_learning(history,tmp_path/'curve.png')
    assert [r['epoch'] for r in rows]==list(range(1,11))
    for ax in captured[0].axes:
        assert len(ax.lines)==2
        for line in ax.lines:
            assert list(line.get_xdata())==list(range(1,11))
    assert rows[-1]['train_accuracy']==1.0
    assert rows[-1]['validation_loss']>rows[0]['validation_loss']
    plt.close(captured[0]);assert hash_file(source)==original_hash


def test_epoch_metrics_rejects_missing_end_of_epoch():
    with pytest.raises(ValueError,match='Missing end-of-epoch'):
        epoch_metrics([{'epoch':1,'step':4,'train_accuracy':.8,'train_loss':.4},
                       {'epoch':.5,'step':2,'eval_loss':.5,'eval_accuracy':.7}])


def test_regularized_training_runs_ten_complete_epochs_and_prints_ten_summaries(tiny,tmp_path,monkeypatch,capsys):
    from labs.lab03.src import workflow
    from labs.lab03.src.datasets.loader import tokenize_data
    model,tok=tiny;initial=tmp_path/'initial';model.save_pretrained(initial);tok.save_pretrained(initial)
    raw=Dataset.from_dict({'text':['excellent movie','terrible film']*2,'label':[1,0]*2})
    ds=DatasetDict(train=raw,validation=raw,test=raw)
    tokens=tokenize_data(ds,tok,16)
    monkeypatch.setattr(workflow,'prepare_data',lambda config:(ds,tok,tokens))
    monkeypatch.setattr(workflow,'run_context',lambda kind,cfg:run_context(kind,cfg,root=tmp_path))
    cfg=workflow.read_config(ROOT/'config/regularized.json')
    cfg['exercise2'].update(model_name=str(initial),model_revision=None,batch_size=4,eval_batch_size=4,fp16=False)
    out=workflow.finetune(cfg,allow_cpu=True,evaluate_test=False)
    rows=json.loads((out/'epoch_history.json').read_text())
    assert [r['epoch'] for r in rows]==list(range(1,11))
    assert all(np.isfinite([r['train_loss_eval'],r['validation_loss']]).all() for r in rows)
    history=json.loads((out/'trainer_history.json').read_text())
    assert sum('eval_accuracy'in r for r in history)==10
    output=capsys.readouterr().out
    assert output.count('Train Loss (eval):')==10
    assert 'Epoch 01/10' in output and 'Epoch 10/10' in output
    config=json.loads((out/'best_model/config.json').read_text())
    assert config['dropout']==.2 and config['attention_dropout']==.2 and config['seq_classif_dropout']==.3
    provenance=json.loads((out/'provenance.json').read_text())
    assert provenance['effective_dropout']['seq_classif_dropout']==.3
    assert not (out/'all_predictions.json').exists()


@pytest.mark.parametrize('overrides',[{'dropout':float('nan')},{'dropout':1.0},{'dropout':-.1},{'hidden_size':32}])
def test_dropout_override_rejected_before_loading_model(overrides):
    from labs.lab03.src.models.classifier import load_model
    with pytest.raises(ValueError):load_model('must-not-download',finetuning=True,overrides=overrides)


def test_epoch_presentation_cached_without_changing_any_original_source(tmp_path):
    from labs.lab03.src.epoch_presentation import ensure_epoch_presentation
    ids=['20261026T184219Z_study_94f2ba7b','20261026T184223Z_candidate_4b1da858',
         '20261026T185017Z_candidate_202c0434','20261026T185804Z_candidate_8308037d']
    for run_id in ids:shutil.copytree(ROOT/'outputs/runs'/run_id,tmp_path/'outputs/runs'/run_id)
    save_json(tmp_path/'config/workflow.json',{'runs':[],'active_runs':{}})
    hashes={str(p):hash_file(p) for run_id in ids for p in (tmp_path/'outputs/runs'/run_id).rglob('*') if p.is_file()}
    out=ensure_epoch_presentation(tmp_path/'outputs/runs'/ids[0],root=tmp_path)
    summary=json.loads((out/'summary.json').read_text())
    assert all(r['points_per_curve']==10 for r in summary['candidates'])
    assert ensure_epoch_presentation(tmp_path/'outputs/runs'/ids[0],root=tmp_path)==out
    assert all(hash_file(p)==digest for p,digest in hashes.items())

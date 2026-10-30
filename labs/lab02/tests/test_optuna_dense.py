"""Loss selection and LR search contracts, without downloading datasets."""
import json
from pathlib import Path
import optuna
import pytest
import torch
from torch.utils.data import DataLoader,TensorDataset
from src.optuna_dense import suggest
from src import experiment


def test_shared_and_separate_rates():
    cfg={'lr_range':[1e-6,1e-3],'weight_decay_range':[1e-7,1e-3]}
    params={'backbone_lr':1e-5,'head_lr':1e-4,'weight_decay':1e-4}
    assert suggest(optuna.trial.FixedTrial(params),'shared',cfg)['head_lr']==1e-5
    assert suggest(optuna.trial.FixedTrial(params),'separate',cfg)['head_lr']==1e-4


def test_loss_selection_uses_same_threshold_as_stopping(tmp_path,monkeypatch):
    losses=iter([.5,.4995,.498,.4975,.4974,.4973])
    monkeypatch.setattr(experiment,'pass_epoch',lambda m,l,d,optimizer=None,mode_fn=None:
        {'loss':1.,'accuracy':.9} if optimizer is not None else {'loss':next(losses),'accuracy':.8})
    model=torch.nn.Linear(2,2)
    batch=DataLoader(TensorDataset(torch.ones(2,2),torch.zeros(2,dtype=torch.long)),batch_size=2)
    summary=experiment.fit(model,batch,batch,torch.optim.SGD(model.parameters(),lr=.1),6,
        torch.device('cpu'),tmp_path,{},stopping={'min_epochs':6,'patience':3,'min_delta':.001},selection_metric='loss')
    saved=torch.load(tmp_path/'best.pt',weights_only=True)
    assert saved['best_epoch']==3
    assert summary['best_val_loss']==pytest.approx(.498)
    assert summary['stop_reason']=='validation_loss_patience'


def test_budget_and_fixed_scope():
    from src.models import build_model,configure_trainable
    cfg=json.loads((Path(__file__).resolve().parents[1]/'config/optuna_dense.json').read_text())
    assert cfg['epochs']==20 and cfg['trials_per_study']==5
    model=build_model('densenet121',pretrained=False)
    configure_trainable(model,'densenet121','finetune','last_block')
    for name,p in model.named_parameters():
        assert p.requires_grad == name.startswith(('features.denseblock4.','features.norm5.','classifier.'))

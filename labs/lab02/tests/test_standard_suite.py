"""Exercise stage reuse and equal shared LR search without network or raw images."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from src import standard_suite as module


def test_cached_head_and_three_shared_lr_trials(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    cfg=json.loads((Path(__file__).resolve().parents[1]/'config/standard.json').read_text())
    assert cfg['trials_per_model']==3 and cfg['finetune_epochs']==10
    cfg.update(head_epochs=2,finetune_epochs=2,early_stopping={'min_epochs':1,'patience':1,'min_delta':.001})
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.fc=torch.nn.Linear(2,10)
        def forward(self,x):return self.fc(x)
    monkeypatch.setattr(module,'build_model',lambda *a,**k:Tiny())
    monkeypatch.setattr(module,'configure_trainable',lambda *a,**k:None)
    monkeypatch.setattr(module,'training_mode',lambda m:m.train())
    cached=[]
    data=(torch.randn(6,2),torch.tensor([0,1,0,1,0,1]),torch.arange(6))
    def cache(*args):
        cached.append(args[3]);return data,.01
    monkeypatch.setattr(module,'cached_features',cache)
    monkeypatch.setattr(module,'image_loader',lambda *a,**k:DataLoader(TensorDataset(*data),batch_size=3))
    suite=object.__new__(module.Suite)
    suite.root=tmp_path;suite.run=tmp_path/'run';suite.run.mkdir()
    suite.cfg=cfg;suite.device=torch.device('cpu');suite.data=tmp_path;suite.callback=None
    suite.ids={s:np.arange(6) for s in ('train','val','test')}
    folder=suite.head('resnet18');suite.head('resnet18')
    assert cached==['train','val']
    saved=torch.load(folder/'best.pt',weights_only=True)
    assert 'fc.weight' in saved['state_dict']
    result=suite.search('resnet18')
    assert len(pd.read_csv(suite.run/'resnet18/trials.csv'))==3
    assert result==suite.search('resnet18')
    for path in (suite.run/'resnet18').glob('trial_*/config.json'):
        metadata=json.loads(path.read_text())
        assert metadata['weight_decay']==1e-4
        assert metadata['unfreeze']=='last_block'


def test_test_evaluation_waits_for_both_winners(tmp_path):
    import pytest
    suite=object.__new__(module.Suite);suite.run=tmp_path
    suite.cfg={'models':['resnet18','densenet121']}
    with pytest.raises(RuntimeError,match='both model searches'):
        suite.evaluate('resnet18')

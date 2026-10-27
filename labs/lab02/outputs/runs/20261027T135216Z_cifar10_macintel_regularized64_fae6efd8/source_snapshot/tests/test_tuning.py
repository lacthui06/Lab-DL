"""Protect validation-only selection, optimizer groups, scheduling and unfreezing."""
import copy
import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import build_model, configure_trainable, parameter_counts, training_mode
from src.tuning import fine_candidates, select_trial, make_optimizer, make_scheduler


@pytest.mark.parametrize('improves,expected,reason', [
    (False, 10, 'validation_loss_patience'),
    (True, 12, 'epoch_budget_exhausted'),
])
def test_stopping_respects_minimum_and_reports_actual_epochs(tmp_path, monkeypatch, improves, expected, reason):
    from src import experiment
    model = nn.Linear(2, 2)
    loader = DataLoader(TensorDataset(torch.ones(2, 2), torch.tensor([0, 1])), batch_size=2)
    state = {'epoch': 0}

    def epoch(model, loader, device, optimizer=None, mode_fn=None):
        if optimizer is not None:
            return {'loss': .8, 'accuracy': .5}
        state['epoch'] += 1
        return {'loss': 1/state['epoch'] if improves else .9, 'accuracy': .6}

    monkeypatch.setattr(experiment, 'pass_epoch', epoch)
    opt = torch.optim.SGD(model.parameters(), lr=.01)
    result = experiment.fit(model, loader, loader, opt, 12, torch.device('cpu'), tmp_path,
        {'model': 'tiny', 'stage': 'finetune', 'batch_size': 2, 'lr': .01},
        stopping={'min_epochs': 10, 'patience': 3, 'min_delta': .0001})
    assert result['epochs_completed'] == expected
    assert result['epochs_planned'] == 12
    assert result['stop_reason'] == reason
    assert 'not established' in result['convergence_claim']


def test_active_batch_grid_and_invalid_stopping_policy():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for name in ['full.json', 'tuned.json', 'cpu_full.json', 'cpu_pilot.json']:
        cfg = json.loads((root/'config'/name).read_text())
        assert cfg['batch_sizes'] == [64, 128]
    cfg = json.loads((root/'config/tuned.json').read_text())
    assert cfg['head_epochs'] == 10 and cfg['finetune_epochs'] == 30
    cfg['early_stopping']['min_epochs'] = 31
    with pytest.raises(ValueError, match='min_epochs'):
        fine_candidates(cfg)


def test_selection_uses_time_only_inside_both_quality_bands():
    rows = [dict(id='accurate', best_val_accuracy=.85, best_val_loss=.51, train_seconds=100, accuracy=.1),
            dict(id='efficient', best_val_accuracy=.8495, best_val_loss=.508, train_seconds=50, accuracy=.2),
            dict(id='low_quality', best_val_accuracy=.84, best_val_loss=.49, train_seconds=1, accuracy=1),
            dict(id='high_loss', best_val_accuracy=.85, best_val_loss=.53, train_seconds=2, accuracy=1)]
    assert select_trial(rows, dict(accuracy_tolerance=.001, loss_tolerance=.005))['id'] == 'efficient'
    assert select_trial(rows)['id'] == 'accurate'
    changed = [{**r, 'accuracy':1-r['accuracy']} for r in rows]
    assert select_trial(changed, dict(accuracy_tolerance=.001, loss_tolerance=.005))['id'] == 'efficient'
    with pytest.raises(ValueError):
        select_trial([{**rows[0], 'train_seconds':float('nan')}])


def test_reject_stale_prefix_cache_for_deeper_unfreeze():
    cfg = dict(cache_frozen_prefix=True, finetune_candidates=[dict(
        backbone_lr=1e-5, head_lr=1e-4, unfreeze='last_two_blocks')])
    with pytest.raises(ValueError, match='cache_frozen_prefix=false'):
        fine_candidates(cfg)
    cfg['cache_frozen_prefix'] = False
    assert fine_candidates(cfg)[0]['unfreeze'] == 'last_two_blocks'


@pytest.mark.parametrize('name', ['resnet18', 'densenet121'])
def test_opening_two_blocks_changes_gradients_and_preserves_earlier_weights(name):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = build_model(name, pretrained=False)
    configure_trainable(model, name, 'finetune')
    previous = parameter_counts(model)['trainable']
    configure_trainable(model, name, 'finetune', 'last_two_blocks')
    assert parameter_counts(model)['trainable'] > previous
    earlier = model.layer2 if name == 'resnet18' else model.features.denseblock2
    opened = model.layer3 if name == 'resnet18' else model.features.denseblock3
    assert all(not p.requires_grad for p in earlier.parameters())
    assert all(p.requires_grad for p in opened.parameters())
    if name == 'densenet121':
        assert all(p.requires_grad for p in model.features.transition3.parameters())
    training_mode(model)
    loss = nn.functional.cross_entropy(model(torch.randn(2,3,64,64)), torch.tensor([0,1]))
    loss.backward()
    assert all(p.grad is None for p in earlier.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in opened.parameters())


@pytest.mark.parametrize('name', ['adamw','sgd'])
def test_optimizer_groups_and_cosine_update(name):
    model = nn.Sequential(nn.Linear(4,4), nn.Linear(4,2))
    optimizer = make_optimizer([dict(params=model[0].parameters(),lr=1e-4),
                                dict(params=model[1].parameters(),lr=1e-3)], name, 1e-4)
    scheduler = make_scheduler(optimizer, 'cosine', 4)
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(2,4)).sum().backward()
        optimizer.step(); scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0, abs=1e-12)
    assert optimizer.param_groups[1]['lr'] == pytest.approx(0, abs=1e-12)
    if name == 'sgd':
        assert optimizer.param_groups[0]['momentum'] == .9


def test_fit_logs_validation_loss_lr_and_time_at_selected_checkpoint(tmp_path):
    from src.experiment import fit
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = nn.Linear(8,2)
    data = TensorDataset(torch.eye(8), torch.arange(8)%2)
    loader = DataLoader(data,batch_size=8)
    optimizer = torch.optim.SGD(model.parameters(),lr=.5)
    scheduler = make_scheduler(optimizer,'cosine',3)
    meta = dict(model='tiny',stage='head',batch_size=8,lr=.5,optimizer='sgd')
    result = fit(model,loader,loader,optimizer,3,torch.device('cpu'),tmp_path,meta,scheduler=scheduler)
    import pandas as pd
    history = pd.read_csv(tmp_path/'history.csv')
    selected = history[history.epoch==result['best_epoch']].iloc[0]
    assert result['best_val_loss'] == pytest.approx(selected.val_loss)
    assert history.lr_group_0.iloc[0] == .5
    assert history.lr_group_0.iloc[-1] < .5
    assert result['train_seconds'] >= result['train_pass_seconds']+result['validation_seconds']
    checkpoint=torch.load(tmp_path/'best.pt',weights_only=True)
    assert checkpoint['val_loss'] == pytest.approx(result['best_val_loss'])


def test_epoch_resume_matches_uninterrupted_training_and_events(tmp_path, monkeypatch):
    """An interrupted epoch is replayed with identical dropout, shuffle and LR."""
    from src import experiment
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    torch.set_num_threads(2)
    data = TensorDataset(torch.eye(8), torch.arange(8) % 2)
    meta = dict(model='tiny', stage='head', batch_size=2, lr=.1, optimizer='adamw')

    def setup():
        experiment.seed_all(42)
        model = nn.Sequential(nn.Linear(8, 4), nn.Dropout(.3), nn.Linear(4, 2))
        opt = torch.optim.AdamW(model.parameters(), lr=.1)
        sched = make_scheduler(opt, 'cosine', 4)
        train = DataLoader(data, batch_size=2, shuffle=True,
                           generator=torch.Generator().manual_seed(42))
        val = DataLoader(data, batch_size=2, generator=torch.Generator().manual_seed(43))
        return model, opt, sched, train, val

    full, opt, sched, train, val = setup()
    experiment.fit(full, train, val, opt, 4, torch.device('cpu'), tmp_path/'full', meta, scheduler=sched)
    model, opt, sched, train, val = setup()
    original = experiment.pass_epoch
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 5:
            torch.rand(9)  # Simulate work consumed in an incomplete epoch.
            raise RuntimeError('simulated interruption')
        return original(*args, **kwargs)

    monkeypatch.setattr(experiment, 'pass_epoch', interrupt)
    with pytest.raises(RuntimeError, match='simulated'):
        experiment.fit(model, train, val, opt, 4, torch.device('cpu'), tmp_path/'resume', meta, scheduler=sched)
    monkeypatch.setattr(experiment, 'pass_epoch', original)
    model, opt, sched, train, val = setup()
    result = experiment.fit(model, train, val, opt, 4, torch.device('cpu'), tmp_path/'resume', meta,
                            scheduler=sched, resume=True)
    assert result['epochs_completed'] == 4
    for key, expected in full.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)
    checkpoint = torch.load(tmp_path/'resume/last.pt', weights_only=True)
    assert checkpoint['epoch'] == 4 and checkpoint['optimizer_state_dict']['state']
    assert checkpoint['scheduler_state_dict'] == sched.state_dict()
    events = EventAccumulator(str(tmp_path/'resume/tensorboard')).Reload()
    assert [e.step for e in events.Scalars('val_loss')] == [1, 2, 3, 4]
    with pytest.raises(ValueError, match='configuration'):
        experiment.fit(model, train, val, opt, 5, torch.device('cpu'), tmp_path/'resume', meta,
                       scheduler=sched, resume=True)


def test_resume_cli_restores_head_trial_without_finalizing_suite(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    import numpy as np
    from src import experiment, models
    from src.delivery import file_hash, write_json
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('resume_cli', root/'scripts/resume_interrupted.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    torch.set_num_threads(2)
    run = tmp_path/'outputs/runs/example'
    trial = run/'resnet18/head_trial_00'
    trial.mkdir(parents=True)
    indices = np.arange(8)
    np.savez(run/'selected_indices.npz', train=indices, val=indices)
    write_json(run/'source_hashes.json', {'src/models.py': file_hash(root/'src/models.py')})
    write_json(run/'status.json', {'status': 'failed'})
    cfg = dict(threads=2, head_epochs=3, finetune_epochs=3, weight_decay=0.)
    meta = dict(model='resnet18', stage='head', optimizer='adamw', lr=.1, batch_size=2,
                seed=42, data_sha256=file_hash(run/'selected_indices.npz'))
    data = (torch.eye(8), torch.arange(8)%2, torch.arange(8))
    for split in ('train', 'val'):
        torch.save(dict(zip(('features', 'labels', 'indices'), data)), trial.parent/f'{split}_features.pt')
    head = nn.Linear(8, 2)
    optimizer = make_optimizer([dict(params=head.parameters(), lr=.1)], 'adamw', 0.)
    original = experiment.pass_epoch
    count = 0

    def interrupt(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError('interrupted')
        return original(*args, **kwargs)

    monkeypatch.setattr(experiment, 'pass_epoch', interrupt)
    with pytest.raises(RuntimeError, match='interrupted'):
        experiment.fit(head, experiment.tensor_loader(data, 2, 42, True),
                       experiment.tensor_loader(data, 2, 42), optimizer, 3,
                       torch.device('cpu'), trial, meta)
    monkeypatch.setattr(experiment, 'pass_epoch', original)
    monkeypatch.setattr(models, 'build_model', lambda *a, **k: nn.ModuleDict({'fc': nn.Linear(8, 2)}))
    monkeypatch.setattr(models, 'classifier', lambda model, name: model['fc'])
    monkeypatch.setattr(models, 'configure_trainable', lambda *a, **k: None)
    cli._resume_trial(run, 'resnet18/head_trial_00', cfg, torch.device('cpu'))
    assert json.loads((trial/'training_summary.json').read_text())['epochs_completed'] == 3
    assert json.loads((run/'status.json').read_text())['status'] == 'failed'
    assert not (run/'comparison.csv').exists()
    with pytest.raises(ValueError, match='directly inside'):
        cli._resume_trial(run, '../outside', cfg, torch.device('cpu'))


@pytest.mark.parametrize('name', ['adamw','sgd'])
def test_one_batch_overfit_with_configurable_optimizer(name):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model=nn.Linear(16,4)
    x=torch.eye(16)[:8]*4
    y=torch.arange(8)%4
    optimizer=make_optimizer([dict(params=model.parameters(),lr=.2 if name=='adamw' else .5)],name,0)
    first=nn.functional.cross_entropy(model(x),y).item()
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        loss=nn.functional.cross_entropy(model(x),y)
        loss.backward(); optimizer.step()
    final=nn.functional.cross_entropy(model(x),y).item()
    assert final < .02 and final < first/20
    assert (model(x).argmax(1)==y).all()


@pytest.mark.parametrize('name', ['resnet18','densenet121'])
def test_real_model_two_block_one_batch_overfit(name):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model=build_model(name,pretrained=False)
    configure_trainable(model,name,'finetune','last_two_blocks')
    training_mode(model)
    x=torch.randn(2,3,64,64)
    y=torch.tensor([0,1])
    optimizer=make_optimizer([dict(params=[p for p in model.parameters() if p.requires_grad],lr=.001)],'adamw',0)
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        loss=nn.functional.cross_entropy(model(x),y)
        loss.backward(); optimizer.step()
    with torch.no_grad():
        logits=model(x)
    assert nn.functional.cross_entropy(logits,y).item() < .05
    assert (logits.argmax(1)==y).all()


def test_search_resets_same_head_and_does_not_touch_test(tmp_path,monkeypatch):
    import src.experiment as exp
    import numpy as np
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model=build_model('resnet18',pretrained=False)
    initial=copy.deepcopy(model.state_dict())
    folder=tmp_path/'resnet18'; folder.mkdir()
    (folder/'selection.json').write_text(json.dumps(dict(head={'trial':0})))
    np.savez(tmp_path/'selected_indices.npz',train=np.arange(2),val=np.arange(2),test=np.arange(2))
    seen=[]
    def loader(data_dir,split,indices,*args,**kwargs):
        assert split in ('train','val'), 'test must not be loaded during search'
        return DataLoader(TensorDataset(torch.randn(2,3,64,64),torch.tensor([0,1]),torch.arange(2)),batch_size=2)
    def fake_fit(model,train,val,optimizer,epochs,device,folder,meta,*args,**kwargs):
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,initial[key],atol=0,rtol=0)
        seen.append(meta['id'])
        # Deliberately mutate both weights and a BN buffer to catch incomplete resets.
        with torch.no_grad():
            model.fc.weight.add_(1.)
            model.bn1.running_mean.add_(2.)
        folder.mkdir()
        result=dict(best_val_accuracy=.8,best_val_loss=.5 if len(seen)==2 else .6,
                    train_seconds=2.,best_epoch=1)
        torch.save(dict(state_dict=model.state_dict(),metadata=meta,best_epoch=1),folder/'best.pt')
        (folder/'training_summary.json').write_text(json.dumps(result))
        (folder/'config.json').write_text(json.dumps(meta))
        (folder/'history.csv').write_text('epoch,val_accuracy,val_loss\n1,.8,.5\n')
        return result
    monkeypatch.setattr(exp,'image_loader',loader)
    monkeypatch.setattr(exp,'fit',fake_fit)
    cfg=dict(seed=42,image_size=64,weight_decay=1e-4,finetune_epochs=1,cache_frozen_prefix=False,
        finetune_candidates=[dict(id='a',optimizer='adamw',backbone_lr=1e-5,head_lr=1e-4),
            dict(id='b',optimizer='sgd',backbone_lr=1e-3,head_lr=1e-2,unfreeze='last_two_blocks')])
    result,meta,total=exp.search_finetune(model,'resnet18',cfg,dict(train=np.arange(2),val=np.arange(2)),
        tmp_path,tmp_path,torch.device('cpu'),None,None,2)
    assert seen==['a','b'] and result['id']=='b' and total==4.
    assert meta['unfreeze']=='last_two_blocks'
    saved=torch.load(folder/'finetune/best.pt',weights_only=True)
    torch.testing.assert_close(saved['state_dict']['fc.weight'],model.fc.weight)


def test_costs_count_entire_fine_search_once_and_selected_recipe_separately(tmp_path):
    from src.costs import cost_table
    import pandas as pd
    folder=tmp_path/'resnet18/head_selected'; folder.mkdir(parents=True)
    (folder/'training_summary.json').write_text(json.dumps(dict(feature_extraction_train_val_seconds=10,
        all_search_training_seconds=20,train_seconds=5)))
    table=pd.DataFrame([dict(model='resnet18',stage='finetune',train_seconds=7,
        all_finetune_search_seconds=30,test_seconds=3,trainable=100,execution='full_image_forward')])
    cost=cost_table(tmp_path,table).iloc[0]
    assert cost.pipeline_training_seconds==60
    assert cost.selected_recipe_training_seconds==22


@pytest.mark.parametrize('regularized', [False, True])
def test_training_search_smoke_and_reporting_without_downloads(tmp_path,monkeypatch,regularized):
    """Exercise real fit/predict/promotion/cost reporting on a small synthetic batch."""
    import src.experiment as exp
    import numpy as np
    import pandas as pd
    from src.reporting import summarize_suite
    torch.set_num_threads(2)
    torch.manual_seed(42)
    images=torch.randn(4,3,64,64)
    labels=torch.tensor([0,1,0,1])
    indices=torch.arange(4)
    augmentation_calls = []
    def loader(data_dir,split,ids,name,batch,seed,shuffle=False,image_size=64,augmentation=False):
        if augmentation:
            assert split == 'train'
            augmentation_calls.append(split)
        return DataLoader(TensorDataset(images,labels,indices),batch_size=batch,shuffle=shuffle,
                          generator=torch.Generator().manual_seed(seed))
    monkeypatch.setattr(exp,'image_loader',loader)
    monkeypatch.setattr(exp,'build_model',lambda name,pretrained=True,**kwargs:build_model(name,pretrained=False,**kwargs))
    monkeypatch.setattr(exp,'export_evaluation',lambda folder,p,y,i,data_dir:dict(test_samples=len(y),
        accuracy=float((p.argmax(1)==y).mean()),loss=1.,macro_f1=.1,macro_precision=.1,macro_recall=.1))
    cfg=dict(profile='synthetic_smoke_not_benchmark',dataset='CIFAR10',models=['resnet18'],seed=42,
        train_limit=4,val_limit=4,test_limit=4,image_size=64,feature_batch_size=4,
        learning_rates=[.001],batch_sizes=[4],head_epochs=1,finetune_epochs=1,
        weight_decay=1e-4,finetune_backbone_lr=1e-5,finetune_head_lr=1e-4,cache_frozen_prefix=False,
        finetune_candidates=[dict(id='a',backbone_lr=1e-5,head_lr=1e-4),
            dict(id='b',optimizer='sgd',backbone_lr=1e-3,head_lr=1e-2,unfreeze='last_two_blocks',scheduler='cosine')])
    if regularized:
        cfg.update(dropout=.2, train_augmentation=True,
                   finetune_candidates=[dict(id='small_tail',backbone_lr=1e-5,head_lr=1e-4,unfreeze='last_units')])
    selected={s:np.arange(4) for s in ['train','val','test']}
    np.savez(tmp_path/'selected_indices.npz',**selected)
    rows=exp.run_model('resnet18',cfg,selected,tmp_path,tmp_path,torch.device('cpu'))
    assert bool(augmentation_calls) == regularized
    assert len(rows)==2 and rows[1]['all_finetune_search_seconds']>=rows[1]['train_seconds']
    pd.DataFrame(rows).to_csv(tmp_path/'comparison.csv',index=False)
    summarize_suite(tmp_path,cfg,selected)
    assert (tmp_path/'validation_recommendation.json').exists()
    cost=pd.read_csv(tmp_path/'cost_comparison.csv')
    assert cost.loc[cost.stage=='finetune','all_finetune_trials_fit_seconds'].iloc[0]>0

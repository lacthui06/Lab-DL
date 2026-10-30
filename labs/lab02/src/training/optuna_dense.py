"""Two validation-only studies from one immutable head checkpoint; no augmentation."""
import contextlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import optuna

from ..datasets.classification import image_loader, prepare_indices
from ..utils.artifacts import write_json, file_hash, resolve_data_dir
from ..training.trainer import seed_all, fit, predict, Tee
from ..models.backbones import build_model, configure_trainable, training_mode, parameter_counts
from ..utils.plots import export_evaluation, save_figure


def suggest(trial, mode, cfg):
    lr = trial.suggest_float('backbone_lr', *cfg['lr_range'], log=True)
    head = lr if mode == 'shared' else trial.suggest_float('head_lr', *cfg['lr_range'], log=True)
    return dict(backbone_lr=lr, head_lr=head,
                weight_decay=trial.suggest_float('weight_decay', *cfg['weight_decay_range'], log=True))


def loaders(data, ids, cfg):
    return {s: image_loader(data, s, ids[s], 'densenet121', cfg['batch_size'],
            cfg['seed'], s == 'train', cfg['image_size']) for s in ('train', 'val')}


def run(config_path, work=None, device_name='auto'):
    root = Path(work or Path(__file__).resolve().parents[2]).resolve()
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    checkpoint = Path(cfg['head_checkpoint']) if cfg.get('head_checkpoint') else None
    if checkpoint is not None:
        checkpoint = checkpoint if checkpoint.is_absolute() else root/checkpoint
    data = resolve_data_dir(root)
    # Fail before downloading data/weights or starting a run if handoff is absent.
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f'Original selected head checkpoint required: {checkpoint}')
    if not (data/'cifar-10-batches-py').is_dir():
        raise FileNotFoundError(f'Official CIFAR10 batches required: {data}')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device_name == 'auto' else device_name)
    torch.set_num_threads(cfg['threads'])
    torch.hub.set_dir(str(root/'data/pretrained'))
    seed_all(cfg['seed'])
    initial = torch.load(checkpoint, map_location='cpu', weights_only=True) if checkpoint else None
    if initial is not None and (initial['metadata'].get('model') != 'densenet121' or initial['metadata'].get('stage') != 'head'):
        raise ValueError('Expected full DenseNet121 head-only checkpoint')
    folder = root/'outputs/runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_dense_optuna_'+uuid4().hex[:8])
    folder.mkdir(parents=True)
    write_json(folder/'config.json', cfg)
    write_json(folder/'provenance.json', {'head_sha256':file_hash(checkpoint) if checkpoint else None, 'head_source':str(checkpoint) if checkpoint else 'new head training from ImageNet',
        'source_hashes':{str(p.relative_to(Path(__file__).resolve().parents[1])):file_hash(p) for p in Path(__file__).resolve().parents[1].rglob('*.py')}})
    write_json(folder/'environment.json', {'torch':str(torch.__version__), 'numpy':np.__version__,
        'optuna':optuna.__version__, 'device':str(device)})
    write_json(folder/'status.json', {'status':'running'})
    try:
        with (folder/'console.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(Tee(__import__('sys').stdout, log)), contextlib.redirect_stderr(Tee(__import__('sys').stderr, log)):
            ids = prepare_indices(data, folder, cfg, root/'outputs/eda')
            if initial is None:
                seed_all(cfg['seed'])
                head_model = build_model('densenet121', pretrained=True).to(device)
                configure_trainable(head_model, 'densenet121', 'head')
                batch = loaders(data, ids, cfg)
                head_folder = folder/'head_selected'
                meta = {'model':'densenet121','stage':'head','seed':cfg['seed'],
                    'batch_size':cfg['batch_size'],'lr':cfg.get('head_lr',1e-3),
                    'initialization':'ImageNet pretrained backbone; new CIFAR10 classifier'}
                fit(head_model,batch['train'],batch['val'],
                    torch.optim.AdamW(head_model.classifier.parameters(),lr=meta['lr'],weight_decay=1e-4),
                    cfg.get('head_epochs',10),device,head_folder,meta,training_mode,
                    selection_metric='loss')
                checkpoint=head_folder/'best.pt'
                initial=torch.load(checkpoint,map_location='cpu',weights_only=True)
                provenance=json.loads((folder/'provenance.json').read_text())
                provenance.update(head_source=str(checkpoint),head_sha256=file_hash(checkpoint))
                write_json(folder/'provenance.json',provenance)
                del head_model,batch
            winners = {}
            for mode in ('shared', 'separate'):
                study_dir = folder/mode
                study_dir.mkdir()
                study = optuna.create_study(direction='minimize', study_name=mode,
                    storage='sqlite:///'+(study_dir/'study.sqlite3').as_posix(),
                    sampler=optuna.samplers.TPESampler(seed=cfg['seed'], n_startup_trials=2),
                    pruner=optuna.pruners.NopPruner())
                def objective(trial):
                    seed_all(cfg['seed'])
                    params = suggest(trial, mode, cfg)
                    trial_dir = study_dir/f'trial_{trial.number:02d}'
                    model = build_model('densenet121', pretrained=False).to(device)
                    model.load_state_dict(initial['state_dict'])
                    configure_trainable(model, 'densenet121', 'finetune', 'last_block')
                    head_ids = {id(p) for p in model.classifier.parameters()}
                    optimizer = torch.optim.AdamW([
                        {'params':[p for p in model.parameters() if p.requires_grad and id(p) not in head_ids], 'lr':params['backbone_lr']},
                        {'params':model.classifier.parameters(), 'lr':params['head_lr']}], weight_decay=params['weight_decay'])
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
                    batch = loaders(data, ids, cfg)
                    meta = {'model':'densenet121', 'stage':'finetune', 'mode':mode,
                        'optimizer':'adamw', 'scheduler':'cosine', 'unfreeze':'last_block',
                        'seed':cfg['seed'], 'batch_size':cfg['batch_size'], 'lr':params['head_lr'],
                        'parameter_counts':parameter_counts(model), **params}
                    try:
                        summary = fit(model, batch['train'], batch['val'], optimizer, cfg['epochs'],
                            device, trial_dir, meta, training_mode, scheduler, cfg['early_stopping'], selection_metric='loss')
                        write_json(trial_dir/'status.json', {'status':'completed'})
                        trial.set_user_attr('checkpoint', str(trial_dir/'best.pt'))
                        return summary['best_val_loss']
                    except Exception as error:
                        trial_dir.mkdir(exist_ok=True)
                        write_json(trial_dir/'status.json', {'status':'failed','error':str(error)})
                        raise
                    finally:
                        print('Trial finished:', mode, trial.number, flush=True)
                try:
                    study.optimize(objective, n_trials=cfg['trials_per_study'], catch=(FloatingPointError,))
                finally:
                    study.trials_dataframe().to_csv(study_dir/'trials.csv', index=False)
                winners[mode] = {'trial':study.best_trial.number, 'validation_loss':study.best_value,
                    'params':study.best_params, 'checkpoint':study.best_trial.user_attrs['checkpoint']}
            # Both winners locked before any test evaluation.
            write_json(folder/'best_parameters.json', winners)
            rows=[]
            for mode, winner in winners.items():
                saved=torch.load(winner['checkpoint'], map_location='cpu', weights_only=True)
                model=build_model('densenet121', pretrained=False).to(device)
                model.load_state_dict(saved['state_dict'])
                test=image_loader(data,'test',ids['test'],'densenet121',cfg['batch_size'],cfg['seed'],image_size=cfg['image_size'])
                probabilities, labels, indices=predict(model,test,device)
                output=folder/mode/'evaluation'; output.mkdir()
                metrics=export_evaluation(output,probabilities,labels,indices,data)
                restored=build_model('densenet121',pretrained=False).to(device)
                restored.load_state_dict(saved['state_dict'])
                again, again_labels, again_ids=predict(restored,test,device)
                np.testing.assert_allclose(probabilities,again,rtol=1e-5,atol=1e-6)
                np.testing.assert_array_equal(indices,again_ids)
                loaded_output=output/'loaded'; loaded_output.mkdir()
                restored_metrics=export_evaluation(loaded_output,again,again_labels,again_ids,data)
                write_json(output/'loaded_test_metrics.json',restored_metrics)
                write_json(output/'roundtrip.json',{'samples':len(labels),'max_probability_error':float(np.max(np.abs(probabilities-again))), 'passed':True})
                rows.append({'mode':mode, **metrics, 'validation_loss':winner['validation_loss']})
            pd.DataFrame(rows).to_csv(folder/'comparison.csv', index=False)
            plot_results(folder)
            write_json(folder/'status.json', {'status':'completed'})
            write_json(root/'outputs/selected_optuna.json',{'run':str(folder.relative_to(root))})
    except BaseException:
        write_json(folder/'status.json',{'status':'failed','traceback':traceback.format_exc()})
        raise
    finally:
        write_json(folder/'manifest.json',{str(p.relative_to(folder)):file_hash(p)
            for p in folder.rglob('*') if p.is_file() and p.name!='manifest.json'})
    return folder


def plot_results(folder):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    folder=Path(folder); winners=json.loads((folder/'best_parameters.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for mode,w in winners.items():
        h=pd.read_csv(Path(w['checkpoint']).parent/'history.csv')
        for ax,metric in zip(axes,['loss','accuracy']):
            ax.plot(h.epoch,h['val_'+metric],label=mode+' val')
            ax.plot(h.epoch,h['train_'+metric],'--',label=mode+' train')
            ax.set(xlabel='Epoch',ylabel=metric); ax.legend()
        data=np.load(folder/mode/'evaluation/test_predictions.npz',allow_pickle=False)
        curves,panels=plt.subplots(1,2,figsize=(12,4)); scores=[]
        for c in range(10):
            y=data['labels']==c; p=data['probabilities'][:,c]
            fpr,tpr,_=roc_curve(y,p); precision,recall,_=precision_recall_curve(y,p)
            panels[0].plot(fpr,tpr,label=str(c)); panels[1].plot(recall,precision,label=str(c))
            scores.append({'class':c,'roc_auc':roc_auc_score(y,p),'average_precision':average_precision_score(y,p)})
        panels[0].set(xlabel='FPR',ylabel='TPR');panels[1].set(xlabel='Recall',ylabel='Precision')
        for panel in panels:panel.legend(fontsize=7)
        save_figure(curves,folder/mode/'evaluation/roc_pr.png',dpi=140);plt.close(curves)
        pd.DataFrame(scores).to_csv(folder/mode/'evaluation/roc_pr_metrics.csv',index=False)
    save_figure(fig,folder/'learning_curves.png',dpi=140);plt.close(fig)

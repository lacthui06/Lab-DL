"""Staged two-model suite with cached head features and live epoch artifacts.

Completed stages can be viewed/reused; interrupted trials are not silently resumed.
"""
import contextlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import optuna

from ..datasets.classification import image_loader, prepare_indices
from ..utils.artifacts import write_json, file_hash, resolve_data_dir
from ..training.trainer import seed_all, cached_features, tensor_loader, fit, predict, Tee
from ..models.backbones import build_model, configure_trainable, classifier, parameter_counts, training_mode
from ..utils.plots import export_evaluation, save_figure


def live_epoch(history, folder, state):
    """Persist and refresh loss/accuracy/LR every epoch, without clearing logs."""
    import matplotlib.pyplot as plt
    from IPython.display import display
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    for ax, metric in zip(axes[:2], ['loss', 'accuracy']):
        ax.plot(h.epoch, h['train_'+metric], label='train')
        ax.plot(h.epoch, h['val_'+metric], label='validation')
        ax.set(xlabel='Epoch', ylabel=metric); ax.legend()
    for key in h.columns:
        if key.startswith('lr_group_'):
            axes[2].plot(h.epoch, h[key], label=key)
    axes[2].set(xlabel='Epoch', ylabel='LR'); axes[2].legend()
    fig.suptitle(f'{folder.parent.name}/{folder.name} | epoch {len(h)} | stale {state["stale"]}')
    save_figure(fig, folder/'learning_curves.png', dpi=120)
    key = str(folder.resolve())
    handles = getattr(live_epoch, '_handles', {})
    if key in handles:
        handles[key].update(fig)
    else:
        handles[key] = display(fig, display_id=True)
    live_epoch._handles = handles
    plt.close(fig)


class Suite:
    def __init__(self, root, config_path=None, run_path=None, device='auto', live=True):
        self.root = Path(root).resolve()
        self.cfg = json.loads(Path(config_path or self.root/'config/standard.json').read_text())
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
        self.data = resolve_data_dir(self.root)
        if not (self.data/'cifar-10-batches-py').is_dir():
            raise FileNotFoundError('Provide official CIFAR10 via LAB02_DATA before creating suite')
        self.callback = live_epoch if live else None
        torch.set_num_threads(self.cfg['threads'])
        torch.hub.set_dir(str(self.root/'data/pretrained'))
        if run_path:
            self.run = Path(run_path).resolve()
            if json.loads((self.run/'config.json').read_text()) != self.cfg:
                raise ValueError('Existing run config differs; select matching config or new run')
        else:
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            self.run = self.root/'outputs/runs'/f'{stamp}_standard_{uuid4().hex[:8]}'
            self.run.mkdir(parents=True)
            write_json(self.run/'config.json', self.cfg)
            write_json(self.run/'environment.json', {'torch':str(torch.__version__), 'optuna':optuna.__version__,
                'numpy':np.__version__, 'device':str(self.device)})
            write_json(self.run/'source_hashes.json', {str(p.relative_to(Path(__file__).resolve().parents[1])):file_hash(p) for p in Path(__file__).resolve().parents[1].rglob('*.py')})
            write_json(self.run/'status.json', {'status':'prepared'})
        self.ids = prepare_indices(self.data, self.run, self.cfg, self.root/'outputs/eda')

    @contextlib.contextmanager
    def logged(self, name):
        try:
            with (self.run/'console.log').open('a', encoding='utf-8') as log:
                with contextlib.redirect_stdout(Tee(sys.stdout,log)), contextlib.redirect_stderr(Tee(sys.stderr,log)):
                    print('Stage:', name, flush=True)
                    yield
        except BaseException as error:
            write_json(self.run/'status.json', {'status':'failed','stage':name,'error':str(error)})
            raise
        finally:
            self.manifest()

    def manifest(self):
        write_json(self.run/'manifest.json', {str(p.relative_to(self.run)):file_hash(p)
            for p in self.run.rglob('*') if p.is_file() and p.name!='manifest.json'})

    def head(self, name):
        folder = self.run/name/'head_selected'
        if (folder/'stage_completed.json').exists():
            return folder
        if folder.exists():
            raise RuntimeError('Interrupted head stage; use a new suite, not silent overwrite')
        with self.logged(name+'/head'):
            seed_all(self.cfg['seed'])
            model = build_model(name).to(self.device)
            configure_trainable(model,name,'head')
            base = self.run/name; base.mkdir(exist_ok=True)
            (base/'architecture.txt').write_text(str(model))
            records = []
            for split in ('train','val'):
                data, seconds = cached_features(model,name,self.data,split,self.ids[split],self.cfg,
                    self.device,base/f'{split}_features.pt')
                records.append(data)
                write_json(base/f'{split}_feature_timing.json',{'seconds':seconds,'samples':len(data[1])})
            head = classifier(model,name)
            optimizer = torch.optim.AdamW(head.parameters(),lr=self.cfg['head_lr'],weight_decay=self.cfg['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=self.cfg['head_epochs'])
            meta = {'model':name,'stage':'head','lr':self.cfg['head_lr'],'batch_size':self.cfg['batch_size'],
                'parameter_counts':parameter_counts(model),'cached_features':True}
            fit(head,tensor_loader(records[0],self.cfg['batch_size'],self.cfg['seed'],True),
                tensor_loader(records[1],self.cfg['batch_size'],self.cfg['seed']),optimizer,
                self.cfg['head_epochs'],self.device,folder,meta,scheduler=scheduler,
                selection_metric='loss',epoch_callback=self.callback)
            chosen = torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
            classifier(model,name).load_state_dict(chosen['state_dict'])
            chosen['state_dict'] = {k:v.detach().cpu() for k,v in model.state_dict().items()}
            torch.save(chosen,folder/'best.pt')
            write_json(folder/'stage_completed.json',{'status':'completed','checkpoint_schema':'full model'})
            print('Head checkpoint locked:',folder,flush=True)
        return folder

    def search(self, name):
        folder = self.run/name
        selection = folder/'selection.json'
        if selection.exists():
            return json.loads(selection.read_text())
        head = folder/'head_selected/best.pt'
        if not head.exists():
            raise FileNotFoundError('Complete head stage first')
        if (folder/'study.sqlite3').exists():
            raise RuntimeError('Interrupted study; no automatic resume. Use a new suite or inspect retained artifacts.')
        with self.logged(name+'/search'):
            initial = torch.load(head,map_location='cpu',weights_only=True)
            study=optuna.create_study(direction='minimize',study_name=name,
                storage='sqlite:///'+(folder/'study.sqlite3').as_posix(),
                sampler=optuna.samplers.TPESampler(seed=self.cfg['seed'],n_startup_trials=2),
                pruner=optuna.pruners.NopPruner())
            def objective(trial):
                seed_all(self.cfg['seed'])
                lr=trial.suggest_float('lr',*self.cfg['lr_range'],log=True)
                model=build_model(name,pretrained=False).to(self.device)
                model.load_state_dict(initial['state_dict'])
                configure_trainable(model,name,'finetune','last_block')
                batch={s:image_loader(self.data,s,self.ids[s],name,self.cfg['batch_size'],self.cfg['seed'],
                    s=='train',self.cfg['image_size']) for s in ('train','val')}
                trial_dir=folder/f'trial_{trial.number:02d}'
                optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=self.cfg['weight_decay'])
                scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=self.cfg['finetune_epochs'])
                meta={'model':name,'stage':'finetune','lr':lr,'batch_size':self.cfg['batch_size'],
                    'unfreeze':'last_block','optimizer':'adamw','scheduler':'cosine','parameter_counts':parameter_counts(model),
                    'head_sha256':file_hash(head),'weight_decay':self.cfg['weight_decay']}
                summary=fit(model,batch['train'],batch['val'],optimizer,self.cfg['finetune_epochs'],self.device,
                    trial_dir,meta,training_mode,scheduler,self.cfg['early_stopping'],
                    selection_metric='loss',epoch_callback=self.callback)
                trial.set_user_attr('best_epoch',summary['best_epoch'])
                trial.set_user_attr('epochs_completed',summary['epochs_completed'])
                return summary['best_val_loss']
            def completed(study, trial):
                table=study.trials_dataframe();table.to_csv(folder/'trials.csv',index=False)
                from IPython.display import display
                display(table)
            try:
                study.optimize(objective,n_trials=self.cfg['trials_per_model'],callbacks=[completed],catch=(FloatingPointError,))
            finally:
                study.trials_dataframe().to_csv(folder/'trials.csv',index=False)
            best={'trial':study.best_trial.number,'lr':study.best_params['lr'],'validation_loss':study.best_value}
            write_json(selection,best)
        return best

    def evaluate(self, name):
        # Lock all models before touching test; notebook trains both before evaluation.
        if not all((self.run/n/'selection.json').exists() for n in self.cfg['models']):
            raise RuntimeError('Complete both model searches before test')
        output=[]
        with self.logged(name+'/test'):
            selection=json.loads((self.run/name/'selection.json').read_text())
            for stage,checkpoint in [('head',self.run/name/'head_selected/best.pt'),
                ('finetune',self.run/name/f"trial_{selection['trial']:02d}"/'best.pt')]:
                dest=self.run/name/'evaluation'/stage
                if (dest/'roundtrip.json').exists():
                    output.append({'model':name,'stage':stage,**json.loads((dest/'metrics.json').read_text())});continue
                dest.mkdir(parents=True,exist_ok=True)
                saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
                model=build_model(name,pretrained=False).to(self.device);model.load_state_dict(saved['state_dict'])
                loader=image_loader(self.data,'test',self.ids['test'],name,self.cfg['batch_size'],self.cfg['seed'],image_size=self.cfg['image_size'])
                probs,labels,indices=predict(model,loader,self.device)
                metrics=export_evaluation(dest,probs,labels,indices,self.data)
                restored=build_model(name,pretrained=False).to(self.device);restored.load_state_dict(saved['state_dict'])
                again,again_labels,again_indices=predict(restored,loader,self.device)
                np.testing.assert_allclose(probs,again,atol=1e-6,rtol=1e-5)
                np.testing.assert_array_equal(indices,again_indices);np.testing.assert_array_equal(labels,again_labels)
                restored_dir=dest/'loaded';restored_dir.mkdir(exist_ok=True)
                export_evaluation(restored_dir,again,again_labels,again_indices,self.data)
                ovr(dest,probs,labels)
                write_json(dest/'roundtrip.json',{'passed':True,'samples':len(labels),'max_probability_error':float(np.max(np.abs(probs-again)))})
                output.append({'model':name,'stage':stage,**metrics})
        return pd.DataFrame(output)


def ovr(folder, probs, labels):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    from ..datasets.eda import CLASSES
    fig,axes=plt.subplots(1,2,figsize=(12,4));rows=[]
    for c,label in enumerate(CLASSES):
        y=labels==c;p=probs[:,c]
        fpr,tpr,_=roc_curve(y,p);precision,recall,_=precision_recall_curve(y,p)
        axes[0].plot(fpr,tpr,label=label);axes[1].plot(recall,precision,label=label)
        pd.DataFrame({'fpr':fpr,'tpr':tpr}).to_csv(folder/f'roc_{c}.csv',index=False)
        pd.DataFrame({'recall':recall,'precision':precision}).to_csv(folder/f'pr_{c}.csv',index=False)
        rows.append({'class':label,'roc_auc':roc_auc_score(y,p),'average_precision':average_precision_score(y,p)})
    axes[0].set(xlabel='FPR',ylabel='TPR');axes[1].set(xlabel='Recall',ylabel='Precision')
    for ax in axes:ax.legend(fontsize=7)
    save_figure(fig,folder/'roc_pr.png',dpi=130);plt.close(fig)
    pd.DataFrame(rows).to_csv(folder/'roc_pr_metrics.csv',index=False)

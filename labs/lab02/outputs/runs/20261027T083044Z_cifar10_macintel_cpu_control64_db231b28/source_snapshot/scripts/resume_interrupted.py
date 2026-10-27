"""Resume between architectures, or explicitly resume one trial at an epoch boundary.

Refuse partial architectures instead of overwriting their trials/checkpoints.
"""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',help='Run directory, absolute or relative to the lab')
    parser.add_argument('--cached-prefix',action='store_true')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--trial', help='Relative trial directory, e.g. resnet18/head_trial_00; resume fit only')
    args=parser.parse_args(argv)
    import contextlib
    import json
    import shutil
    import traceback
    import numpy as np
    import pandas as pd
    import torch
    from src.experiment import run_model,Tee
    from src.delivery import resolve_run,resolve_data_dir,write_json,file_hash
    from src.reporting import summarize_suite
    from src.tuning import fine_candidates

    run=resolve_run(ROOT,args.run)
    cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
    if cfg.get('dataset')!='CIFAR10': raise ValueError('Legacy runs require their source snapshot')
    if json.loads((run/'status.json').read_text(encoding='utf-8'))['status']=='completed':
        raise SystemExit('Run already completed')
    if args.cached_prefix: cfg['cache_frozen_prefix']=True
    fine_candidates(cfg)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if cfg.get('require_cuda') and device.type!='cuda': raise ValueError('Tuning requires CUDA')
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    environment=json.loads((run/'environment.json').read_text(encoding='utf-8'))
    if environment['device']!=str(device):
        raise ValueError('Resume device differs from original run; use matching --device to preserve timing comparability')
    if args.trial:
        if args.cached_prefix:
            raise ValueError('Trial resume cannot change cache policy')
        _resume_trial(run, args.trial, cfg, device)
        return
    work=run.parents[2]
    data_dir=resolve_data_dir(work)
    table=pd.read_csv(run/'comparison.csv') if (run/'comparison.csv').exists() else pd.DataFrame()
    for name in cfg['models']:
        done=len(table) and len(table[table.model==name])==2
        folder=run/name
        if not done and folder.exists() and any(folder.iterdir()):
            raise RuntimeError(f'Partial architecture {name}; this command only resumes at architecture boundaries')
    torch.set_num_threads(cfg['threads'])
    torch.hub.set_dir(str(work/'data/pretrained'))
    selected=dict(np.load(run/'selected_indices.npz',allow_pickle=False))
    rows=table.to_dict('records')
    if args.cached_prefix: write_json(run/'resume_config.json',cfg)
    write_json(run/'status.json',dict(status='running',resumed=True))
    write_json(run/'resume_note.json',dict(reason='Continue at architecture boundary',completed_architectures_preserved=True))
    try:
        with (run/'console.log').open('a',encoding='utf-8') as log:
            with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
                for name in cfg['models']:
                    if len(table) and len(table[table.model==name])==2: continue
                    rows.extend(run_model(name,cfg,selected,data_dir,run,device))
                    pd.DataFrame(rows).to_csv(run/'comparison.csv',index=False)
                summarize_suite(run,cfg,selected)
                write_json(run/'status.json',dict(status='completed',resumed=True))
                write_json(work/'outputs/selected_run.json',dict(run=str(run)))
    except BaseException:
        write_json(run/'status.json',dict(status='failed',traceback=traceback.format_exc()))
        raise
    finally:
        hashes={}
        for section in ['src','scripts','tests','config']:
            for source in (ROOT/section).glob('*'):
                if not source.is_file() or source.suffix not in ('.py','.json'): continue
                relative=source.relative_to(ROOT)
                destination=run/'resume_source_snapshot'/relative
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,destination)
                hashes[str(relative)]=file_hash(source)
        write_json(run/'resume_source_hashes.json',hashes)
        write_json(run/'manifest.json',{str(p.relative_to(run)):file_hash(p) for p in run.rglob('*')
            if p.is_file() and p.name!='manifest.json'})


def _resume_trial(run, relative, cfg, device):
    """Restore only a future-schema trial; never relabel an incomplete suite completed."""
    import json
    import numpy as np
    import torch
    from src.data import image_loader
    from src.delivery import file_hash, resolve_data_dir, write_json
    from src.experiment import fit, tensor_loader
    from src.models import build_model, classifier, configure_trainable, training_mode, CachedTail
    from src.tuning import make_optimizer, make_scheduler, stopping_policy

    folder = (run/relative).resolve()
    if run.resolve() not in folder.parents or folder.parent.parent != run.resolve():
        raise ValueError('Trial must be directly inside a model directory of this run')
    saved = torch.load(folder/'last.pt', map_location='cpu', weights_only=True)
    meta = saved['signature']['metadata']
    name, stage = meta['model'], meta['stage']
    if folder.parent.name != name or stage not in ('head', 'finetune'):
        raise ValueError('Trial model/stage metadata differs from its path')
    if meta['data_sha256'] != file_hash(run/'selected_indices.npz'):
        raise ValueError('Selected data indices changed since training')
    for relative_source, digest in json.loads((run/'source_hashes.json').read_text()).items():
        if file_hash(ROOT/relative_source) != digest:
            raise ValueError(f'Source changed since training: {relative_source}; use original source')
    torch.set_num_threads(cfg['threads'])
    model = build_model(name, pretrained=False).to(device)
    configure_trainable(model, name, stage, meta.get('unfreeze', 'last_block'))
    data_dir = resolve_data_dir(run.parents[2])
    with np.load(run/'selected_indices.npz', allow_pickle=False) as data:
        selected = {key: data[key] for key in data.files}
    loaders = []
    for split in ('train', 'val'):
        if stage == 'head' or meta.get('execution') == 'cached_frozen_prefix':
            cached = torch.load(folder.parent/f'{split}_features.pt', map_location='cpu', weights_only=True)
            if not np.array_equal(cached['indices'].numpy(), selected[split]):
                raise ValueError(f'{split} feature cache indices differ from selected indices')
            values = cached['features'] if stage == 'head' else torch.from_numpy(
                np.load(folder.parent/f'{split}_prefix.npy', mmap_mode='c'))
            loader = tensor_loader((values, cached['labels'], cached['indices']), meta['batch_size'],
                                  meta['seed'], split == 'train')
        else:
            loader = image_loader(data_dir, split, selected[split], name, meta['batch_size'],
                                  meta['seed'], split == 'train', meta['image_size'])
        loaders.append(loader)
    if stage == 'head':
        target = classifier(model, name)
        groups = [dict(params=target.parameters(), lr=meta['lr'])]
    else:
        head_ids = {id(p) for p in classifier(model, name).parameters()}
        groups = [dict(params=[p for p in model.parameters() if p.requires_grad and id(p) not in head_ids],
                       lr=meta['backbone_lr']),
                  dict(params=classifier(model, name).parameters(), lr=meta['lr'])]
        target = CachedTail(model, name) if meta.get('execution') == 'cached_frozen_prefix' else model
    epochs = cfg['head_epochs'] if stage == 'head' else cfg['finetune_epochs']
    optimizer = make_optimizer(groups, meta['optimizer'], cfg['weight_decay'], cfg.get('momentum', .9))
    scheduler = make_scheduler(optimizer, meta.get('scheduler', 'none'), epochs)
    try:
        result = fit(target, *loaders, optimizer, epochs, device, folder, meta,
                     training_mode if stage == 'finetune' else None, scheduler=scheduler,
                     stopping=stopping_policy(cfg) if stage == 'finetune' else None, resume=True)
        print(f"Trial fit completed: {result['epochs_completed']} epochs. "
              'Suite selection/test/report have NOT been continued or marked completed.')
    finally:
        write_json(run/'manifest.json', {str(p.relative_to(run)):file_hash(p) for p in run.rglob('*')
            if p.is_file() and p.name != 'manifest.json'})


if __name__=='__main__':
    main()

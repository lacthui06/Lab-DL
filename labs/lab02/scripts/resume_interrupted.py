"""Resume only between architectures; called manually as documented in README.

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


if __name__=='__main__':
    main()

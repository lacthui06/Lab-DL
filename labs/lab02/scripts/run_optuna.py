"""Explicit Optuna execution; reading notebooks never starts training."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config',default=str(ROOT/'config/optuna_dense.json'))
parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
parser.add_argument('--dry-run',action='store_true')
args=parser.parse_args()
if args.dry_run:
    cfg=json.loads(Path(args.config).read_text())
    print(json.dumps({'config':cfg,'studies':['shared','separate'],
        'total_trials':2*cfg['trials_per_study'],'training_started':False},indent=2))
else:
    from src.optuna_dense import run
    print(run(args.config,device_name=args.device))

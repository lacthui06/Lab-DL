"""Tạo preview augmentation và lưu artifact trước khi chạy GPU training."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.workflow import read_config
from src.datasets.loader import load_data
from src.datasets.augmentation import augment_train,export_augmentation
from src.utils.runs import ROOT,run_context

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'config/frozen_trial.json')
    args=parser.parse_args();cfg=read_config(args.config)
    raw=load_data(cfg['exercise2']);_,summary=augment_train(raw,cfg['exercise2']['augmentation'],cfg['seed'])
    with run_context('augmentation_preview',cfg) as out:
        export_augmentation(summary,out)
        print('Official train:',summary['original_train_size'])
        print('Synthetic added:',summary['added_count'])
        print('Effective train:',summary['effective_train_size'])
        print('Validation:',len(raw['validation']),'Test:',len(raw['test']))

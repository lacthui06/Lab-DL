"""EDA/preprocessing/features train-only, lưu bảng và hình dùng trong notebook/report."""
from pathlib import Path
import argparse
import sys
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.workflow import read_config
from src.datasets.loader import load_data
from src.utils.runs import ROOT,run_context
from src.data_preparation import export_data_review

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'config/frozen_trial.json')
    args=parser.parse_args();cfg=read_config(args.config);settings=cfg['exercise2']
    with run_context('dataset_preview',cfg) as out:
        raw=load_data(settings)
        tokenizer=AutoTokenizer.from_pretrained(settings['model_name'],revision=settings['model_revision'])
        statistics,features=export_data_review(raw,tokenizer,settings,out)
        print('Official splits:',{name:len(ds) for name,ds in raw.items()})
        print('Preprocessing:',statistics)
        print('Train TF-IDF shape:',features['train_shape'])

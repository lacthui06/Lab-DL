"""EDA dữ liệu gốc trước augmentation; lưu bảng/hình để hiển thị ngay trong notebook."""
from pathlib import Path
import argparse
import sys
import numpy as np
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from labs.lab03.src.workflow import read_config
from labs.lab03.src.datasets.loader import load_data
from labs.lab03.src.utils.runs import ROOT,run_context,save_json
from labs.lab03.src.utils.plots import plot_lengths,finish
from labs.lab03.src.analysis import _table_image

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'config/augmented.json')
    args=parser.parse_args();cfg=read_config(args.config);settings=cfg['exercise2']
    with run_context('dataset_preview',cfg) as out:
        raw=load_data(settings)
        tokenizer=AutoTokenizer.from_pretrained(settings['model_name'],revision=settings['model_revision'])
        lengths=[len(ids) for ids in tokenizer(raw['train']['text'],truncation=False)['input_ids']]
        counts={split:{str(c):int(sum(int(y)==c for y in raw[split]['label'])) for c in (0,1)} for split in raw}
        examples=[]
        for label in (0,1):
            indices=[i for i,y in enumerate(raw['train']['label']) if y==label][:2]
            examples.extend({'index':i,'text':raw['train'][i]['text'],'label':label} for i in indices)
        payload={'split_sizes':{split:len(ds) for split,ds in raw.items()},'label_counts':counts,
            'train_token_lengths':{'mean':float(np.mean(lengths)),'max':max(lengths),
                'truncated_fraction':float(np.mean(np.asarray(lengths)>settings['max_length']))},
            'examples':examples,'examples_split':'train','augmentation_applied':False}
        save_json(out/'data_preview.json',payload)
        plot_lengths(lengths,settings['max_length'],out/'token_length_distribution.png')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,len(counts),figsize=(5*len(counts),4),squeeze=False)
        for ax,(split,support) in zip(axes[0],counts.items()):
            bars=ax.bar(['Negative','Positive'],[support['0'],support['1']]);ax.bar_label(bars)
            ax.set(title=split,ylabel='Official examples')
        finish(fig,out/'class_distribution.png')
        _table_image(examples,['index','text','label'],out/'train_examples.png','Original train examples; heldout texts not displayed',widths=[.08,.84,.08])
        print('Official splits:',payload['split_sizes']);print('Train token lengths:',payload['train_token_lengths'])

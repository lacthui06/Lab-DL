"""Notebook Kaggle phải có 10 phase riêng và EDA thực, không giấu mọi output ở cell cuối."""
import ast
import json
import runpy
import sys
from datasets import Dataset,DatasetDict
import nbformat
from labs.lab03.src.utils.runs import ROOT,run_context


def test_augmented_notebook_has_ten_explained_code_cells_and_single_training_invocation():
    n=nbformat.read(ROOT/'notebooks/07_kaggle_augmented_epoch_training.ipynb',as_version=4)
    nbformat.validate(n)
    code=[c for c in n.cells if c.cell_type=='code']
    assert len(code)==10 and n.metadata['lab03']['code_cell_count']==10
    for i,c in enumerate(n.cells):
        if c.cell_type=='code':
            ast.parse(c.source)
            assert n.cells[i-1].cell_type=='markdown'
    combined='\n'.join(c.source for c in code)
    assert combined.count('run_python("labs/lab03/scripts/run_study.py"')==1
    assert 'preview_dataset.py' in code[4].source
    assert 'preview_augmentation.py' in code[5].source
    assert 'epoch_history.json' in code[6].source
    assert 'frozen_selection.json' in code[7].source
    assert 'test_comparison.json' in code[8].source
    assert 'package_study.py' in code[9].source
    assert 'show_figures' in code[3].source and 'show_figures' in code[4].source
    assert n.metadata['lab03']['execution_mode']=='train'
    helper=(ROOT/'scripts/kaggle_augmented_epoch_cells.py').read_text()
    assert helper.count('# %%\n')==10
    assert all(c.source in helper for c in code)


def test_dataset_preview_persists_original_counts_and_three_figures(tiny,tmp_path,monkeypatch):
    _,tokenizer=tiny
    raw=Dataset.from_dict({'text':['excellent movie','terrible film']*2,'label':[1,0]*2})
    ds=DatasetDict(train=raw,validation=raw,test=raw)
    monkeypatch.setattr('labs.lab03.src.datasets.loader.load_data',lambda cfg:ds)
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained',lambda *a,**k:tokenizer)
    monkeypatch.setattr('labs.lab03.src.utils.runs.run_context',lambda kind,cfg:run_context(kind,cfg,root=tmp_path))
    monkeypatch.setattr(sys,'argv',['preview_dataset.py','--config',str(ROOT/'config/augmented.json')])
    runpy.run_path(str(ROOT/'scripts/preview_dataset.py'),run_name='__main__')
    state=json.loads((tmp_path/'config/workflow.json').read_text())
    out=tmp_path/'outputs/runs'/state['active_runs']['dataset_preview']
    preview=json.loads((out/'data_preview.json').read_text())
    assert preview['split_sizes']=={'train':4,'validation':4,'test':4}
    assert preview['examples_split']=='train' and preview['augmentation_applied'] is False
    assert preview['label_counts']['train']=={'0':2,'1':2}
    assert len(list(out.glob('*.png')))==3


def test_frozen_notebook_preserves_ten_phases_and_download_routes():
    notebook=nbformat.read(ROOT/'notebooks/08_kaggle_frozen_encoder_trial.ipynb',as_version=4)
    nbformat.validate(notebook)
    cells=[c for c in notebook.cells if c.cell_type=='code']
    assert len(cells)==10
    for c in cells:ast.parse(c.source)
    assert sum('run_frozen_trial.py' in c.source for c in cells)==1
    assert '/files/' in cells[-1].source and 'bundle_manifest.json' in cells[-1].source
    helper=(ROOT/'scripts/kaggle_frozen_trial_cells.py').read_text()
    assert all(c.source in helper for c in cells)

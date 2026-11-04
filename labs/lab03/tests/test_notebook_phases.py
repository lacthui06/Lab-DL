"""Hai notebook phải phân vai rõ, có view/train và cell code hợp lệ."""
import ast
import json
import runpy
import sys
from datasets import Dataset,DatasetDict
import nbformat
from labs.lab03.src.utils.runs import ROOT,run_context


def test_only_two_notebooks_cover_preparation_and_modeling():
    paths=sorted((ROOT/'notebooks').glob('*.ipynb'))
    assert [p.name for p in paths]==['01_eda_preprocessing_feature_engineering.ipynb','02_modeling.ipynb']
    for path in paths:
        notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
        assert 'Module/file' in notebook.cells[0].source
        codes=[c.source for c in notebook.cells if c.cell_type=='code']
        for source in codes:ast.parse(source)
        assert sum('MODE = "view"' in source for source in codes)==1
        assert 'session.setup()' in codes[0]
    preparation=nbformat.read(paths[0],as_version=4)
    modeling=nbformat.read(paths[1],as_version=4)
    assert 'session.prepare()' in '\n'.join(c.source for c in preparation.cells)
    assert 'session.train()' not in '\n'.join(c.source for c in preparation.cells)
    source='\n'.join(c.source for c in modeling.cells)
    assert source.count('session.train()')==1
    assert 'session.preflight()' in source and 'session.export()' in source


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

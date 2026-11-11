"""Ba notebook phân vai rõ ràng (EDA, Feature Engineering, Modeling), có view/train và cell code hợp lệ."""
import ast
import json
import runpy
import sys
from datasets import Dataset,DatasetDict
import nbformat
from labs.lab03.src.utils.runs import ROOT,run_context


def test_three_notebooks_cover_eda_feature_and_modeling():
    paths=sorted((ROOT/'notebooks').glob('*.ipynb'))
    assert [p.name for p in paths]==['01_eda.ipynb','02_feature_engineering.ipynb','03_modeling.ipynb']
    for path in paths:
        notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
        first_cell_src = notebook.cells[0].source
        assert 'Pipeline Execution Roadmap' in first_cell_src or 'Bảng Lộ Trình' in first_cell_src or 'Module' in first_cell_src
        codes=[c.source for c in notebook.cells if c.cell_type=='code']
        for source in codes:ast.parse(source)
        assert sum(any(f'MODE = "{mode}"' in source for mode in ('view', 'train', 'prepare')) for source in codes)==1
        assert 'session.setup()' in codes[0]
    eda=nbformat.read(paths[0],as_version=4)
    fe=nbformat.read(paths[1],as_version=4)
    modeling=nbformat.read(paths[2],as_version=4)
    assert 'session.prepare()' in '\n'.join(c.source for c in eda.cells)
    assert 'session.train()' not in '\n'.join(c.source for c in eda.cells)
    assert 'session.train()' not in '\n'.join(c.source for c in fe.cells)
    source='\n'.join(c.source for c in modeling.cells)
    assert source.count('session.train()')==1
    assert 'session.preflight()' in source and 'session.export()' in source
    assert len(eda.cells)>=10 and len(fe.cells)>=10 and len(modeling.cells)>=20
    for notebook in (eda,fe,modeling):
        assert all(not any(o.output_type == 'error' for o in c.outputs) for c in notebook.cells if c.cell_type=='code')
        text='\n'.join(c.source for c in notebook.cells)
        assert '10 epoch' not in text and '10epoch' not in text and 'lịch sử' not in text
        assert 'final_review' not in text and 'config/report.json' not in text


def test_dataset_preview_persists_original_counts_and_three_figures(tiny,tmp_path,monkeypatch):
    _,tokenizer=tiny
    raw=Dataset.from_dict({'text':['excellent movie','terrible film']*2,'label':[1,0]*2})
    ds=DatasetDict(train=raw,validation=raw,test=raw)
    monkeypatch.setattr('labs.lab03.src.datasets.loader.load_data',lambda cfg:ds)
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained',lambda *a,**k:tokenizer)
    monkeypatch.setattr('labs.lab03.src.utils.runs.run_context',lambda kind,cfg:run_context(kind,cfg,root=tmp_path))
    monkeypatch.setattr(sys,'argv',['preview_dataset.py','--config',str(ROOT/'config/frozen_trial.json')])
    runpy.run_path(str(ROOT/'scripts/preview_dataset.py'),run_name='__main__')
    state=json.loads((tmp_path/'config/workflow.json').read_text())
    out=tmp_path/'outputs/runs'/state['active_runs']['dataset_preview']
    preview=json.loads((out/'data_preview.json').read_text())
    assert preview['split_sizes']=={'train':4,'validation':4,'test':4}
    assert preview['examples_split']=='train' and preview['augmentation_applied'] is False
    assert preview['label_counts']['train']=={'0':2,'1':2}
    assert len(list(out.glob('*.png')))>=9
    assert json.loads((out/'feature_summary.json').read_text())['fitted_split']=='train'

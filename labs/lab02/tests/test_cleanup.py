"""Production reference graph, merged CLIs, shared data contract and regressions."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]


def load_cli(name):
    spec=importlib.util.spec_from_file_location('cli_'+name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grouped_modules_preserve_public_imports():
    """Notebook/CLI legacy imports resolve to the same canonical module object."""
    from importlib import import_module
    aliases = {
        'data':'datasets.classification', 'eda':'datasets.eda',
        'experiment':'training.trainer', 'tuning':'training.tuning',
        'standard_suite':'training.standard_suite', 'optuna_dense':'training.optuna_dense',
        'reporting':'utils.plots', 'delivery':'utils.artifacts', 'costs':'utils.costs',
        'notebook_views':'utils.notebook_views', 'merged_results':'utils.merged_results'}
    for old, new in aliases.items():
        assert import_module('src.'+old) is import_module('src.'+new)
    from src.models import build_model
    from src.models.backbones import build_model as canonical
    assert build_model is canonical
    from src.training.trainer import ROOT as training_root
    assert training_root == ROOT
    for relative in aliases.values():
        assert (ROOT/'src'/Path(*relative.split('.'))).with_suffix('.py').exists()


@pytest.mark.parametrize('name',['run','infer','analyze_data','download_assets','execute_notebook','finalize','resume_interrupted'])
def test_cli_help_does_not_execute_work(name):
    result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts'/f'{name}.py'),'--help'],
        cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    assert 'usage:' in result.stdout


def test_cli_modules_are_import_safe(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:pytest.fail('Download during import'))
    for name in ['infer','download_assets','execute_notebook','resume_interrupted','finalize','analyze_data','run']:
        load_cli(name)


def test_chunked_download_verifies_headers_checksum_and_reuses_archive(tmp_path,monkeypatch):
    cli=load_cli('download_assets')
    payload=b'official-archive-fixture'
    monkeypatch.setattr(cli,'SIZE',len(payload))
    monkeypatch.setattr(cli,'MD5',hashlib.md5(payload).hexdigest())
    calls=[]
    class Response:
        status=206
        headers={'Content-Range':f'bytes 0-{len(payload)-1}/{len(payload)}'}
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self): return payload
    monkeypatch.setattr(cli.urllib.request,'urlopen',lambda *a,**k:(calls.append(a) or Response()))
    path=cli.download_archive(tmp_path/'valid')
    assert path.read_bytes()==payload
    assert cli.download_archive(tmp_path/'valid')==path and len(calls)==1
    monkeypatch.setattr(cli,'MD5','wrong')
    with pytest.raises(RuntimeError,match='checksum'):
        cli.download_archive(tmp_path/'bad')
    assert not (tmp_path/'bad/cifar-10-python.tar.gz').exists()


def test_io_helpers_are_lightweight_and_data_paths_are_shared(tmp_path,monkeypatch):
    from src.delivery import resolve_data_dir,resolve_run,write_json,file_hash
    monkeypatch.setenv('LAB02_DATA',str(tmp_path/'external'))
    assert resolve_data_dir(tmp_path)==(tmp_path/'external').resolve()
    assert resolve_data_dir(tmp_path,tmp_path/'explicit')==(tmp_path/'explicit').resolve()
    write_json(tmp_path/'outputs/selected_run.json',dict(run='outputs/runs/a'))
    assert resolve_run(tmp_path)==(tmp_path/'outputs/runs/a').resolve()
    assert file_hash(tmp_path/'outputs/selected_run.json')==hashlib.sha256((tmp_path/'outputs/selected_run.json').read_bytes()).hexdigest()
    result=subprocess.run([sys.executable,'-c',"import sys; import src.delivery; assert 'torch' not in sys.modules; assert 'src.experiment' not in sys.modules"],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_rgb_formula_matches_independent_population_statistics():
    from src.eda import rgb_statistics
    rng=np.random.default_rng(42)
    raw=rng.integers(0,256,size=(5,32,32,3),dtype=np.uint8)
    ids=np.array([1,3])
    actual=rgb_statistics(raw,ids)
    expected=raw[ids].astype(np.float64)/255
    np.testing.assert_allclose(actual['mean'],expected.mean((0,1,2)),atol=1e-12)
    np.testing.assert_allclose(actual['std'],expected.std((0,1,2)),atol=1e-12)
    assert (actual['hist'].sum(1)==2048).all()


def test_training_reuses_eda_locked_split(tmp_path,monkeypatch):
    import src.data as data
    labels=np.repeat(np.arange(10),5000)
    class FakeCifar:
        def __init__(self,root,train=True,download=False): self.targets=labels if train else np.repeat(np.arange(10),1000)
        def __len__(self): return len(self.targets)
    monkeypatch.setattr(data.datasets,'CIFAR10',FakeCifar)
    run=tmp_path/'run';run.mkdir()
    cfg=dict(seed=43,split_seed=42,train_limit=0,val_limit=0,test_limit=0)
    first=data.prepare_indices(tmp_path,run,cfg,split_dir=tmp_path/'shared')
    before=(tmp_path/'shared/split_indices.npz').read_bytes()
    cfg['seed']=44
    second=data.prepare_indices(tmp_path,run,cfg,split_dir=tmp_path/'shared')
    assert (tmp_path/'shared/split_indices.npz').read_bytes()==before
    np.testing.assert_array_equal(first['train'],second['train'])
    assert (run/'split_indices.npz').read_bytes()==before


def test_quality_cost_handles_new_csv_execution_column(tmp_path,monkeypatch):
    import matplotlib.pyplot as plt
    from src import notebook_views as view
    run=ROOT/json.loads((ROOT/'outputs/selected_run.json').read_text())['run']
    frame=pd.read_csv(run/'comparison.csv')
    costs=pd.read_csv(run/'cost_comparison.csv')
    frame=frame.drop(columns=['execution'],errors='ignore').merge(costs[['model','stage','execution']],on=['model','stage'])
    frame.to_csv(tmp_path/'comparison.csv',index=False)
    for model in frame.model.unique():
        folder=tmp_path/model/'head_selected';folder.mkdir(parents=True)
        (folder/'training_summary.json').write_bytes((run/model/'head_selected/training_summary.json').read_bytes())
    monkeypatch.setattr(view,'_save',lambda fig,*args:plt.close(fig))
    tables=[];monkeypatch.setattr(view,'display',tables.append)
    view.quality_cost(tmp_path,tmp_path)
    assert 'execution' in tables[0] and len(tables[0])==4


def test_complete_delivery_uses_requested_run_and_recommended_checkpoint(tmp_path,monkeypatch):
    import src.delivery as delivery
    import unittest.mock as mock
    root=tmp_path/'lab';(root/'outputs/runs/wanted').mkdir(parents=True)
    run=root/'outputs/runs/wanted'
    delivery.write_json(run/'status.json',dict(status='completed'))
    delivery.write_json(run/'verification.json',{})
    delivery.write_json(run/'validation_recommendation.json',dict(model='densenet121',stage='finetune'))
    delivery.write_json(root/'outputs/selected_run.json',dict(run='outputs/runs/other'))
    (root/'notebooks').mkdir()
    for name in ['01_lab02_eda.ipynb','02_lab02_modeling.ipynb']:
        delivery.write_json(root/'notebooks'/name,dict(cells=[dict(cell_type='code',execution_count=1,outputs=[dict(output_type='stream')])]))
    monkeypatch.setattr(delivery,'__file__',str(root/'src/utils/artifacts.py'))
    monkeypatch.setattr(delivery,'finalize_run',lambda *a:None)
    calls=[]
    def fake_run(args,**kwargs):
        calls.append((args,kwargs))
        return types.SimpleNamespace(returncode=0,stdout='{}' if 'infer.py' in ' '.join(args) else 'passed',stderr='')
    with mock.patch('subprocess.run',fake_run): delivery.complete_delivery(str(run))
    notebook_calls=[k for args,k in calls if 'execute_notebook.py' in ' '.join(args)]
    assert len(notebook_calls)==2
    assert all(k['env']['LAB02_RUN']==str(run.resolve()) for k in notebook_calls)
    infer=next(args for args,k in calls if 'infer.py' in ' '.join(args))
    assert str(run/'densenet121/finetune/best.pt') in infer


def test_finalizer_writes_one_config_driven_report(tmp_path,monkeypatch):
    """Synthetic probabilities + real label contract exercise finalizer/HTML/events."""
    import os
    import shutil
    from PIL import Image
    from sklearn.metrics import classification_report,log_loss
    from torch.utils.tensorboard import SummaryWriter
    from torchvision.datasets import CIFAR10
    import src.delivery as delivery
    from src.eda import CLASSES
    data_dir=Path(os.environ.get('LAB02_DATA',str(ROOT/'data')))
    if not (data_dir/'cifar-10-batches-py').exists(): pytest.skip('Raw CIFAR-10 required')
    labels=np.asarray(CIFAR10(str(data_dir),train=False).targets)
    root=tmp_path/'project';run=root/'outputs/runs/fixture'
    run.mkdir(parents=True)
    cfg=json.loads((ROOT/'config/cpu_full.json').read_text())
    cfg.update(profile='fixture_gpu224',image_size=224,finetune_epochs=10,head_epochs=1,
               models=['resnet18'],learning_rates=[.001],batch_sizes=[32])
    delivery.write_json(run/'config.json',cfg)
    delivery.write_json(run/'status.json',dict(status='completed'))
    delivery.write_json(run/'source_hashes.json',{})
    for name in ['split_indices.npz','selected_indices.npz']:
        shutil.copy2(ROOT/'outputs/eda/split_indices.npz',run/name)
    original=ROOT/json.loads((ROOT/'outputs/selected_run.json').read_text())['run']
    (run/'eda').mkdir()
    for name in ['class_counts.csv','training_examples.png']:
        shutil.copy2(original/'eda'/name,run/'eda'/name)
    probabilities=np.full((10000,10),.02)
    probabilities[np.arange(10000),labels]=.82
    report=pd.DataFrame(classification_report(labels,labels,target_names=CLASSES,output_dict=True)).T
    counts=pd.DataFrame(np.diag(np.bincount(labels)),index=CLASSES,columns=CLASSES)
    choice=dict(trial=0,lr=.001,batch_size=32,best_val_accuracy=.9,best_val_loss=.3,train_seconds=1.,best_epoch=1)
    delivery.write_json(run/'resnet18/selection.json',dict(head=choice))
    pd.DataFrame([choice]).to_csv(run/'resnet18/search_results.csv',index=False)
    delivery.write_json(run/'resnet18/head_selected/training_summary.json',dict(
        feature_extraction_train_val_seconds=2.,all_search_training_seconds=1.,train_seconds=1.))
    rows=[]
    for stage,folder,epochs in [('head','head_selected',1),('finetune','finetune',10)]:
        path=run/'resnet18'/folder;path.mkdir(exist_ok=True)
        np.savez_compressed(path/'test_predictions.npz',probabilities=probabilities,labels=labels,indices=np.arange(10000))
        metric=dict(test_samples=10000,accuracy=1.,macro_f1=1.,macro_precision=1.,macro_recall=1.,
                    loss=log_loss(labels,probabilities,labels=np.arange(10)))
        delivery.write_json(path/'metrics.json',metric)
        delivery.write_json(path/'roundtrip.json',dict(samples=8,max_probability_error=0.,passed=True))
        report.to_csv(path/'classification_report.csv');counts.to_csv(path/'confusion_counts.csv')
        for name in ['confusion_matrix','prediction_examples']:
            Image.new('RGB',(8,8),'white').save(path/f'{name}.png')
        history=pd.DataFrame([dict(epoch=i,train_loss=.1,val_loss=.3 if stage=='head' else .2,
            train_accuracy=1.,val_accuracy=.9 if stage=='head' else .92) for i in range(1,epochs+1)])
        history.to_csv(path/'history.csv',index=False)
        event_dir=run/'resnet18'/('head_trial_00' if stage=='head' else 'finetune')/'tensorboard'
        with SummaryWriter(str(event_dir)) as writer:
            for row in history.to_dict('records'):
                for tag in ['train_loss','val_loss','train_accuracy','val_accuracy']:
                    writer.add_scalar(tag,row[tag],row['epoch'])
        rows.append(dict(model='resnet18',stage=stage,**metric,best_val_accuracy=.9 if stage=='head' else .92,
            best_val_loss=.3 if stage=='head' else .2,total=100,trainable=10 if stage=='head' else 50,
            train_seconds=1. if stage=='head' else 10.,test_seconds=1.,
            execution='pooled_feature_cache' if stage=='head' else 'full_image_forward'))
    pd.DataFrame(rows).to_csv(run/'comparison.csv',index=False)
    monkeypatch.setattr(delivery,'__file__',str(root/'src/utils/artifacts.py'))
    monkeypatch.setenv('LAB02_DATA',str(data_dir))
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:'fixture==0\n')
    delivery.finalize_run(str(run))
    text=(run/'report_vi.md').read_text(encoding='utf-8')
    assert text.count('## 8. Khảo sát lỗi và hình bổ sung')==1
    assert '10 epoch fine-tune' in text and 'CPU profile chỉ có 2 epoch' not in text
    assert 'data:image/png;base64,' in (run/'report_vi.html').read_text(encoding='utf-8')
    assert json.loads((run/'verification.json').read_text())['status']=='passed'

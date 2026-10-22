"""Feature semantics, train-only scaling and durable failure evidence."""
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.datasets.features import extract_features
from src.models.mlp import FashionMLP
from src.utils.runs import Run


def test_confusion_counts_match_labels_and_row_colors():
    import matplotlib.pyplot as plt
    from src.utils.plots import draw_confusion_counts
    counts=np.array([[8,2,0],[1,9,0],[0,0,0]])
    fig,ax=plt.subplots()
    try:
        im=draw_confusion_counts(ax,counts,["a","b","c"],"Fixture")
        assert {(int(t.get_position()[1]),int(t.get_position()[0])):t.get_text()
                for t in ax.texts} == {idx:str(int(v)) for idx,v in np.ndenumerate(counts)}
        np.testing.assert_allclose(im.get_array(),[[.8,.2,0],[.1,.9,0],[0,0,0]])
        assert im.get_clim()==(0,1)
    finally:
        plt.close(fig)


def test_feature_geometry_and_empty_image():
    images=np.zeros((2,28,28),dtype=np.float32)
    images[1,4:24,10:18]=1
    f=extract_features(images)
    assert f.shape==(2,10) and np.isfinite(f).all()
    assert np.all(f[0,:8]==0) and f[0,8]==1
    assert f[1,2]==pytest.approx(160/784)
    assert f[1,3]==pytest.approx(8/28)
    assert f[1,4]==pytest.approx(20/28)
    assert f[1,5]==pytest.approx(.4)
    assert f[1,6]==pytest.approx(.5)
    assert f[1,7]==pytest.approx(.5)
    assert f[1,8]==1


def test_invalid_feature_input():
    with pytest.raises(ValueError): extract_features(np.ones((2,28,28))*255)
    with pytest.raises(ValueError): extract_features(np.ones((28,28)))


def test_three_input_contracts():
    for dim,hidden in ((784,(128,)),(784,(256,128)),(794,(256,128))):
        m=FashionMLP(in_features=dim,hidden_dims=hidden)
        x=torch.randn(8,dim)
        loss=torch.nn.functional.cross_entropy(m(x),torch.arange(8))
        loss.backward()
        assert m(x).shape==(8,10) and torch.isfinite(loss)


def test_failure_is_logged_and_manifested(tmp_path):
    with pytest.raises(RuntimeError):
        with Run('failure',root=tmp_path) as run:
            print('before failure')
            raise RuntimeError('expected failure')
    assert json.loads((run.path/'status.json').read_text())['status']=='failed'
    assert 'expected failure' in (run.path/'console.log').read_text()
    assert 'console.log' in {r['path'] for r in json.loads((run.path/'manifest.json').read_text())}


def test_three_one_batch_overfit():
    from src.training.trainer import sanity_overfit_check
    from src.utils.helpers import set_seed
    torch.set_num_threads(2)
    for dim, hidden in ((784,(128,)), (784,(256,128)), (794,(256,128))):
        set_seed(42)
        x=torch.randn(8,dim)
        ok,loss,acc=sanity_overfit_check(FashionMLP(in_features=dim,hidden_dims=hidden),
            x,torch.arange(8),device=torch.device("cpu"),num_steps=120,target_loss=.15)
        assert ok, (dim,loss,acc)


def test_plot_exports_with_synthetic_fixture(tmp_path,monkeypatch):
    # Synthetic fixture stays in pytest temp storage, never reported as experiment evidence.
    import src.utils.comparison_plots as plots
    from src.utils.runs import write_json
    monkeypatch.setattr(plots,"Run",lambda name,config: Run(name,config,root=tmp_path/"exports"))
    y=np.tile(np.arange(10),2)
    prob=np.full((20,10),.02)
    prob[np.arange(20),y]=.82
    paths=[]
    for name in ("SmallMLP","MainMLP","MainMLP_FE"):
        p=tmp_path/name; p.mkdir(); paths.append(p)
        write_json(p/"metrics.json",dict(name=name,accuracy=1.,macro_f1=1.,parameters=10,train_seconds=1))
        write_json(p/"config.json",dict(seed=42,data_sha256="synthetic",epochs=1,batch_size=8,learning_rate=.001,optimizer="Adam"))
        write_json(p/"history.json",dict(train_loss=[1],val_loss=[1],train_acc=[100],val_acc=[100]))
        write_json(p/"classification_report.json",{str(c):{"f1-score":1} for c in range(10)})
        np.savez(p/"test_predictions.npz",labels=y,probabilities=prob)
        np.save(p/"grid_pixels.npy",np.zeros((16,784)))
    out=plots.build_comparison(paths)
    assert len(list(out.glob("*_ovr_curves.png")))==3
    assert not (out/"f1_heatmap.png").exists()
    assert not (out/"pr_by_class.png").exists()
    assert not (out/"pr_roc_micro.png").exists()
    assert len(__import__("pandas").read_csv(out/"ovr_scores_by_class.csv"))==30
    assert (out/"MainMLP_pr_0.csv").is_file()
    assert (out/"manifest.json").is_file()


def test_optimizer_returns_best_epoch_weights(monkeypatch,tmp_path):
    import src.training.experiments as experiments
    model=torch.nn.Linear(2,2)
    with torch.no_grad(): model.weight.zero_(); model.bias.zero_()
    monkeypatch.setattr(experiments,"build_model",lambda *a,**k:model)
    def step(*args):
        with torch.no_grad(): model.weight.add_(1)
    values=iter([90.,80.])
    monkeypatch.setattr(experiments,"train_one_epoch",step)
    monkeypatch.setattr(experiments,"evaluate",lambda *a:(1.,next(values),[],[]))
    _,winner,_,fig=experiments.compare_optimizers_wallclock(None,None,
        optimizer_configs={"Adam":{"lr":.001}},num_epochs=2,use_scheduler=False,
        device=torch.device("cpu"),save_fig_path=tmp_path/"optimizer.png")
    import matplotlib.pyplot as plt
    plt.close(fig)
    assert torch.all(winner.weight==1)


def test_model_runner_smoke_all_three(tmp_path,monkeypatch):
    import src.training.comparison as comparison
    monkeypatch.setattr(comparison,"DEVICE",torch.device("cpu"))
    monkeypatch.setattr(comparison,"Run",lambda name,config:Run(name,config,root=tmp_path/"runs"))
    rng=np.random.default_rng(42)
    arrays={}
    for split in ("train","val","test"):
        arrays[split+"_pixels"]=rng.normal(size=(20,784)).astype(np.float32)
        arrays[split+"_features"]=rng.normal(size=(20,10)).astype(np.float32)
        arrays[split+"_labels"]=np.tile(np.arange(10),2)
    data=tmp_path/"synthetic_fixture.npz"; np.savez(data,**arrays)
    for name in comparison.CONFIGS:
        out=comparison.run_model(data,name,epochs=2,batch_size=20)
        assert len(json.loads((out/"history.json").read_text())["train_loss"])==2
        assert json.loads((out/"config.json").read_text())["device"]=="cpu"
        assert (out/"best.pt").is_file() and (out/"learning_curves.png").is_file()
        assert json.loads((out/"status.json").read_text())["status"]=="completed"
        pred=np.load(out/"test_predictions.npz")
        assert pred["probabilities"].shape==(20,10)
        assert np.allclose(pred["probabilities"].sum(1),1)
    custom=comparison.run_model(data,"MainMLP",epochs=1,batch_size=20,
                                model_config={"in_features":784,"hidden_dims":[32]})
    assert json.loads((custom/"config.json").read_text())["hidden_dims"]==[32]
    from src.inference import load_trained_run
    restored,test=load_trained_run(custom)
    assert sum(p.numel() for p in restored.parameters())==25450
    x,_=next(iter(test))
    assert restored(x.to(next(restored.parameters()).device)).shape==(20,10)

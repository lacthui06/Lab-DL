"""Shared configuration, view mode, preprocessing and notebook cleanup contracts."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.workflow import model_stage,saved_run,validate_mode
from src.datasets.features import extract_features,FEATURE_NAMES
from src.datasets.eda import compute_handcrafted_features


def test_view_rejects_stale_comparison_after_model_selection(tmp_path):
    import json
    from src.workflow import comparison_stage
    paths = [tmp_path/name for name in ("SmallMLP_run", "MainMLP_run", "MainMLP_FE_run")]
    # Stored Windows paths must remain usable when runs move to another machine.
    sources = ["C:\\old\\outputs\\runs\\"+p.name for p in paths]
    (tmp_path/"config.json").write_text(json.dumps({"source_runs":sources}))
    cfg = {"runs":{"comparison":str(tmp_path)}}
    assert comparison_stage(cfg,paths,"view") == tmp_path
    with pytest.raises(ValueError,match="different model runs"):
        comparison_stage(cfg,[paths[0],tmp_path/"another_MainMLP",paths[2]],"view")


def test_view_never_trains(tmp_path,monkeypatch):
    import src.workflow as w
    monkeypatch.setattr(w,"run_model",lambda *a,**k:pytest.fail("View must not train"))
    cfg={"runs":{"models":{"MainMLP":str(tmp_path)}}}
    assert model_stage(cfg,"MainMLP","view")==tmp_path
    with pytest.raises(ValueError):validate_mode("invalid")


def test_model_stage_forwards_custom_architecture(tmp_path,monkeypatch):
    import src.workflow as w
    config=w.load_workflow_config()
    config["runs"]["preparation"]=str(tmp_path)
    config["models"]["MainMLP"]={"in_features":784,"hidden_dims":[32]}
    received={}
    def capture(*args,**kwargs):
        received.update(kwargs)
        return tmp_path
    monkeypatch.setattr(w,"run_model",capture)
    assert w.model_stage(config,"MainMLP","train")==tmp_path
    assert received["model_config"]=={"in_features":784,"hidden_dims":[32]}


def test_eda_uses_model_feature_definition():
    rng=np.random.default_rng(42)
    raw=rng.integers(0,256,size=(10,28,28),dtype=np.uint8)
    features=extract_features(raw.astype(np.float32)/255)
    table=compute_handcrafted_features(raw,np.zeros(10,dtype=int),["class"])
    assert table.iloc[0].edge_density_mean==pytest.approx(round(float(features[:,9].mean()),4))


def test_inference_uses_saved_architecture(tmp_path,monkeypatch):
    import json,torch
    from src.models.mlp import FashionMLP
    from src.inference import load_trained_run
    data=tmp_path/"data.npz";a={}
    for split in ("train","val","test"):
        a[split+"_pixels"]=np.zeros((10,784),dtype=np.float32)
        a[split+"_features"]=np.zeros((10,10),dtype=np.float32)
        a[split+"_labels"]=np.arange(10)
    np.savez(data,**a)
    config=dict(name="MainMLP_FE",in_features=794,hidden_dims=[16],data_path=str(data),seed=42,batch_size=10)
    (tmp_path/"config.json").write_text(json.dumps(config))
    torch.save({"model_state_dict":FashionMLP(in_features=794,hidden_dims=[16]).state_dict()},tmp_path/"best.pt")
    model,test=load_trained_run(tmp_path)
    x,y=next(iter(test))
    assert model(x.to(next(model.parameters()).device)).shape==(10,10)
    import src.inference as inference
    from src.utils.runs import Run
    monkeypatch.setattr(inference,"Run",lambda name,config:Run(name,config,root=tmp_path/"exports"))
    exported,restored=inference.save_load_model(tmp_path)
    assert json.loads((exported/"roundtrip.json").read_text())["passed"]
    assert json.loads((exported/"test_metrics.json").read_text())["samples"]==10
    assert (exported/"loaded_model_test_samples.png").is_file()
    assert (exported/"loaded_model_test_samples.csv").is_file()
    assert np.load(exported/"test_predictions.npz")["probabilities"].shape==(10,10)

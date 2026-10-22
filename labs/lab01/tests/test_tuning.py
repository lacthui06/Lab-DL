"""Real Optuna plumbing on synthetic fixtures, not benchmark evidence."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def test_equal_budget_fixed_tuning_refit(tmp_path,monkeypatch):
    import pytest
    pytest.importorskip("optuna")
    import src.training.tuning as tuning
    import src.training.comparison as comparison
    from src.utils.runs import Run
    torch.set_num_threads(2)
    monkeypatch.setattr(tuning,"DEVICE",torch.device("cpu"))
    monkeypatch.setattr(comparison,"DEVICE",torch.device("cpu"))
    factory=lambda name,config,root=None:Run(name,config,root=root or tmp_path/"runs")
    monkeypatch.setattr(tuning,"Run",factory)
    monkeypatch.setattr(comparison,"Run",factory)
    rng=np.random.default_rng(42);arrays={}
    for split in ("train","val","test"):
        arrays[split+"_pixels"]=rng.normal(size=(20,784)).astype(np.float32)
        arrays[split+"_features"]=rng.normal(size=(20,10)).astype(np.float32)
        arrays[split+"_labels"]=np.tile(np.arange(10),2)
    p=tmp_path/"synthetic.npz";np.savez(p,**arrays)
    fixed=tuning.run_fixed_lr(p,epochs=2,batch_size=20)
    assert (fixed/"fixed_lr_curves.png").is_file()
    study=tuning.run_tuning(p,trials_per_optimizer=1,epochs=2,batch_size=20)
    import pandas as pd
    trials=pd.read_csv(study/"all_trials.csv")
    assert trials.groupby("optimizer").size().to_dict()=={"Adam":1,"RMSprop":1,"SGD":1}
    assert (study/"studies.sqlite3").is_file() and (study/"tuning_progress.png").is_file()
    curves=tuning.plot_best_optimizer_histories(study)
    assert (curves/"best_optimizer_learning_curves.png").is_file()
    history=pd.read_csv(curves/"best_optimizer_history.csv")
    assert set(history.optimizer)==set(tuning.OPTIMIZERS)
    assert set(("train_loss","val_loss","train_acc","val_acc")).issubset(history.columns)
    for name in tuning.OPTIMIZERS:
        trial=pd.read_csv(study/f"{name}_trials.csv").iloc[0]
        assert (Path(trial.user_attrs_run_path)/"best.pt").is_file()
    final=tuning.refit_winner(p,study,epochs=2,batch_size=20)
    assert (final/"test_predictions.npz").is_file()
    results=tuning.summarize_optimizer_runs(fixed,study,final)
    assert (results/"final_ovr_curves.png").is_file()
    assert len(pd.read_csv(results/"fixed_vs_tuned.csv"))==3
    # Validation candidates must never export test predictions.
    assert not list((study/"trials").rglob("test_predictions.npz"))
    with pytest.raises(ValueError,match="Refit must use"):
        tuning.refit_winner(p,study,epochs=3,batch_size=20)


def test_pruned_trial_has_distinct_status(tmp_path):
    import pytest
    optuna=pytest.importorskip("optuna")
    from src.utils.runs import Run
    with pytest.raises(optuna.TrialPruned):
        with Run("pruning",root=tmp_path) as run:
            raise optuna.TrialPruned()
    assert json.loads((run.path/"status.json").read_text())["status"]=="pruned"

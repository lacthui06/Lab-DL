"""Equal-budget optimizer studies; objectives use validation only."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from ..models.mlp import FashionMLP
from ..utils.helpers import DEVICE, SOURCE_DIR, set_seed
from ..utils.plots import draw_confusion_counts

SEARCH = json.loads((SOURCE_DIR/"config/workflow.json").read_text(encoding="utf-8"))["search"]
from ..utils.runs import Run, write_json
from .comparison import loaders, run_model
from .trainer import Trainer, build_optimizer

OPTIMIZERS = ("SGD", "RMSprop", "Adam")


def plot_best_optimizer_histories(tuning_path):
    """Compare train/validation histories of each optimizer's best completed trial."""
    tuning_path = Path(tuning_path)
    best = json.loads((tuning_path/"best_configs.json").read_text(encoding="utf-8"))
    if {r["optimizer"] for r in best} != set(OPTIMIZERS):
        raise ValueError("Expected a best completed trial for each optimizer")
    with Run("best_optimizer_learning_curves", {"tuning_run":str(tuning_path.resolve()),
             "source":"best completed trial per optimizer; no retraining"}) as run:
        fig, axes = plt.subplots(2,2,figsize=(14,9))
        frames = []
        try:
            for row in best:
                name = row["optimizer"]
                trial_id = row["run_path"].replace("\\", "/").rstrip("/").split("/")[-1]
                source = tuning_path/"trials"/name/trial_id
                history = pd.read_csv(source/"history.csv")
                history.insert(0,"optimizer",name)
                history.insert(1,"trial_number",row["trial_number"])
                frames.append(history)
                for ax, field in zip(axes.flat,("train_loss","val_loss","train_acc","val_acc")):
                    ax.plot(history.epoch,history[field],label=name)
                    ax.set(xlabel="Epoch",ylabel="Accuracy (%)" if field.endswith("acc") else "CrossEntropy loss",
                           title=field.replace("_"," ").title())
                    ax.legend(); ax.grid(alpha=.2)
            fig.suptitle("Best parameters per optimizer: completed trial learning curves")
            fig.tight_layout();fig.savefig(run.path/"best_optimizer_learning_curves.png",dpi=150)
        finally:
            plt.close(fig)
        pd.concat(frames,ignore_index=True).to_csv(run.path/"best_optimizer_history.csv",index=False)
        pd.DataFrame(best).to_csv(run.path/"best_parameters.csv",index=False)
        print("Saved train/validation loss and accuracy for SGD, RMSprop and Adam:",run.path)
    return run.path


def train_candidate(data_path, optimizer_name, params, epochs=10, seed=42,
                    batch_size=128, root=None, trial=None):
    config=dict(optimizer=optimizer_name,params=params,epochs=epochs,seed=seed,
                batch_size=batch_size,model="MainMLP",architecture=[784,256,128,10],
                device=str(DEVICE),scheduler="CosineAnnealingLR",selection="best val accuracy",
                data_sha256=hashlib.sha256(Path(data_path).read_bytes()).hexdigest())
    with Run(optimizer_name,config,root=root) as run:
        if trial is not None: trial.set_user_attr("run_path",str(run.path.resolve()))
        set_seed(seed)
        train,val,_=loaders(data_path,"MainMLP",seed,batch_size)
        model=FashionMLP().to(DEVICE)
        opt=build_optimizer(optimizer_name,model,learning_rate=params["lr"],
                            weight_decay=params.get("weight_decay",0),momentum=params.get("momentum",.9))
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
        trainer=Trainer(model,torch.nn.CrossEntropyLoss(),opt,DEVICE,scheduler)
        history=[]; best=-1; start=time.perf_counter()
        for ep in range(epochs):
            train_loss,train_acc=trainer.train_one_epoch(train)
            val_loss,val_acc,_,_=trainer.evaluate(val)
            lr=opt.param_groups[0]["lr"]
            scheduler.step()
            history.append(dict(epoch=ep+1,train_loss=train_loss,train_acc=train_acc,
                                val_loss=val_loss,val_acc=val_acc,learning_rate=lr))
            pd.DataFrame(history).to_csv(run.path/"history.csv",index=False)
            write_json(run.path/"history.json",history)
            if val_acc>best:
                best=val_acc
                torch.save({"model_state_dict":model.state_dict(),"metadata":history[-1]},run.path/"best.pt")
            result=dict(best_val_acc=best,best_epoch=max(history,key=lambda r:r["val_acc"])["epoch"],
                        seconds=time.perf_counter()-start,epochs_completed=ep+1)
            write_json(run.path/"validation.json",result)
            print(f"{optimizer_name} epoch {ep+1}/{epochs}: val_acc={val_acc:.3f}, loss={val_loss:.4f}")
            if trial is not None:
                trial.report(best,ep)
                if trial.should_prune():
                    write_json(run.path/"pruning.json",{"pruned":True,"epoch":ep+1})
                    import optuna
                    raise optuna.TrialPruned()
    return run.path,result


def run_fixed_lr(data_path,epochs=10,seed=42,batch_size=128,root=None):
    config=dict(phase="fixed_lr",epochs=epochs,seed=seed,batch_size=batch_size,lr=.001,
                data_sha256=hashlib.sha256(Path(data_path).read_bytes()).hexdigest())
    with Run("optimizer_fixed_lr",config,root=root) as run:
        rows=[]; histories={}
        for name in OPTIMIZERS:
            params=dict(lr=.001,weight_decay=0.)
            if name=="SGD": params["momentum"]=.9
            path,result=train_candidate(data_path,name,params,epochs,seed,batch_size,root=run.path/"candidates")
            rows.append(dict(optimizer=name,run_path=str(path.resolve()),**result))
            pd.DataFrame(rows).to_csv(run.path/"summary.csv",index=False)
            histories[name]=pd.read_csv(path/"history.csv")
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for name,h in histories.items():
            axes[0].plot(h.epoch,h.val_acc,label=name)
            axes[1].plot(h.epoch,h.val_loss,label=name)
        for ax,title in zip(axes,("Validation accuracy (%)","Validation loss")):
            ax.set_title(title);ax.set_xlabel("Epoch");ax.legend()
        fig.tight_layout();fig.savefig(run.path/"fixed_lr_curves.png",dpi=150);plt.close(fig)
        print(pd.DataFrame(rows).to_string(index=False))
    return run.path


def run_tuning(data_path,trials_per_optimizer=10,epochs=10,seed=42,batch_size=128,root=None):
    config=dict(trials_per_optimizer=trials_per_optimizer,epochs=epochs,seed=seed,batch_size=batch_size,
                data_sha256=hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                sampler={"name":"TPESampler","n_startup_trials":SEARCH["startup_trials"],"seed":seed},
                selection="best validation accuracy",
                search={key:SEARCH[key] for key in ("SGD_lr","adaptive_lr","weight_decay","SGD_momentum")},
                pruner={"name":"MedianPruner","startup_trials":SEARCH["pruner_startup_trials"],
                        "warmup_steps":SEARCH["pruner_warmup_steps"]})
    with Run("optimizer_tuning",config,root=root) as run:
        import optuna
        # Use our durable callback log rather than a handler bound to a closed prior run.
        optuna.logging.disable_default_handler()
        best_rows=[]; all_trials=[]
        for name in OPTIMIZERS:
            study=optuna.create_study(study_name=name,direction="maximize",
                storage="sqlite:///"+str((run.path/"studies.sqlite3").resolve()).replace("\\","/"),
                sampler=optuna.samplers.TPESampler(seed=seed,n_startup_trials=SEARCH["startup_trials"]),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=SEARCH["pruner_startup_trials"],n_warmup_steps=SEARCH["pruner_warmup_steps"]))
            def objective(trial):
                lo,hi=SEARCH["SGD_lr"] if name=="SGD" else SEARCH["adaptive_lr"]
                params=dict(lr=trial.suggest_float("lr",lo,hi,log=True),
                            weight_decay=trial.suggest_float("weight_decay",*SEARCH["weight_decay"],log=True))
                if name=="SGD": params["momentum"]=trial.suggest_float("momentum",*SEARCH["SGD_momentum"])
                _,result=train_candidate(data_path,name,params,epochs,seed,batch_size,
                    root=run.path/"trials"/name,trial=trial)
                return result["best_val_acc"]
            def save_trials(study,trial):
                study.trials_dataframe().to_csv(run.path/f"{name}_trials.csv",index=False)
                print(f"{name} trial {trial.number}: {trial.state.name}; value={trial.value}; params={trial.params}")
            study.optimize(objective,n_trials=trials_per_optimizer,callbacks=[save_trials])
            complete=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
            if not complete: raise RuntimeError(f"No completed trials for {name}")
            best=study.best_trial
            best_rows.append(dict(optimizer=name,best_val_acc=best.value,trial_number=best.number,
                                 run_path=best.user_attrs["run_path"],**best.params))
            write_json(run.path/"best_configs.json",best_rows)
            frame=study.trials_dataframe();frame.insert(0,"optimizer",name);all_trials.append(frame)
        trials=pd.concat(all_trials,ignore_index=True)
        trials.to_csv(run.path/"all_trials.csv",index=False)
        rows=[]
        for r in best_rows:
            subset=trials[trials.optimizer==r["optimizer"]]
            rows.append({**r,"trials":len(subset),"completed":int((subset.state=="COMPLETE").sum()),
                         "pruned":int((subset.state=="PRUNED").sum()),
                         "seconds":float(subset.duration.dt.total_seconds().sum())})
        pd.DataFrame(rows).to_csv(run.path/"best_configs.csv",index=False)
        fig,axes=plt.subplots(1,2,figsize=(13,4))
        for name in OPTIMIZERS:
            frame=trials[(trials.optimizer==name)&(trials.state=="COMPLETE")].sort_values("number")
            axes[0].step(frame.number+1,frame.value.cummax(),where="post",label=name)
            axes[1].scatter(frame.params_lr,frame.value,label=name)
        axes[0].set(xlabel="Trial number",ylabel="Best completed validation accuracy (%)")
        axes[1].set(xlabel="Learning rate",ylabel="Validation accuracy (%)",xscale="log")
        for ax in axes:ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(run.path/"tuning_progress.png",dpi=150);plt.close(fig)
        print(pd.DataFrame(rows).to_string(index=False))
    return run.path


def refit_winner(data_path,tuning_path,epochs=10,seed=42,batch_size=128):
    """Freeze validation-selected config, then evaluate test in run_model once."""
    tuning_path=Path(tuning_path)
    config=json.loads((tuning_path/"config.json").read_text())
    if (epochs,seed,batch_size)!=(config["epochs"],config["seed"],config["batch_size"]):
        raise ValueError("Refit must use the study's epochs, seed and batch size")
    if hashlib.sha256(Path(data_path).read_bytes()).hexdigest()!=config["data_sha256"]:
        raise ValueError("Refit data differs from tuning data")
    configs=json.loads((tuning_path/"best_configs.json").read_text())
    winner=max(configs,key=lambda r:r["best_val_acc"])
    with Run("refit_selected_optimizer",{"study":str(tuning_path.resolve()),"winner":winner}) as run:
        path=run_model(data_path,"MainMLP",epochs=epochs,seed=seed,batch_size=batch_size,
                       learning_rate=winner["lr"],optimizer_name=winner["optimizer"],
                       weight_decay=winner["weight_decay"],momentum=winner.get("momentum",.9))
        write_json(run.path/"refit_run.json",{"path":str(path.resolve())})
    return path


def summarize_optimizer_runs(fixed_path,tuning_path,final_path):
    """Persist fixed-vs-tuned validation comparison and final test diagnostics."""
    from sklearn.metrics import roc_curve,auc,precision_recall_curve,average_precision_score,confusion_matrix
    from ..utils.helpers import CLASS_NAMES
    fixed_path,tuning_path,final_path=map(Path,(fixed_path,tuning_path,final_path))
    with Run("optimizer_results",{"fixed":str(fixed_path.resolve()),"tuning":str(tuning_path.resolve()),
                                  "final":str(final_path.resolve())}) as run:
        fixed=pd.read_csv(fixed_path/"summary.csv")
        tuned=pd.read_csv(tuning_path/"best_configs.csv")
        table=fixed[["optimizer","best_val_acc","seconds"]].merge(
            tuned[["optimizer","best_val_acc","seconds"]],on="optimizer",suffixes=("_fixed","_tuned"))
        table.to_csv(run.path/"fixed_vs_tuned.csv",index=False)
        fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(table))
        ax.bar(x-.18,table.best_val_acc_fixed,width=.36,label="Fixed LR")
        ax.bar(x+.18,table.best_val_acc_tuned,width=.36,label="Tuned")
        ax.set_xticks(x,table.optimizer);ax.set_ylabel("Best validation accuracy (%)");ax.legend()
        fig.tight_layout();fig.savefig(run.path/"fixed_vs_tuned.png",dpi=150);plt.close(fig)
        pred=np.load(final_path/"test_predictions.npz");y=pred["labels"];probs=pred["probabilities"]
        fig,axes=plt.subplots(1,2,figsize=(16,6));scores=[]
        for c,name in enumerate(CLASS_NAMES):
            binary=(y==c).astype(int)
            fpr,tpr,rt=roc_curve(binary,probs[:,c]);precision,recall,pt=precision_recall_curve(binary,probs[:,c])
            score=auc(fpr,tpr);ap=average_precision_score(binary,probs[:,c])
            scores.append(dict(class_name=name,roc_auc=score,average_precision=ap))
            pd.DataFrame(dict(fpr=fpr,tpr=tpr,threshold=rt)).to_csv(run.path/f"roc_{c}.csv",index=False)
            pd.DataFrame(dict(precision=precision,recall=recall,threshold=np.r_[pt,np.nan])).to_csv(run.path/f"pr_{c}.csv",index=False)
            axes[0].plot(fpr,tpr,label=f"{name} AUC={score:.3f}")
            axes[1].plot(recall,precision,label=f"{name} AP={ap:.3f}")
        axes[0].set(xlabel="False positive rate",ylabel="True positive rate",title="Selected configuration: ROC OvR")
        axes[1].set(xlabel="Recall",ylabel="Precision",title="Selected configuration: Precision-Recall OvR")
        for ax in axes:ax.legend(fontsize=8,loc="lower left");ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(run.path/"final_ovr_curves.png",dpi=150);plt.close(fig)
        pd.DataFrame(scores).to_csv(run.path/"final_ovr_scores.csv",index=False)
        counts=confusion_matrix(y,probs.argmax(1),labels=range(10))
        pd.DataFrame(counts,index=CLASS_NAMES,columns=CLASS_NAMES).to_csv(run.path/"final_confusion_counts.csv")
        fig,ax=plt.subplots(figsize=(8,7))
        im=draw_confusion_counts(ax,counts,CLASS_NAMES,"Selected configuration: test confusion matrix")
        fig.colorbar(im,ax=ax,label="Fraction of true class")
        fig.tight_layout();fig.savefig(run.path/"final_confusion_matrix.png",dpi=150);plt.close(fig)
        write_json(run.path/"final_metrics.json",json.loads((final_path/"metrics.json").read_text()))
        print(table.to_string(index=False))
    return run.path

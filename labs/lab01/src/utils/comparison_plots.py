"""Rebuild comparison figures from persisted runs without retraining."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_recall_curve, average_precision_score,
                             roc_curve, auc)
from .helpers import CLASS_NAMES
from .runs import Run
from .plots import draw_confusion_counts


def build_comparison(run_paths):
    paths = [Path(p) for p in run_paths]
    records = [json.loads((p/"metrics.json").read_text()) for p in paths]
    if len(records) != 3 or {r["name"] for r in records} != {"SmallMLP", "MainMLP", "MainMLP_FE"}:
        raise ValueError("Supply exactly the three distinct model configurations")
    configs=[json.loads((p/"config.json").read_text()) for p in paths]
    for key in ("seed","data_sha256","epochs","batch_size","learning_rate","optimizer"):
        if len({c[key] for c in configs}) != 1:
            raise ValueError(f"Comparison mismatch: {key}")
    for key, default in (("weight_decay", 0.0), ("momentum", 0.9), ("scheduler", "CosineAnnealingLR")):
        if len({c.get(key, default) for c in configs}) != 1:
            raise ValueError(f"Comparison mismatch: {key}")
    histories=[json.loads((p/"history.json").read_text()) for p in paths]
    predictions=[np.load(p/"test_predictions.npz") for p in paths]
    y=predictions[0]["labels"]
    if any(not np.array_equal(y,d["labels"]) for d in predictions):
        raise ValueError("Test ordering differs")
    if all("sample_indices" in d for d in predictions):
        if any(not np.array_equal(predictions[0]["sample_indices"],d["sample_indices"]) for d in predictions):
            raise ValueError("Test sample indices differ")
    with Run("comparison", {"source_runs":[str(p.resolve()) for p in paths]}) as run:
        def save(fig,name):
            fig.tight_layout(); fig.savefig(run.path/f"{name}.png",dpi=150,bbox_inches="tight"); plt.close(fig)
        df=pd.DataFrame(records); df.to_csv(run.path/"comparison.csv",index=False)
        names=df.name.tolist()
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for name,h in zip(names,histories):
            axes[0].plot(range(1,len(h["val_acc"])+1),h["val_acc"],label=name)
            axes[1].plot(range(1,len(h["val_loss"])+1),h["val_loss"],label=name)
        for ax,title in zip(axes,("Validation accuracy (%)","Validation loss")):
            ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend()
        save(fig,"validation_comparison")
        for fields,filename in [(("accuracy","macro_f1"),"test_scores"),(("parameters","train_seconds"),"cost")]:
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            for ax,k in zip(axes,fields):
                ax.bar(names,df[k]); ax.set_title(k)
            save(fig,filename)
        f1=np.array([[json.loads((p/"classification_report.json").read_text())[str(c)]["f1-score"] for p in paths] for c in range(10)])
        pd.DataFrame(f1,index=CLASS_NAMES,columns=names).to_csv(run.path/"f1_by_class.csv")
        fig,axes=plt.subplots(1,3,figsize=(19,6))
        for ax,name,d in zip(axes,names,predictions):
            cm=confusion_matrix(y,d["probabilities"].argmax(1),labels=range(10),normalize="true")
            pd.DataFrame(cm,index=CLASS_NAMES,columns=CLASS_NAMES).to_csv(run.path/f"{name}_confusion_normalized.csv")
            counts=confusion_matrix(y,d["probabilities"].argmax(1),labels=range(10))
            pd.DataFrame(counts,index=CLASS_NAMES,columns=CLASS_NAMES).to_csv(run.path/f"{name}_confusion_counts.csv")
            im = draw_confusion_counts(ax, counts, CLASS_NAMES, name)
            fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="Fraction of true class")
        save(fig,"confusion_comparison")
        binary=np.eye(10)[y]
        scores=[]
        for name,d in zip(names,predictions):
            fig,axes=plt.subplots(1,2,figsize=(16,6))
            colors=plt.get_cmap("tab10")
            for c in range(10):
                prob=d["probabilities"][:,c]
                fpr,tpr,roc_threshold=roc_curve(binary[:,c],prob)
                roc_auc=auc(fpr,tpr)
                precision,recall,pr_threshold=precision_recall_curve(binary[:,c],prob)
                ap=average_precision_score(binary[:,c],prob)
                scores.append(dict(name=name,class_index=c,class_name=CLASS_NAMES[c],
                                   roc_auc=roc_auc,average_precision=ap))
                pd.DataFrame(dict(fpr=fpr,tpr=tpr,threshold=roc_threshold)).to_csv(run.path/f"{name}_roc_{c}.csv",index=False)
                pd.DataFrame(dict(precision=precision,recall=recall,threshold=np.r_[pr_threshold,np.nan])).to_csv(run.path/f"{name}_pr_{c}.csv",index=False)
                axes[0].plot(fpr,tpr,color=colors(c),label=f"{CLASS_NAMES[c]} AUC={roc_auc:.3f}")
                axes[1].plot(recall,precision,color=colors(c),label=f"{CLASS_NAMES[c]} AP={ap:.3f}")
            axes[0].plot([0,1],[0,1],color="gray",linestyle=":",label="Chance")
            axes[0].set(xlabel="False positive rate",ylabel="True positive rate",title=f"{name}: ROC one-vs-rest")
            axes[1].set(xlabel="Recall",ylabel="Precision",title=f"{name}: Precision-Recall one-vs-rest")
            for ax in axes:
                ax.set_xlim(0,1); ax.set_ylim(0,1.02)
                ax.legend(fontsize=8,loc="lower left")
                ax.grid(alpha=.2)
            save(fig,f"{name}_ovr_curves")
        pd.DataFrame(scores).to_csv(run.path/"ovr_scores_by_class.csv",index=False)
        for p,name,h,d in zip(paths,names,histories,predictions):
            pixels=np.load(p/"grid_pixels.npy").reshape(-1,28,28)
            fig,axes=plt.subplots(4,4,figsize=(10,10))
            for i,ax in enumerate(axes.flat):
                pred=int(d["probabilities"][i].argmax())
                ax.imshow(pixels[i],cmap="gray"); ax.axis("off")
                ax.set_title(f"True: {CLASS_NAMES[y[i]]}\nPred: {CLASS_NAMES[pred]}",fontsize=8)
            save(fig,f"{name}_prediction_grid")
        print(df.to_string(index=False))
    return run.path

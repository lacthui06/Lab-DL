"""Trực quan hóa và xuất artifacts từ sweep validation và checkpoint eval thật."""
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from .evidence import selected_run, read_json, verify_manifest
from .threshold_evidence import audit_threshold_experiment
from .utils.runs import ROOT, run_context, save_json, hash_file
from .utils.plots import finish
from .analysis import _table_image


def _source_key(paths):
    return {"source_runs":{k:p.name for k,p in paths.items()},
            "source_manifests":{k:hash_file(p/"manifest.json") for k,p in paths.items()},
            "analysis_code":hash_file(Path(__file__)),
            "audit_code":hash_file(Path(__file__).with_name("threshold_evidence.py")),
            "plot_helpers":hash_file(Path(__file__).with_name("analysis.py"))}


def _metric_plot(rows, path, title):
    keys=["accuracy","precision","recall","binary_f1","macro_f1"]
    x=np.arange(len(keys));fig,ax=plt.subplots(figsize=(11,4.5))
    for i,row in enumerate(rows):
        bars=ax.bar(x+(i-.5)*.36,[row[k] for k in keys],.36,label=row["method"])
        ax.bar_label(bars,fmt="%.4f",fontsize=8)
    ax.set(xticks=x,xticklabels=keys,ylim=(0,1.12),ylabel="Metric",title=title)
    ax.legend(fontsize=8,loc="lower left");finish(fig,path)


def _per_class(cm, method):
    cm=np.asarray(cm);rows=[]
    for i,label in enumerate(["NEGATIVE","POSITIVE"]):
        tp=int(cm[i,i]);fn=int(cm[i].sum()-tp);fp=int(cm[:,i].sum()-tp);tn=int(cm.sum()-tp-fn-fp)
        rows.append({"method":method,"label":label,"precision":tp/(tp+fp) if tp+fp else 0.,
            "recall_TPR":tp/(tp+fn),"F1":2*tp/(2*tp+fp+fn),"FPR":fp/(fp+tn),"support":tp+fn})
    return rows


def matching_threshold_runs(training_run=None, root=ROOT):
    """Chọn bộ run cùng policy/checkpoint, bỏ qua inference không liên quan."""
    root=Path(root)
    state=read_json(root/"config/workflow.json")
    kinds=["threshold_selection","evaluation","inference"]
    # Không ghép active inference không liên quan với evaluation của checkpoint cũ.
    candidates = {k: [] for k in kinds}
    for row in reversed(state["runs"]):
        kind = row["kind"]
        if kind in candidates and row["status"] == "completed":
            path = root / "outputs/runs" / row["id"]
            if (path / "decision_threshold.json").is_file():
                candidates[kind].append(path)
    # Registry tối giản có thể chỉ cung cấp active_runs.
    for kind in kinds:
        if kind in state["active_runs"]:
            path = selected_run(kind, root)
            if (path / "decision_threshold.json").is_file() and path not in candidates[kind]:
                candidates[kind].insert(0, path)
    paths = None
    for evaluation in candidates["evaluation"]:
        cfg = read_json(evaluation / "config.json")
        recorded_training = Path(cfg["evaluated_checkpoint"]).parent.name
        if training_run is not None and Path(training_run).name != recorded_training:
            continue
        policy = read_json(evaluation / "decision_threshold.json")
        matching = {k: next((p for p in candidates[k] if read_json(p / "decision_threshold.json") == policy), None)
                    for k in ["threshold_selection", "inference"]}
        if all(matching.values()):
            paths = dict(matching, evaluation=evaluation)
            break
    return paths


def ensure_threshold_analysis(training_run=None, root=ROOT):
    """Cache theo manifest và code; không ghép nhầm kết quả của model/run khác."""
    root = Path(root)
    state = read_json(root / "config/workflow.json")
    paths = matching_threshold_runs(training_run, root)
    if paths is None:
        return None
    expected=_source_key(paths)
    for row in reversed(state["runs"]):
        if row["kind"]!="threshold_analysis" or row["status"]!="completed": continue
        path=root/"outputs/runs"/row["id"]
        if (path/"analysis_source.json").exists() and read_json(path/"analysis_source.json")==expected:
            verify_manifest(path);return path
    return analyze_threshold_experiment(paths,root)


def analyze_threshold_experiment(paths,root=ROOT):
    """So sánh quyết định và loss, không fit lại threshold bằng nhãn test."""
    audit=audit_threshold_experiment(paths["threshold_selection"],paths["evaluation"],paths["inference"])
    threshold=audit["policy"]["selected_threshold"]
    raw=read_json(paths["evaluation"]/"all_predictions.json")
    labels=np.array([r["true_label"] for r in raw]);probs=np.array([r["probabilities"] for r in raw])
    baseline=probs.argmax(1);selected=(probs[:,1]>=threshold).astype(int)
    cm0=np.array([[int(sum((labels==a)&(baseline==b))) for b in [0,1]] for a in [0,1]])
    cm1=np.array([[int(sum((labels==a)&(selected==b))) for b in [0,1]] for a in [0,1]])
    base=audit["test_argmax"]
    test_rows=[{"method":"Argmax","accuracy":base["accuracy"],"precision":base["binary_precision"],
        "recall":base["binary_recall"],"binary_f1":base["binary_f1"],"macro_f1":base["macro_f1"]},
        {"method":f"Validation threshold {threshold:g}",**audit["test_selected"]}]
    validation_rows=[{"method":"Reference threshold 0.5",**audit["validation_baseline"]},
                     {"method":f"Selected threshold {threshold:g}",**audit["validation_selected"]}]
    per_class=_per_class(cm0,"Argmax")+_per_class(cm1,f"Threshold {threshold:g}")
    changed=[]
    for status,mask in [("corrected_error",(baseline!=labels)&(selected==labels)),
                        ("introduced_error",(baseline==labels)&(selected!=labels)),
                        ("unchanged",baseline==selected)]:
        for i in np.flatnonzero(mask)[:2]:
            changed.append({"index":int(i),"text":raw[i]["text"],"true_label":int(labels[i]),
                            "argmax_label":int(baseline[i]),"selected_label":int(selected[i]),
                            "P_positive":float(probs[i,1]),"status":status})
    with run_context("threshold_analysis",{"source_runs":{k:p.name for k,p in paths.items()},
        "purpose":"audit and present frozen validation threshold on test; no new fitting"},root=root) as out:
        save_json(out/"analysis_source.json",_source_key(paths))
        save_json(out/"threshold_audit.json",audit)
        tables={"validation_comparison":validation_rows,"test_comparison":test_rows,
                "per_class_comparison":per_class,"changed_prediction_examples":changed,
                "validation_grid":audit["policy"]["curve"],
                "operating_points":[{"method":"Argmax","FPR":cm0[0,1]/cm0[0].sum(),"TPR":cm0[1,1]/cm0[1].sum()},
                                    {"method":f"Threshold {threshold:g}","FPR":cm1[0,1]/cm1[0].sum(),"TPR":cm1[1,1]/cm1[1].sum()}]}
        save_json(out/"tables.json",tables)
        summary={"selected_threshold":threshold,"selection_split":"validation","selection_metric":"macro_f1",
            "source_runs":{k:p.name for k,p in paths.items()},"source_training_run":Path(read_json(paths['evaluation']/"config.json")["evaluated_checkpoint"]).parent.name,
            "test_argmax":base,"test_selected":audit["test_selected"],"test_loss":audit["test_loss"],
            "test_loss_interpretation":"Measured after checkpoint reload; independent of decision threshold",
            "corrected_errors":int(((baseline!=labels)&(selected==labels)).sum()),
            "introduced_errors":int(((baseline==labels)&(selected!=labels)).sum()),
            "accuracy_delta_percentage_points":100*(audit["test_selected"]["accuracy"]-base["accuracy"]),
            "binary_f1_delta_percentage_points":100*(audit["test_selected"]["binary_f1"]-base["binary_f1"]),
            "macro_f1_delta_percentage_points":100*(audit["test_selected"]["macro_f1"]-base["macro_f1"]),
            "test_selection_performed":False,"local_model_execution":False,
            "roc_auc_reloaded_scores":float(roc_auc_score(labels,probs[:,1])),
            "average_precision_reloaded_scores":float(average_precision_score(labels,probs[:,1]))}
        save_json(out/"summary.json",summary)
        shutil.copyfile(paths["threshold_selection"]/"validation_threshold/threshold_curves.png",out/"validation_threshold_curves.png")
        shutil.copyfile(paths["inference"]/"inference.png",out/"inference_result.png")
        _metric_plot(validation_rows,out/"validation_comparison.png","Validation: reference vs selected threshold")
        _metric_plot(test_rows,out/"test_comparison.png","Test: same checkpoint, fixed decision policies")
        fig,axes=plt.subplots(1,2,figsize=(12,4.8))
        for ax,cm,title in zip(axes,[cm0,cm1],["Argmax",f"Validation threshold {threshold:g}"]):
            norm=cm/cm.sum(1,keepdims=True)
            annot=np.array([[f"{cm[i,j]}\n{norm[i,j]:.1%}" for j in [0,1]] for i in [0,1]])
            sns.heatmap(norm,annot=annot,fmt="",vmin=0,vmax=1,cmap="Blues",ax=ax,
                xticklabels=["Negative","Positive"],yticklabels=["Negative","Positive"])
            ax.set(xlabel="Predicted",ylabel="True",title=title)
        finish(fig,out/"confusion_comparison.png")
        _table_image(per_class,["method","label","precision","recall_TPR","F1","FPR","support"],out/"per_class_comparison.png","Per-class metrics: operating threshold trade-off")
        _table_image(changed,["text","true_label","argmax_label","selected_label","P_positive","status"],
                     out/"changed_predictions.png","Real examples affected by threshold",[.5,.07,.09,.1,.1,.14])
        fig,ax=plt.subplots(figsize=(8,4));x=np.arange(3)
        for i,(cm,method) in enumerate([(cm0,"Argmax"),(cm1,f"Threshold {threshold:g}")]):
            bars=ax.bar(x+(i-.5)*.35,[np.trace(cm),cm[0,1],cm[1,0]],.35,label=method)
            ax.bar_label(bars)
        ax.set(xticks=x,xticklabels=["Correct","False Positive","False Negative"],ylabel="Test examples");ax.legend();finish(fig,out/"error_tradeoff.png")
        fpr,tpr,_=roc_curve(labels,probs[:,1]);precision,recall,_=precision_recall_curve(labels,probs[:,1])
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        axes[0].plot(fpr,tpr,label="Reloaded-model ROC");axes[0].plot([0,1],[0,1],'--',color='gray')
        axes[1].plot(recall,precision,label="Reloaded-model PR")
        for cm,row,marker in zip([cm0,cm1],test_rows,['o','s']):
            axes[0].scatter(cm[0,1]/cm[0].sum(),cm[1,1]/cm[1].sum(),s=65,marker=marker,label=row["method"])
            axes[1].scatter(row["recall"],row["precision"],s=65,marker=marker,label=row["method"])
        axes[0].set(xlabel="FPR",ylabel="TPR");axes[1].set(xlabel="Recall",ylabel="Precision")
        for ax in axes:ax.legend(fontsize=8)
        finish(fig,out/"operating_points.png")
        print("Verified frozen threshold:",threshold,"on validation; no test selection performed")
        print("Saved comparative figures and tables:",out)
    return out

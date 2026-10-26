"""Phân tích và trực quan hóa artifacts thật, lưu thành run riêng để truy vết."""
from pathlib import Path
import json
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, brier_score_loss, log_loss)
from .evidence import (read_json, audit_finetuning, audit_exercise1, verify_manifest)
from .training.thresholds import export_threshold_analysis
from .utils.runs import ROOT, hash_file, run_context, save_json
from .utils.plots import finish


def _table_image(rows, columns, path, title, widths=None):
    """Bảng PNG có nguồn JSON song hành; không chỉ hiển thị trong notebook."""
    fig, ax = plt.subplots(figsize=(14, max(2.6, .65 * (len(rows) + 2))))
    ax.axis("off")
    body = []
    for row in rows:
        cells = []
        for key in columns:
            value = row.get(key, "")
            if isinstance(value, float): value = f"{value:.5f}"
            cells.append(textwrap.fill(str(value), 65 if key == "text" else 30))
        body.append(cells)
    table = ax.table(cellText=body, colLabels=columns, loc="center", cellLoc="left", colWidths=widths)
    table.auto_set_font_size(False); table.set_fontsize(9)
    for (i, _), cell in table.get_celld().items():
        cell.set_height(.83 / (len(rows) + 1))
        if i == 0: cell.set_facecolor("#cfe2f3")
        elif i % 2: cell.set_facecolor("#f3f6fa")
    ax.set_title(title, pad=18)
    finish(fig, path)


def _cache_key(source_run):
    """Cache phụ thuộc manifest nguồn và code phân tích, không chỉ tên run."""
    files = [Path(__file__), Path(__file__).parent / "training/thresholds.py"]
    return {"source_run": Path(source_run).name,
            "source_manifest_sha256": hash_file(Path(source_run) / "manifest.json"),
            "analysis_code": {p.name: hash_file(p) for p in files}}


def _cached(source_run, kind, root):
    registry = Path(root) / "config/workflow.json"
    if not registry.exists(): return None
    state = read_json(registry)
    expected = _cache_key(source_run)
    for row in reversed(state.get("runs", [])):
        if row["kind"] != kind or row["status"] != "completed": continue
        path = Path(root) / "outputs/runs" / row["id"]
        if not (path / "analysis_source.json").exists(): continue
        if read_json(path / "analysis_source.json") == expected:
            verify_manifest(path)
            return path
    return None


def ensure_exercise1_analysis(source_run, root=ROOT):
    """Đọc cache hợp lệ hoặc tạo mới bảng token và hình inference từ run thật."""
    audit = audit_exercise1(source_run)
    cached = _cached(source_run, "exercise1_analysis", root)
    if cached: return cached
    source = Path(source_run)
    with run_context("exercise1_analysis", {"source_run": source.name}, root=root) as out:
        save_json(out / "analysis_source.json", _cache_key(source))
        raw = read_json(source / "tokenizer_breakdown.json")
        tokens = [dict(position=i, tokens=t, input_ids=idx, attention_mask=mask)
                  for i, (t, idx, mask) in enumerate(zip(raw["tokens"], raw["input_ids"], raw["attention_mask"]))]
        predictions = read_json(source / "predictions.json")
        comparison = [dict(text=b["text"], pipeline_label=a["label"], manual_label=b["label"],
                     pipeline_score=a["score"], manual_score=b["score"], difference=abs(a["score"]-b["score"]))
                     for a, b in zip(predictions["pipeline"], predictions["manual"])]
        summary = {"source_run": source.name, "sentence": raw["text"], "tokenization": tokens,
                   "predictions": predictions["manual"], "comparison": comparison, "audit": audit}
        save_json(out / "summary.json", summary)
        _table_image(tokens, ["position", "tokens", "input_ids", "attention_mask"],
                     out / "tokenizer_table.png", "WordPiece tokenizer breakdown")
        _table_image(predictions["manual"], ["text", "label", "score"],
                     out / "sentiment_predictions.png", "Pretrained sentiment predictions", [.74, .13, .13])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        positions = np.arange(len(comparison))
        axes[0].bar(positions-.16, [r["pipeline_score"] for r in comparison], .32, label="Pipeline")
        axes[0].bar(positions+.16, [r["manual_score"] for r in comparison], .32, label="Manual")
        axes[0].set(xticks=positions, xticklabels=[f"Sentence {i+1}" for i in range(len(comparison))],
                    ylabel="Predicted class probability", ylim=(0, 1.1)); axes[0].legend()
        axes[1].bar(positions, [r["difference"] for r in comparison])
        axes[1].axhline(1e-5, color="red", linestyle="--", label="Tolerance 1e-5")
        axes[1].set(xlabel="Sentence index", ylabel="Absolute score difference", ylim=(0, 1.2e-5)); axes[1].legend()
        finish(fig, out / "pipeline_manual_comparison.png")
        print("Saved Exercise 1 visual artifacts from:", source.name)
    return out


def _reliability(labels, probs):
    predictions = probs.argmax(1); confidence = probs.max(1)
    correct = (predictions == labels).astype(float)
    indices = np.minimum((confidence * 10).astype(int), 9)
    bins = []; ece = 0.
    for i in range(10):
        mask = indices == i; count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean = float(confidence[mask].mean()) if count else None
        if count: ece += count / len(labels) * abs(accuracy - mean)
        bins.append({"lower": i/10, "upper": (i+1)/10, "count": count,
                     "mean_confidence": mean, "accuracy": accuracy})
    return bins, ece


def analyze_run(source_run, root=ROOT):
    """Tạo phân tích mô tả test; tuyệt đối không chọn threshold từ test."""
    source = Path(source_run)
    audit = audit_finetuning(source)
    rows = read_json(source / "all_predictions.json")
    labels = np.array([r["true_label"] for r in rows])
    probs = np.array([r["probabilities"] for r in rows])
    positive = probs[:, 1]; predicted = probs.argmax(1)
    with run_context("analysis", {"source_run": source.name, "purpose": "descriptive_test_analysis_only"}, root=root) as out:
        save_json(out / "analysis_source.json", _cache_key(source))
        bins, ece = _reliability(labels, probs)
        fpr, tpr, roc_thresholds = roc_curve(labels, positive)
        precision, recall, pr_thresholds = precision_recall_curve(labels, positive)
        summary = {"source_run": source.name, "audit": audit, "metrics": audit["metrics"],
                   "roc_auc": float(roc_auc_score(labels, positive)),
                   "average_precision": float(average_precision_score(labels, positive)),
                   "brier_score": float(brier_score_loss(labels, positive)), "confidence_ece": float(ece),
                   "recomputed_nll_from_saved_probabilities": float(log_loss(labels, probs / probs.sum(axis=1, keepdims=True), labels=[0, 1])),
                   "test_loss_status": "Trainer test loss not logged in historical run; reconstructed NLL is separate",
                   "nll_probability_preprocessing": "renormalize saved float32 probabilities per row to remove rounding drift",
                   "selection_performed": False, "validation_probabilities_available": (source / "validation_predictions.json").exists()}
        save_json(out / "analysis_summary.json", summary)
        save_json(out / "roc_pr_coordinates.json", {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                  "roc_thresholds": [float(v) if np.isfinite(v) else None for v in roc_thresholds],
                  "precision": precision.tolist(), "recall": recall.tolist(), "pr_thresholds": pr_thresholds.tolist()})
        save_json(out / "reliability_bins.json", bins)
        history = read_json(source / "trainer_history.json"); epochs=[]
        for row in history:
            if "eval_accuracy" not in row: continue
            train_eval = next(r for r in history if r.get("epoch") == row["epoch"] and "train_accuracy" in r)
            train_log = next(r for r in history if r.get("epoch") == row["epoch"] and "loss" in r)
            epochs.append({"epoch": row["epoch"], "step": row["step"], "optimization_loss": train_log["loss"],
                 "train_eval_loss": train_eval["train_loss"], "train_accuracy": train_eval["train_accuracy"],
                 "validation_loss": row["eval_loss"], "validation_accuracy": row["eval_accuracy"]})
        cm=np.array(audit["confusion_matrix"]); per_class=[]
        for i,name in enumerate(["NEGATIVE", "POSITIVE"]):
            tp=int(cm[i,i]);fn=int(cm[i].sum()-tp);fp=int(cm[:,i].sum()-tp);tn=int(cm.sum()-tp-fn-fp)
            report=audit["classification_report"][name]
            per_class.append({"label":name,"precision":report["precision"],"recall_TPR":tp/(tp+fn),
                              "F1":report["f1-score"],"FPR":fp/(fp+tn),"support":int(report["support"])})
        comparison=[{"model":"Constant NEGATIVE",**audit["constant_negative_baseline"]},
                    {"model":"Fine-tuned DistilBERT (argmax)",**audit["metrics"]}]
        config=read_json(source / "config.json"); stats=read_json(source / "data_statistics.json")
        versions=read_json(source / "environment_snapshot.json")
        lengths=[len(r["text"].split()) for r in rows]
        tables={"epoch_history":epochs,"per_class_metrics":per_class,"metrics_comparison":comparison,
                "split_sizes":[{"split":k,"samples":v} for k,v in stats["split_sizes"].items()],
                "token_statistics":[stats["train_token_lengths"]],"training_config":[config["exercise2"]],
                "environment_versions":[{"library":k,"version":v} for k,v in versions["versions"].items()],
                "test_word_length_statistics":{"mean":float(np.mean(lengths)),"median":float(np.median(lengths)),"max":int(max(lengths))},
                "artifact_catalog":[{"file":p.name,"source_run":source.name} for p in sorted(source.glob('*.png'))]}
        save_json(out / "tables.json", tables)
        fig,ax=plt.subplots(figsize=(7,4)); ax.bar(["Negative", "Positive"],np.bincount(labels,minlength=2))
        ax.set(ylabel="Test examples",title="Observed test class distribution: 533 / 533");finish(fig,out/"class_distribution.png")
        fig,ax=plt.subplots(figsize=(9,4));sns.histplot(x=lengths,hue=labels,bins=30,element="step",ax=ax)
        ax.set(xlabel="Whitespace-separated words (not WordPiece tokens)",ylabel="Test examples",title="Review length by true label");finish(fig,out/"review_length_distribution.png")
        fig,ax=plt.subplots(figsize=(9,4))
        for label,name in [(0,"True Negative"),(1,"True Positive")]:
            ax.hist(positive[labels==label],bins=np.linspace(0,1,31),alpha=.6,label=name)
        ax.axvline(.5,linestyle="--",color="black",label="Reference 0.5 (not fitted)")
        ax.set(xlabel="P(Positive)",ylabel="Test examples");ax.legend();finish(fig,out/"probability_distribution.png")
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        axes[0].plot(fpr,tpr,label=f"ROC AUC={summary['roc_auc']:.4f}");axes[0].plot([0,1],[0,1],'--',color='gray')
        axes[0].set(xlabel="False positive rate",ylabel="True positive rate");axes[0].legend()
        axes[1].plot(recall,precision,label=f"AP={summary['average_precision']:.4f}")
        axes[1].axhline(labels.mean(),linestyle='--',color='gray',label='Positive prevalence')
        axes[1].set(xlabel="Recall",ylabel="Precision");axes[1].legend();finish(fig,out/"roc_pr_curves.png")
        observed=[b for b in bins if b['count']]
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        axes[0].plot([0,1],[0,1],'--',color='gray')
        axes[0].plot([b['mean_confidence'] for b in observed],[b['accuracy'] for b in observed],marker='o')
        axes[0].set(xlabel="Mean predicted-class confidence",ylabel="Accuracy in bin",title=f"Top-label reliability; ECE={ece:.4f}",xlim=(0,1),ylim=(0,1))
        axes[1].bar([b['lower']+.05 for b in bins],[b['count'] for b in bins],width=.085)
        axes[1].set(xlabel="Predicted-class confidence",ylabel="Number of test examples");finish(fig,out/"calibration_curve.png")
        _table_image(per_class,["label","precision","recall_TPR","F1","FPR","support"],out/"per_class_metrics.png","Per-class test metrics (one-vs-rest)")
        keys=['accuracy','macro_f1','macro_precision','macro_recall'];positions=np.arange(len(keys))
        fig,ax=plt.subplots(figsize=(9,4))
        for i,row in enumerate(comparison):ax.bar(positions+(i-.5)*.35,[row[k] for k in keys],.35,label=row['model'])
        ax.set(xticks=positions,xticklabels=keys,ylim=(0,1.08));ax.legend(fontsize=8);finish(fig,out/"metric_comparison.png")
        export_threshold_analysis(labels,positive,out/"threshold_test_diagnostic",split="test",select=False)
        print("Saved descriptive test analysis; no test-derived threshold selected.")
    return out


def ensure_analysis(source_run, root=ROOT):
    """Tái sử dụng phân tích đã băm đúng code hoặc tạo run mới, không sửa run gốc."""
    audit_finetuning(source_run)
    return _cached(source_run,"analysis",root) or analyze_run(source_run,root)

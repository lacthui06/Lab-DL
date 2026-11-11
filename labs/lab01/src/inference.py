"""Evaluate/infer using the same prepared data and architecture as training."""
import json
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, log_loss, accuracy_score
from .training.comparison import CONFIGS, loaders
from .models.mlp import FashionMLP
from .utils.helpers import DEVICE, CLASS_NAMES, SOURCE_DIR, MODELS_DIR, load_checkpoint, save_checkpoint
from .utils.runs import Run, write_json


def resolve_run_path(target="final"):
    """Resolve a model name ('final', 'MainMLP', 'SmallMLP', 'MainMLP_FE') or path to a run directory."""
    if isinstance(target, Path):
        p = target
    else:
        p = Path(str(target).strip())

    # 1. Direct path check (absolute or relative to current working directory)
    if p.is_file() and p.name == "best.pt":
        return p.parent
    if (p / "config.json").exists() and (p / "best.pt").exists():
        return p

    # 1b. Check models/manifest.json for source run mapping
    manifest_path = MODELS_DIR / "manifest.json"
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            minfo = m.get("models", {})
            target_str = str(target).strip()
            target_lower = target_str.lower()
            alias_map = {"final": m.get("winner_name", "final"), "best": m.get("winner_name", "final"), "winner": m.get("winner_name", "final")}
            lookup_key = alias_map.get(target_lower, target_str)
            for k, info in minfo.items():
                if k.lower() == lookup_key.lower() or k == lookup_key:
                    cand = Path(info.get("source_run", ""))
                    if (cand / "config.json").exists() and (cand / "best.pt").exists():
                        return cand
        except Exception:
            pass

    # 2. Check relative to SOURCE_DIR
    candidate = SOURCE_DIR / p
    if candidate.is_file() and candidate.name == "best.pt":
        return candidate.parent
    if (candidate / "config.json").exists() and (candidate / "best.pt").exists():
        return candidate

    # 3. Logical name check in config/workflow.json
    workflow_path = SOURCE_DIR / "config" / "workflow.json"
    if workflow_path.exists():
        try:
            wf = json.loads(workflow_path.read_text(encoding="utf-8"))
            runs = wf.get("runs", {})
            models = runs.get("models", {})
            target_str = str(target).strip()
            target_lower = target_str.lower()
            key_map = {
                "final": runs.get("final"),
                "winner": runs.get("final"),
                "best": runs.get("final"),
                "smallmlp": models.get("SmallMLP"),
                "mainmlp": models.get("MainMLP"),
                "mainmlp_fe": models.get("MainMLP_FE"),
            }
            for k, rel_path in models.items():
                if k.lower() == target_lower or k == target_str:
                    key_map[target_str] = rel_path
                    break

            mapped = key_map.get(target_str) or key_map.get(target_lower)
            if mapped:
                resolved = SOURCE_DIR / mapped
                if (resolved / "config.json").exists() and (resolved / "best.pt").exists():
                    return resolved
        except Exception:
            pass

    # 4. Search by run directory prefix/substring under outputs/runs
    runs_dir = SOURCE_DIR / "outputs" / "runs"
    if runs_dir.exists():
        target_str = str(target).strip()
        matches = [m for m in runs_dir.glob(f"*{target_str}*") if (m / "config.json").exists() and (m / "best.pt").exists()]
        if matches:
            matches.sort(key=lambda x: x.name, reverse=True)
            return matches[0]

    raise FileNotFoundError(
        f"Cannot find model or checkpoint for '{target}'. "
        f"Valid names: 'final', 'SmallMLP', 'MainMLP', 'MainMLP_FE', "
        f"or a valid path containing config.json and best.pt."
    )


def load_model(target="final", device=None):
    """Loads a trained FashionMLP model directly by name, path, or from models/ in eval mode.

    Args:
        target: Model name ('final', 'MainMLP', 'SmallMLP', 'MainMLP_FE') or path to checkpoint / run directory.
        device: Torch device (defaults to DEVICE).

    Returns:
        FashionMLP model loaded with weights and set to eval().
    """
    device = device or DEVICE
    target_str = str(target).strip()
    target_lower = target_str.lower()

    # 1. Direct checkpoint check in models/
    direct_pt = None
    if isinstance(target, Path) and target.is_file() and target.suffix == ".pt":
        direct_pt = target
    elif (MODELS_DIR / f"{target_str}.pt").exists():
        direct_pt = MODELS_DIR / f"{target_str}.pt"
    elif target_lower in ("final", "winner", "best") and (MODELS_DIR / "best_model.pt").exists():
        direct_pt = MODELS_DIR / "best_model.pt"

    if direct_pt is not None:
        state = torch.load(str(direct_pt), map_location=device, weights_only=False)
        state_dict = state.get("model_state_dict", state)
        meta = state.get("metadata", {})
        linear = [v for k, v in state_dict.items() if k.endswith(".weight") and v.ndim == 2]
        dim = meta.get("in_features", linear[0].shape[1])
        hidden = meta.get("hidden_dims", [w.shape[0] for w in linear[:-1]])
        classes = meta.get("num_classes", linear[-1].shape[0])
        model = FashionMLP(in_features=dim, hidden_dims=hidden, num_classes=classes)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        val_acc = meta.get("val_acc")
        acc_text = ""
        if isinstance(val_acc, (int, float)):
            val_pct = val_acc if val_acc > 1.0 else val_acc * 100
            acc_text = f" (best val acc: {val_pct:.2f}%)"

        try:
            rel = direct_pt.relative_to(SOURCE_DIR)
        except ValueError:
            rel = direct_pt
        print(f"Loaded model directly from {rel}{acc_text}")
        return model

    # 2. Fall back to run-based resolution
    run_path = resolve_run_path(target)
    config = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
    model_name = config.get("name", "MainMLP")
    default_dim, default_hidden = CONFIGS.get(model_name, (784, (256, 128)))
    dim = config.get("in_features", default_dim)
    hidden = config.get("hidden_dims", default_hidden)
    model = FashionMLP(in_features=dim, hidden_dims=hidden)
    ckpt_path = run_path / "best.pt"
    meta = load_checkpoint(model, ckpt_path, device)
    model.eval()

    val_acc = meta.get("val_acc")
    acc_text = ""
    if isinstance(val_acc, (int, float)):
        val_pct = val_acc if val_acc > 1.0 else val_acc * 100
        acc_text = f" (best val acc: {val_pct:.2f}%)"

    try:
        rel = ckpt_path.relative_to(SOURCE_DIR)
    except ValueError:
        rel = ckpt_path
    print(f"Loaded {model_name} from {rel}{acc_text}")
    return model


def load_trained_run(path, data_path=None):
    path = resolve_run_path(path)
    cfg = json.loads((path / "config.json").read_text())
    default_dim, default_hidden = CONFIGS[cfg["name"]]
    dim, hidden = cfg.get("in_features", default_dim), cfg.get("hidden_dims", default_hidden)
    model = FashionMLP(in_features=dim, hidden_dims=hidden)
    load_checkpoint(model, path / "best.pt", DEVICE)
    model.eval()
    _, _, test = loaders(data_path or cfg["data_path"], cfg["name"], cfg["seed"], cfg["batch_size"])
    return model, test


def evaluate_run(path,data_path=None):
    with Run("evaluation",{"source_run":str(Path(path).resolve()),"data_override":str(data_path) if data_path else None}) as run:
        model,test=load_trained_run(path,data_path)
        probabilities=[];labels=[]
        with torch.inference_mode():
            for x,y in test:
                probabilities.append(model(x.to(DEVICE)).softmax(1).cpu().numpy());labels.append(y.numpy())
        probabilities=np.concatenate(probabilities);labels=np.concatenate(labels)
        np.savez_compressed(run.path/"test_predictions.npz",probabilities=probabilities,labels=labels)
        report=classification_report(labels,probabilities.argmax(1),output_dict=True,zero_division=0)
        write_json(run.path/"classification_report.json",report)
        metrics=dict(accuracy=accuracy_score(labels,probabilities.argmax(1)),macro_f1=report["macro avg"]["f1-score"],loss=log_loss(labels,probabilities,labels=list(range(10))))
        write_json(run.path/"metrics.json",metrics);print(metrics)
    return run.path


def infer_sample(path,index=0,data_path=None):
    with Run("inference",{"source_run":str(Path(path).resolve()),"index":index}) as run:
        model,test=load_trained_run(path,data_path)
        x,y=test.dataset[index]
        with torch.inference_mode():prob=model(x[None].to(DEVICE)).softmax(1)[0].cpu().numpy()
        result=dict(index=index,true_label=int(y),predicted_label=int(prob.argmax()),probabilities=prob.tolist())
        write_json(run.path/"prediction.json",result);print(result)
    return run.path


def save_load_model(path, data_path=None):
    """Export/reload the selected checkpoint, verify roundtrip and evaluate the full test split."""
    path = Path(path)
    config = json.loads((path/"config.json").read_text(encoding="utf-8"))
    default_dim, default_hidden = CONFIGS[config["name"]]
    config.setdefault("in_features", default_dim)
    config.setdefault("hidden_dims", list(default_hidden))
    config.update(source_run=str(path.resolve()),
                  data_path=str(Path(data_path or config["data_path"]).resolve()))
    with Run("model_save_load", config) as run:
        model, test = load_trained_run(path, config["data_path"])
        x, y = next(iter(test))
        x, y = x[:16], y[:16]
        with torch.inference_mode():
            before = model(x.to(DEVICE)).softmax(1).cpu()
        save_checkpoint(model, run.path/"best.pt", metadata={
            "source_checkpoint":str((path/"best.pt").resolve()),
            "selection":"best validation checkpoint from refit",
            "in_features":config["in_features"], "hidden_dims":config["hidden_dims"]})
        restored, _ = load_trained_run(run.path)
        with torch.inference_mode():
            after = restored(x.to(DEVICE)).softmax(1).cpu()
        max_error = float((before-after).abs().max())
        passed = bool(torch.allclose(before, after, atol=1e-6, rtol=1e-5))
        np.savez_compressed(run.path/"roundtrip_predictions.npz",labels=y.numpy(),
                            before=before.numpy(),after=after.numpy())
        result = dict(passed=passed,samples=len(y),max_probability_error=max_error,
                      predicted_labels_unchanged=bool(torch.equal(before.argmax(1),after.argmax(1))),
                      checkpoint="best.pt",source_run=str(path.resolve()))
        write_json(run.path/"roundtrip.json",result)
        (run.path/"model_architecture.txt").write_text(str(restored),encoding="utf-8")
        print("Saved checkpoint:",run.path/"best.pt")
        print("Fresh model loaded in eval mode:",restored)
        print("Save/load verification:",result)
        if not passed:
            raise RuntimeError("Model predictions changed after save/load")
        # Test the freshly restored instance on the entire held-out split.
        probabilities, labels = [], []
        with torch.inference_mode():
            for batch_x, batch_y in test:
                probabilities.append(restored(batch_x.to(DEVICE)).softmax(1).cpu().numpy())
                labels.append(batch_y.numpy())
        probabilities, labels = np.concatenate(probabilities), np.concatenate(labels)
        predicted = probabilities.argmax(1)
        np.savez_compressed(run.path/"test_predictions.npz",labels=labels,probabilities=probabilities)
        report = classification_report(labels,predicted,labels=list(range(10)),output_dict=True,zero_division=0)
        write_json(run.path/"classification_report.json",report)
        metrics = dict(samples=len(labels),accuracy=float(accuracy_score(labels,predicted)),
                       macro_f1=report["macro avg"]["f1-score"],
                       loss=float(log_loss(labels,probabilities,labels=list(range(10)))))
        write_json(run.path/"test_metrics.json",metrics)
        # Include successes and errors; choose by sample order, never by confidence.
        correct = np.flatnonzero(predicted==labels)[:8]
        wrong = np.flatnonzero(predicted!=labels)[:8]
        indices = np.r_[correct,wrong]
        if len(indices)<16:
            indices=np.r_[indices,np.setdiff1d(np.arange(len(labels)),indices)[:16-len(indices)]]
        rows=[]
        fig,axes=plt.subplots(4,4,figsize=(13,12))
        for ax in axes.flat: ax.axis("off")
        for ax,i in zip(axes.flat,indices):
            image=test.dataset.tensors[0][i,:784].numpy().reshape(28,28)
            confidence=float(probabilities[i,predicted[i]])
            ax.imshow(image,cmap="gray")
            ax.set_title(f"#{i} True: {CLASS_NAMES[labels[i]]}\nPred: {CLASS_NAMES[predicted[i]]} ({confidence:.1%})",
                         fontsize=9,color="green" if predicted[i]==labels[i] else "red")
            rows.append(dict(test_index=int(i),true_label=int(labels[i]),true_class=CLASS_NAMES[labels[i]],
                             predicted_label=int(predicted[i]),predicted_class=CLASS_NAMES[predicted[i]],
                             probability=confidence,correct=bool(predicted[i]==labels[i])))
        fig.suptitle("Reloaded model: test predictions (green=correct, red=incorrect)")
        fig.tight_layout();fig.savefig(run.path/"loaded_model_test_samples.png",dpi=150);plt.close(fig)
        pd.DataFrame(rows).to_csv(run.path/"loaded_model_test_samples.csv",index=False)
        print("Reloaded model, full test split:",metrics)
    return run.path, restored

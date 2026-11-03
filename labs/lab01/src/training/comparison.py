"""Three controlled MLP experiments with persisted numerical evidence."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score, log_loss
from ..models.mlp import FashionMLP
from .trainer import Trainer, sanity_overfit_check, build_optimizer
from ..utils.helpers import DEVICE, set_seed, load_checkpoint
from ..utils.runs import Run, write_json

from ..utils.helpers import SOURCE_DIR
_WORKFLOW = json.loads((SOURCE_DIR / "config/workflow.json").read_text(encoding="utf-8"))
CONFIGS = {name: (cfg["in_features"], tuple(cfg["hidden_dims"]))
           for name, cfg in _WORKFLOW["models"].items()}


def loaders(data_path, name, seed=42, batch_size=128):
    d = np.load(data_path)
    result = []
    for split in ("train", "val", "test"):
        x = d[split+"_pixels"]
        if name == "MainMLP_FE":
            x = np.concatenate((x, d[split+"_features"]), axis=1)
        ds = TensorDataset(torch.from_numpy(x.copy()).float(), torch.from_numpy(d[split+"_labels"]).long())
        result.append(DataLoader(ds, batch_size=batch_size, shuffle=split=="train",
                      generator=torch.Generator().manual_seed(seed)))
    return result


def run_model(data_path, name, epochs=10, seed=42, batch_size=128, learning_rate=0.001, optimizer_name="Adam", weight_decay=0.0, momentum=0.9, model_config=None):
    dim, hidden = CONFIGS[name]
    if model_config is not None:
        dim, hidden = model_config["in_features"], tuple(model_config["hidden_dims"])
    config = dict(name=name, epochs=epochs, seed=seed, batch_size=batch_size,
                  learning_rate=learning_rate, data_path=str(Path(data_path).resolve()), optimizer=optimizer_name, weight_decay=weight_decay, momentum=momentum,
                  device=str(DEVICE))
    config.update(in_features=dim, hidden_dims=list(hidden), scheduler="CosineAnnealingLR")
    with Run(name, config) as run:
        import hashlib
        config["data_sha256"] = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
        write_json(run.path / "config.json", config)
        print(f"Training device: {DEVICE}; epochs: {epochs}")
        set_seed(seed)
        train, val, test = loaders(data_path, name, seed, batch_size)
        probe = FashionMLP(in_features=dim, hidden_dims=hidden)
        x, y = next(iter(train))
        ok, loss, acc = sanity_overfit_check(probe, x[:8], y[:8], device=DEVICE, num_steps=120, target_loss=0.15)
        write_json(run.path / "sanity.json", dict(passed=bool(ok), loss=loss, accuracy=acc))
        if not ok:
            raise RuntimeError("One-batch sanity failed")
        set_seed(seed)
        train, val, test = loaders(data_path, name, seed, batch_size)
        model = FashionMLP(in_features=dim, hidden_dims=hidden).to(DEVICE)
        opt = build_optimizer(optimizer_name, model, learning_rate=learning_rate,
                              weight_decay=weight_decay, momentum=momentum)
        trainer = Trainer(model, torch.nn.CrossEntropyLoss(), opt, DEVICE,
                          torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs))
        history = {k: [] for k in ("train_loss","train_acc","val_loss","val_acc")}
        best, start = -1, time.perf_counter()
        for epoch in range(epochs):
            # One epoch at a time allows durable partial history if interrupted.
            h = trainer.fit(train, val, epochs=1, verbose=False)
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train loss: {h['train_loss'][0]:.4f} | Val accuracy: {h['val_acc'][0]:.2f}%")
            for k in history:
                history[k].extend(h[k])
            if h["val_acc"][0] > best:
                best = h["val_acc"][0]
                torch.save({"model_state_dict": model.state_dict(),
                            "metadata": {"epoch":epoch+1, "val_acc":best}}, run.path / "best.pt")
            write_json(run.path / "history.json", history)
            pd.DataFrame(history).to_csv(run.path / "history.csv", index_label="epoch_zero_based")
        seconds = time.perf_counter()-start
        load_checkpoint(model, run.path / "best.pt", DEVICE)
        probs, targets = [], []
        model.eval()
        with torch.inference_mode():
            for x, y in test:
                probs.append(model(x.to(DEVICE)).softmax(1).cpu().numpy())
                targets.append(y.numpy())
        probs, targets = np.concatenate(probs), np.concatenate(targets)
        np.savez_compressed(run.path / "test_predictions.npz", probabilities=probs, labels=targets,
                            sample_indices=np.arange(len(targets)))
        # Identical test images for all prediction grids; strip only feature columns.
        np.save(run.path / "grid_pixels.npy", test.dataset.tensors[0][:16,:784].numpy())
        report = classification_report(targets, probs.argmax(1), labels=list(range(10)), output_dict=True, zero_division=0)
        write_json(run.path / "classification_report.json", report)
        pd.DataFrame(report).T.to_csv(run.path / "classification_report.csv")
        metrics = dict(name=name, seed=seed, accuracy=accuracy_score(targets,probs.argmax(1)),
                       macro_f1=report["macro avg"]["f1-score"], precision=report["macro avg"]["precision"],
                       recall=report["macro avg"]["recall"], loss=log_loss(targets,probs,labels=list(range(10))),
                       train_seconds=seconds, parameters=sum(p.numel() for p in model.parameters()),
                       best_epoch=int(np.argmax(history["val_acc"]))+1)
        write_json(run.path / "metrics.json", metrics)
        from ..utils.plots import plot_loss_and_acc_curves
        import matplotlib.pyplot as plt
        fig = plot_loss_and_acc_curves(history, title=name, save_path=run.path / "learning_curves.png")
        plt.close(fig)
        print(metrics)
    return run.path

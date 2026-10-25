"""End-to-end transfer-learning suite; test data is evaluated after selection."""
import argparse
import contextlib
import copy
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import traceback
import uuid

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from .data import image_loader, image_transform, prepare_indices, export_eda, file_hash
from .models import (SPECS, build_model, classifier, features, configure_trainable,
                     parameter_counts, training_mode)
from .reporting import export_evaluation, summarize_suite

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def cached_features(model, name, data_dir, split, indices, cfg, device, dest):
    model.eval()
    xs, ys, ids = [], [], []
    loader = image_loader(data_dir, split, indices, name, cfg["feature_batch_size"], cfg["seed"], image_size=cfg["image_size"])
    start = time.perf_counter()
    with torch.inference_mode():
        for batch_no, (images, labels, original) in enumerate(loader):
            xs.append(features(model, name, images.to(device)).cpu())
            ys.append(labels)
            ids.append(original)
            if batch_no % 25 == 0:
                print(f"{name} extract {split}: {min((batch_no + 1) * loader.batch_size, len(indices))}/{len(indices)}", flush=True)
    sync(device)
    result = (torch.cat(xs), torch.cat(ys), torch.cat(ids))
    torch.save(dict(zip(("features", "labels", "indices"), result)), dest)
    return result, time.perf_counter() - start


def tensor_loader(data, batch, seed, shuffle=False):
    return DataLoader(TensorDataset(*data), batch_size=batch, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed))


def pass_epoch(model, loader, device, optimizer=None, mode_fn=None):
    training = optimizer is not None
    if training:
        (mode_fn or (lambda m: m.train()))(model)
    else:
        model.eval()
    total_loss, correct, count = 0., 0, 0
    with torch.enable_grad() if training else torch.inference_mode():
        for batch_number, (inputs, labels, *_) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
            if batch_number and batch_number % 250 == 0:
                print(f"  {'train' if training else 'val'}: {count}/{len(loader.dataset)}", flush=True)
    return {"loss": total_loss / count, "accuracy": correct / count}


def fit(model, train, val, optimizer, epochs, device, folder, metadata, mode_fn=None):
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "config.json", metadata)
    writer = SummaryWriter(str(folder / "tensorboard"))
    writer.add_text("configuration", json.dumps(metadata, indent=2))
    history, best = [], -1.
    sync(device)
    start = time.perf_counter()
    try:
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            tr = pass_epoch(model, train, device, optimizer, mode_fn)
            va = pass_epoch(model, val, device)
            row = {"epoch": epoch, "train_loss": tr["loss"], "train_accuracy": tr["accuracy"],
                   "val_loss": va["loss"], "val_accuracy": va["accuracy"],
                   "epoch_seconds": time.perf_counter() - epoch_start}
            history.append(row)
            pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
            for i, group in enumerate(optimizer.param_groups):
                writer.add_scalar(f"learning_rate/group_{i}", group["lr"], epoch)
            for param_name, param in model.named_parameters():
                if param.requires_grad and param.ndim > 1:
                    writer.add_histogram(f"weights/{param_name}", param, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f"gradients/{param_name}", param.grad, epoch)
                    break
            writer.flush()
            # Ties retain the earliest checkpoint; selection does not consult test.
            if va["accuracy"] > best:
                best = va["accuracy"]
                torch.save({"state_dict": model.state_dict(), "metadata": metadata,
                            "best_epoch": epoch, "val_accuracy": best}, folder / "best.pt")
            print(f"{folder.name} epoch {epoch}/{epochs}: train={tr['accuracy']:.4f}, val={va['accuracy']:.4f}", flush=True)
        sync(device)
        result = {"best_val_accuracy": best, "train_seconds": time.perf_counter() - start,
                  "best_epoch": torch.load(folder / "best.pt", weights_only=True, map_location="cpu")["best_epoch"]}
        writer.add_hparams({k: metadata[k] for k in ("model", "stage", "batch_size", "lr")},
                           {"best_val_accuracy": best})
        writer.flush()
        write_json(folder / "training_summary.json", result)
        return result
    finally:
        writer.close()


def predict(model, loader, device):
    model.eval()
    probs, labels, indices = [], [], []
    with torch.inference_mode():
        for x, y, idx in loader:
            probs.append(model(x.to(device)).softmax(1).cpu().numpy())
            labels.append(y.numpy())
            indices.append(idx.numpy())
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(indices)


def run_model(name, cfg, selected, data_dir, run, device):
    seed_all(cfg["seed"])
    folder = run / name
    folder.mkdir(exist_ok=True)
    model = build_model(name).to(device)
    configure_trainable(model, name, "head")
    (folder / "architecture.txt").write_text(str(model), encoding="utf-8")
    params = [{"name": n, "shape": str(tuple(p.shape)), "parameters": p.numel(),
               "head_stage_trainable": p.requires_grad} for n, p in model.named_parameters()]
    pd.DataFrame(params).to_csv(folder / "parameters.csv", index=False)
    preprocess = SPECS[name][1].transforms()
    write_json(folder / "preprocessing.json", {"weights": str(SPECS[name][1]),
               "dataset": "CIFAR10", "channels": "RGB", "image_size": cfg["image_size"],
               "transform": str(image_transform(name, cfg['image_size'])),
               "augmentation": "none; deterministic cached features and fair stage comparison"})
    batch = next(iter(image_loader(data_dir, "train", selected["train"][:16], name, 16, cfg["seed"], image_size=cfg["image_size"])))
    with SummaryWriter(str(folder / "input_preview")) as writer:
        image = batch[0] * torch.tensor(preprocess.std)[None, :, None, None] + torch.tensor(preprocess.mean)[None, :, None, None]
        writer.add_image("input_images", make_grid(image.clamp(0, 1), nrow=4))
        writer.add_text("architecture", str(model))
        with torch.no_grad():
            writer.add_graph(model.eval(), batch[0][:1].to(device))
    train_features, train_sec = cached_features(model, name, data_dir, "train", selected["train"], cfg, device, folder / "train_features.pt")
    val_features, val_sec = cached_features(model, name, data_dir, "val", selected["val"], cfg, device, folder / "val_features.pt")
    initial_head = copy.deepcopy(classifier(model, name).state_dict())
    trials = []
    for number, (lr, bs) in enumerate(itertools.product(cfg["learning_rates"], cfg["batch_sizes"])):
        seed_all(cfg["seed"])
        head = classifier(model, name)
        head.load_state_dict(initial_head)
        meta = {"model": name, "stage": "head", "lr": lr, "batch_size": bs,
                "seed": cfg["seed"], "weights": str(SPECS[name][1]),
                "dataset": "CIFAR10", "image_size": cfg["image_size"],
                "data_sha256": file_hash(run / "selected_indices.npz")}
        trial = folder / f"head_trial_{number:02d}"
        result = fit(head, tensor_loader(train_features, bs, cfg["seed"], True),
                     tensor_loader(val_features, bs, cfg["seed"]),
                     torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=cfg["weight_decay"]),
                     cfg["head_epochs"], device, trial, meta)
        trials.append({"trial": number, "lr": lr, "batch_size": bs, **result})
        pd.DataFrame(trials).to_csv(folder / "search_results.csv", index=False)
    winner = max(trials, key=lambda r: r["best_val_accuracy"])
    chosen = folder / f"head_trial_{winner['trial']:02d}"
    checkpoint = torch.load(chosen / "best.pt", map_location=device, weights_only=True)
    classifier(model, name).load_state_dict(checkpoint["state_dict"])
    head_folder = folder / "head_selected"
    head_folder.mkdir()
    head_meta = {**checkpoint["metadata"], "selected_trial": winner["trial"],
                 "parameter_counts": parameter_counts(model)}
    torch.save({"state_dict": model.state_dict(), "metadata": head_meta,
                "best_epoch": checkpoint["best_epoch"]}, head_folder / "best.pt")
    pd.read_csv(chosen / "history.csv").to_csv(head_folder / "history.csv", index=False)
    write_json(head_folder / "training_summary.json", {**winner,
               "feature_extraction_train_val_seconds": train_sec + val_sec,
               "all_search_training_seconds": sum(t["train_seconds"] for t in trials)})
    # Selection is persisted before ANY test images are loaded for this model.
    write_json(folder / "selection.json", {"criterion": "max validation accuracy, earliest trial on tie",
               "head": winner, "finetune_policy": "last block + classifier, fixed config, best validation epoch"})
    configure_trainable(model, name, "finetune")
    head_ids = {id(p) for p in classifier(model, name).parameters()}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": cfg["finetune_backbone_lr"]},
        {"params": classifier(model, name).parameters(), "lr": cfg["finetune_head_lr"]}],
        weight_decay=cfg["weight_decay"])
    bs = winner["batch_size"]
    meta = {"model": name, "stage": "finetune", "lr": cfg["finetune_head_lr"],
            "backbone_lr": cfg["finetune_backbone_lr"], "batch_size": bs,
            "seed": cfg["seed"], "weights": str(SPECS[name][1]),
            "dataset": "CIFAR10", "image_size": cfg["image_size"],
            "initialization": "best head checkpoint", "batchnorm": "running statistics frozen",
            "parameter_counts": parameter_counts(model)}
    fine_folder = folder / "finetune"
    seed_all(cfg["seed"])
    fine_result = fit(model,
        image_loader(data_dir, "train", selected["train"], name, bs, cfg["seed"], True, cfg["image_size"]),
        image_loader(data_dir, "val", selected["val"], name, bs, cfg["seed"], image_size=cfg["image_size"]),
        optimizer, cfg["finetune_epochs"], device, fine_folder, meta, training_mode)
    # Evaluate two predeclared conditions after their checkpoints are locked.
    rows = []
    for stage, stage_folder, result, metadata in [
        ("head", head_folder, winner, head_meta),
        ("finetune", fine_folder, fine_result, meta)]:
        saved = torch.load(stage_folder / "best.pt", map_location=device, weights_only=True)
        restored = build_model(name, pretrained=False).to(device)
        restored.load_state_dict(saved["state_dict"])
        restored.eval()
        loader = image_loader(data_dir, "test", selected["test"], name, bs, cfg["seed"], image_size=cfg["image_size"])
        sync(device)
        start = time.perf_counter()
        probabilities, labels, indices = predict(restored, loader, device)
        sync(device)
        elapsed = time.perf_counter() - start
        metric = export_evaluation(stage_folder, probabilities, labels, indices, data_dir)
        # Real roundtrip: compare restored instance to in-memory saved weights.
        model.load_state_dict(saved["state_dict"])
        model.eval()
        sample = next(iter(loader))[0][:8].to(device)
        with torch.inference_mode():
            error = (model(sample).softmax(1) - restored(sample).softmax(1)).abs().max().item()
        write_json(stage_folder / "roundtrip.json", {"samples": len(sample),
                   "max_probability_error": error, "passed": error <= 1e-6})
        if error > 1e-6:
            raise AssertionError("Checkpoint roundtrip failed")
        rows.append({"model": name, "stage": stage, **metric,
                     "best_val_accuracy": result["best_val_accuracy"],
                     "train_seconds": result["train_seconds"], "test_seconds": elapsed,
                     "feature_extraction_train_val_seconds": train_sec + val_sec,
                     **metadata["parameter_counts"]})
        del restored
    return rows


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        self.log.flush()

    def flush(self):
        self.stream.flush()
        self.log.flush()


def run_suite(config_path, work_dir=None, device_override="auto"):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if cfg.get("dataset") != "CIFAR10":
        raise ValueError("This pipeline requires a CIFAR10 config; legacy FashionMNIST runs are read-only.")
    torch.set_num_threads(cfg["threads"])
    seed_all(cfg["seed"])
    work = Path(work_dir or ROOT).resolve()
    torch.hub.set_dir(str(work / "data" / "pretrained"))
    data_dir = work / "data"
    run = work / "outputs" / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + cfg["profile"] + "_" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_override == "auto" else device_override)
    write_json(run / "config.json", cfg)
    write_json(run / "environment.json", {"python": sys.version, "torch": torch.__version__,
               "torchvision": __import__("torchvision").__version__, "device": str(device),
               "platform": platform.platform(), "cpu_count": os.cpu_count()})
    source = {str(p.relative_to(ROOT)): file_hash(p) for p in ROOT.rglob("*.py")
              if ".venv" not in p.parts and "outputs" not in p.parts}
    write_json(run / "source_hashes.json", source)
    for relative in source:
        snapshot = run / 'source_snapshot' / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, snapshot)
    for config in (ROOT / 'config').glob('*.json'):
        dest = run / 'source_snapshot' / 'config' / config.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config, dest)
    write_json(run / "status.json", {"status": "running"})
    try:
        with (run / "console.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                print(f"Run: {run}\nDevice: {device}", flush=True)
                selected = prepare_indices(data_dir, run, cfg)
                export_eda(data_dir, run, selected)
                write_json(run / "data_summary.json", {"selected_counts": {k: len(v) for k, v in selected.items()},
                    "dataset": "CIFAR10", "image_size": cfg["image_size"],
                    "full_counts": {"train": 45000, "val": 5000, "test": 10000},
                    "split_origin": "stratified official CIFAR10 training pool; independent official test",
                    "test_usage": "evaluation only after validation selection",
                    "class_counts": "balanced stratification; inspect indices for exact membership"})
                rows = []
                for name in cfg["models"]:
                    rows.extend(run_model(name, cfg, selected, data_dir, run, device))
                    pd.DataFrame(rows).to_csv(run / "comparison.csv", index=False)
                summarize_suite(run, cfg, selected)
                write_json(run / "status.json", {"status": "completed"})
                write_json(work / "outputs" / "selected_run.json", {"run": str(run)})
                print(f"Completed: {run}", flush=True)
    except BaseException:
        write_json(run / "status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise
    finally:
        # Includes all artifacts except the manifest itself; original weights stay untouched.
        manifest = {str(p.relative_to(run)): file_hash(p) for p in run.rglob("*") if p.is_file() and p.name != "manifest.json"}
        write_json(run / "manifest.json", manifest)
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "cpu_pilot.json"))
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    run_suite(args.config, args.work_dir, args.device)


if __name__ == "__main__":
    main()

"""End-to-end transfer-learning suite; test data is evaluated after selection."""
import argparse
import contextlib
import copy
from datetime import datetime, timezone
import itertools
import importlib.metadata as package_metadata
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

from ..datasets.data import image_loader, image_transform, prepare_indices
from ..datasets.eda import export_summary
from ..utils.delivery import write_json, file_hash, resolve_data_dir
from ..models.models import (SPECS, build_model, classifier, features, configure_trainable,
                             parameter_counts, training_mode, last_block, CachedTail)
from .tuning import fine_candidates, select_trial, make_optimizer, make_scheduler, stopping_policy
from ..utils.reporting import export_evaluation, summarize_suite

ROOT = Path(__file__).resolve().parents[2]


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
    prefix_cache, offset = None, 0
    def save_prefix(module, arguments):
        nonlocal prefix_cache, offset
        value = arguments[0].detach().cpu().numpy()
        if prefix_cache is None:
            prefix_cache = np.lib.format.open_memmap(dest.with_name(f'{split}_prefix.npy'),
                mode='w+', dtype=np.float32, shape=(len(indices), *value.shape[1:]))
        prefix_cache[offset:offset + len(value)] = value
        offset += len(value)
    handle = last_block(model, name).register_forward_pre_hook(save_prefix) if cfg.get('cache_frozen_prefix') else None
    try:
        with torch.inference_mode():
            for batch_no, (images, labels, original) in enumerate(loader):
                xs.append(features(model, name, images.to(device)).cpu())
                ys.append(labels)
                ids.append(original)
                if batch_no % 25 == 0:
                    print(f"{name} extract {split}: {min((batch_no + 1) * loader.batch_size, len(indices))}/{len(indices)}", flush=True)
    finally:
        if handle is not None:
            handle.remove()
        if prefix_cache is not None:
            prefix_cache.flush()
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
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss; trial stopped before optimizer update')
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
            if batch_number and batch_number % 250 == 0:
                print(f"  {'train' if training else 'val'}: {count}/{len(loader.dataset)}", flush=True)
    return {"loss": total_loss / count, "accuracy": correct / count}


def _save_checkpoint(state, path):
    """Commit on the same filesystem; an interrupted write preserves the last epoch."""
    temporary = path.with_suffix('.tmp')
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fit(model, train, val, optimizer, epochs, device, folder, metadata, mode_fn=None, scheduler=None,
        stopping=None, resume=False):
    if epochs < 1:
        raise ValueError('epochs must be positive')
    folder.mkdir(parents=True, exist_ok=True)
    history, best, best_loss = [], -1., float('inf')
    stopping = stopping or {}
    minimum = stopping.get('min_epochs', epochs)
    patience = stopping.get('patience', epochs+1)
    delta = stopping.get('min_delta', 0.)
    monitor_best, stale, stop_reason = float('inf'), 0, 'epoch_budget_exhausted'
    train_pass_seconds, validation_seconds = 0., 0.
    elapsed_before, first_epoch, best_checkpoint = 0., 1, None
    signature = {'metadata': metadata, 'epochs': epochs, 'stopping': stopping,
                 'optimizer': type(optimizer).__name__,
                 'scheduler': type(scheduler).__name__ if scheduler is not None else None,
                 'optimizer_groups': [{k: v for k, v in group.items() if k != 'params'}
                                      for group in optimizer.param_groups],
                 'loaders': [{'samples': len(loader.dataset), 'batch_size': loader.batch_size,
                              'workers': loader.num_workers, 'sampler': type(loader.sampler).__name__}
                             for loader in (train, val)],
                 'device': str(device), 'torch': str(torch.__version__),
                 'python': platform.python_version(), 'numpy': np.__version__,
                 'torchvision': package_metadata.version('torchvision'),
                 'threads': torch.get_num_threads(),
                 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.machine()}
    if resume:
        saved = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
        if saved.get('schema_version') != 1:
            raise ValueError('Unsupported resume checkpoint schema')
        if saved['signature'] != signature:
            raise ValueError('Resume configuration/environment differs from saved trial')
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(saved['scheduler_state_dict'])
        history, best_checkpoint = saved['history'], saved['best_checkpoint']
        first_epoch = saved['epoch'] + 1
        best, best_loss = best_checkpoint['val_accuracy'], best_checkpoint['val_loss']
        monitor_best, stale = saved['monitor_best'], saved['stale']
        stop_reason = saved['stop_reason']
        train_pass_seconds, validation_seconds = saved['train_pass_seconds'], saved['validation_seconds']
        elapsed_before = saved['elapsed_seconds']
        random.setstate(saved['python_rng'])
        numpy_rng = saved['numpy_rng']
        np.random.set_state((numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32),
                             numpy_rng[2], numpy_rng[3], numpy_rng[4]))
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        for loader, state in zip((train, val), saved['loader_rng']):
            if (loader.generator is None) != (state is None):
                raise ValueError('Resume DataLoader generator differs')
            if state is not None:
                loader.generator.set_state(state)
        _save_checkpoint(best_checkpoint, folder / 'best.pt')
    elif (folder / 'last.pt').exists() or (folder / 'history.csv').exists():
        raise ValueError('Trial already contains history; use explicit resume or a new run')
    write_json(folder / "config.json", metadata)
    writer = SummaryWriter(str(folder / "tensorboard"), purge_step=first_epoch if resume else None)
    writer.add_text("configuration", json.dumps(metadata, indent=2))
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    start = time.perf_counter()
    try:
        for epoch in range(first_epoch, epochs + 1):
            if stop_reason == 'validation_loss_patience':
                break
            epoch_start = time.perf_counter()
            rates = {f'lr_group_{i}': group['lr'] for i, group in enumerate(optimizer.param_groups)}
            sync(device)
            train_start = time.perf_counter()
            tr = pass_epoch(model, train, device, optimizer, mode_fn)
            sync(device)
            train_elapsed = time.perf_counter()-train_start
            val_start = time.perf_counter()
            va = pass_epoch(model, val, device)
            sync(device)
            val_elapsed = time.perf_counter()-val_start
            train_pass_seconds += train_elapsed
            validation_seconds += val_elapsed
            row = {"epoch": epoch, "train_loss": tr["loss"], "train_accuracy": tr["accuracy"],
                   "val_loss": va["loss"], "val_accuracy": va["accuracy"],
                   "epoch_seconds": time.perf_counter() - epoch_start,
                   "train_pass_seconds": train_elapsed, "validation_seconds": val_elapsed,
                   "train_images_per_second": len(train.dataset)/max(train_elapsed, 1e-12), **rates}
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
            # On equal validation accuracy prefer lower validation loss, then earlier epoch.
            if va['accuracy'] > best or (va['accuracy'] == best and va['loss'] < best_loss):
                best = va["accuracy"]
                best_loss = va['loss']
                best_checkpoint = {"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "metadata": metadata,
                                   "best_epoch": epoch, "val_accuracy": best, 'val_loss':best_loss}
                _save_checkpoint(best_checkpoint, folder / "best.pt")
            if scheduler is not None:
                scheduler.step()
            print(f"{folder.name} epoch {epoch}/{epochs}: train={tr['accuracy']:.4f}, val={va['accuracy']:.4f}", flush=True)
            if va['loss'] < monitor_best-delta:
                monitor_best, stale = va['loss'], 0
            else:
                stale += 1
            if epoch >= minimum and stale >= patience:
                stop_reason = 'validation_loss_patience'
            numpy_rng = np.random.get_state()
            _save_checkpoint({'schema_version': 1, 'signature': signature, 'epoch': epoch,
                'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'history': history, 'best_checkpoint': best_checkpoint,
                'monitor_best': monitor_best, 'stale': stale, 'stop_reason': stop_reason,
                'train_pass_seconds': train_pass_seconds, 'validation_seconds': validation_seconds,
                'elapsed_seconds': elapsed_before + time.perf_counter() - start,
                'python_rng': random.getstate(),
                'numpy_rng': (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                'loader_rng': [loader.generator.get_state() if loader.generator is not None else None
                               for loader in (train, val)]}, folder / 'last.pt')
            if stop_reason == 'validation_loss_patience':
                break
        sync(device)
        result = {"best_val_accuracy": best, 'best_val_loss':best_loss,
                  'epochs_planned': epochs, 'epochs_completed': len(history),
                  'stop_reason': stop_reason, 'early_stopping': stopping,
                  'minimum_observed_val_loss': min(r['val_loss'] for r in history),
                  'last_val_loss': history[-1]['val_loss'],
                  'convergence_claim': 'not established; inspect validation trend and stopping evidence',
                  "train_seconds": elapsed_before + time.perf_counter() - start,
                  'resumed': resume, 'resume_boundary': 'completed epoch; incomplete epoch replayed',
                  'timing_policy': 'active fit segments; downtime excluded; resumed timing not a fresh benchmark',
                  'train_pass_seconds':train_pass_seconds, 'validation_seconds':validation_seconds,
                  'train_images_per_second':len(train.dataset)*len(history)/max(train_pass_seconds,1e-12),
                  'peak_cuda_memory_mib':torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
                  'peak_memory_scope': 'current fit process segment only' if resume else 'whole fit',
                  "best_epoch": torch.load(folder / "best.pt", weights_only=True, map_location="cpu")["best_epoch"]}
        writer.add_hparams({k: metadata[k] for k in ('model','stage','batch_size','lr',
                          'optimizer','backbone_lr','unfreeze','scheduler') if k in metadata},
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


def search_finetune(model, name, cfg, selected, data_dir, run, device,
                    train_features, val_features, batch_size):
    """Start every candidate from the SAME selected head; evaluate validation only."""
    folder = run / name
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    candidates = fine_candidates(cfg)
    trials = []
    prefix_data = {}
    if cfg.get('cache_frozen_prefix'):
        for split, data in [('train', train_features), ('val', val_features)]:
            mapped = np.load(folder / f'{split}_prefix.npy', mmap_mode='c')
            prefix_data[split] = (torch.from_numpy(mapped), data[1], data[2])
    for number, candidate in enumerate(candidates):
        seed_all(cfg['seed'])
        model.load_state_dict(initial)
        model.zero_grad(set_to_none=True)
        configure_trainable(model, name, 'finetune', candidate['unfreeze'],
                            max_fraction=cfg.get('max_trainable_fraction'))
        head_ids = {id(p) for p in classifier(model, name).parameters()}
        backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
        groups = [dict(params=backbone, lr=candidate['backbone_lr']),
                  dict(params=classifier(model, name).parameters(), lr=candidate['head_lr'])]
        optimizer = make_optimizer(groups, candidate['optimizer'], cfg['weight_decay'], cfg.get('momentum', .9))
        scheduler = make_scheduler(optimizer, candidate['scheduler'], cfg['finetune_epochs'],
                                   cfg.get('scheduler_min_ratio', .1))
        meta = {'model':name, 'stage':'finetune', **candidate, 'lr':candidate['head_lr'],
                'scheduler_min_ratio':cfg.get('scheduler_min_ratio', .1),
                'batch_size':batch_size, 'seed':cfg['seed'], 'weights':str(SPECS[name][1]),
                'dataset':'CIFAR10', 'image_size':cfg['image_size'],
                'initialization':'same selected head checkpoint for every fine-tune trial',
                'batchnorm':'running statistics frozen', 'parameter_counts':parameter_counts(model),
                'dropout':cfg.get('dropout', 0.),
                'train_augmentation':cfg.get('train_augmentation', False),
                'data_sha256':file_hash(run/'selected_indices.npz'),
                'execution':'cached_frozen_prefix' if prefix_data else 'full_image_forward'}
        if prefix_data:
            fit_model = CachedTail(model, name).to(device)
            model.eval()
            sample = next(iter(image_loader(data_dir, 'train', selected['train'][:8], name, 8,
                cfg['seed'], image_size=cfg['image_size'])))[0].to(device)
            with torch.inference_mode():
                direct = model(sample)
                equivalent = fit_model(prefix_data['train'][0][:len(sample)].to(device))
            torch.testing.assert_close(direct, equivalent, atol=1e-4, rtol=1e-5)
            write_json(folder/'prefix_equivalence.json', {'passed':True, 'samples':len(sample),
                'max_logit_error':(direct-equivalent).abs().max().item(),
                'policy':'frozen deterministic prefix; last block only'})
            train_loader = tensor_loader(prefix_data['train'], batch_size, cfg['seed'], True)
            val_loader = tensor_loader(prefix_data['val'], batch_size, cfg['seed'])
        else:
            fit_model = model
            train_loader = image_loader(data_dir, 'train', selected['train'], name, batch_size,
                cfg['seed'], True, cfg['image_size'], augmentation=cfg.get('train_augmentation', False))
            val_loader = image_loader(data_dir, 'val', selected['val'], name, batch_size,
                cfg['seed'], image_size=cfg['image_size'])
        trial_folder = folder / (f'finetune_trial_{number:02d}' if 'finetune_candidates' in cfg else 'finetune')
        result = fit(fit_model, train_loader, val_loader, optimizer, cfg['finetune_epochs'], device,
                     trial_folder, meta, training_mode, scheduler=scheduler, stopping=stopping_policy(cfg))
        trials.append({'trial':number, **candidate, **result, 'trainable':meta['parameter_counts']['trainable'],
                       'execution':meta['execution']})
        pd.DataFrame(trials).to_csv(folder/'finetune_search_results.csv', index=False)
        del optimizer, scheduler, fit_model, train_loader, val_loader
    winner = select_trial(trials, cfg.get('selection_policy'))
    chosen = folder / (f"finetune_trial_{winner['trial']:02d}" if 'finetune_candidates' in cfg else 'finetune')
    destination = folder/'finetune'
    if chosen != destination:
        destination.mkdir()
        for filename in ['best.pt','history.csv','config.json','training_summary.json']:
            shutil.copy2(chosen/filename, destination/filename)
    checkpoint = torch.load(destination/'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    selection = json.loads((folder/'selection.json').read_text(encoding='utf-8'))
    selection.update(finetune=winner, finetune_policy=cfg.get('selection_policy', {}),
                     criterion='validation accuracy band -> validation loss band -> fit time; no test input')
    write_json(folder/'selection.json', selection)
    return winner, checkpoint['metadata'], sum(t['train_seconds'] for t in trials)


def run_model(name, cfg, selected, data_dir, run, device):
    fine_candidates(cfg)  # Reject incompatible cache/depth before loading weights or extracting features.
    sync(device)
    model_start = time.perf_counter()
    seed_all(cfg["seed"])
    folder = run / name
    folder.mkdir(exist_ok=True)
    model = build_model(name, dropout=cfg.get('dropout', 0.)).to(device)
    configure_trainable(model, name, "head")
    (folder / "architecture.txt").write_text(str(model), encoding="utf-8")
    params = [{"name": n, "shape": str(tuple(p.shape)), "parameters": p.numel(),
               "head_stage_trainable": p.requires_grad} for n, p in model.named_parameters()]
    pd.DataFrame(params).to_csv(folder / "parameters.csv", index=False)
    preprocess = SPECS[name][1].transforms()
    write_json(folder / "preprocessing.json", {"weights": str(SPECS[name][1]),
               "dataset": "CIFAR10", "channels": "RGB", "image_size": cfg["image_size"],
               "transform": str(image_transform(name, cfg['image_size'])),
               "augmentation": ("fine-tune train only: RandomCrop(32,padding=4,reflect), HorizontalFlip(0.5); head cache/val/test deterministic"
                                if cfg.get('train_augmentation') else "none; deterministic cached features"),
               "dropout": cfg.get('dropout', 0.)})
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
                "dropout": cfg.get('dropout', 0.),
                "seed": cfg["seed"], "weights": str(SPECS[name][1]),
                "dataset": "CIFAR10", "image_size": cfg["image_size"],
                'optimizer':cfg.get('head_optimizer', 'adamw'),
                "data_sha256": file_hash(run / "selected_indices.npz")}
        trial = folder / f"head_trial_{number:02d}"
        optimizer = make_optimizer([dict(params=head.parameters(), lr=lr)],
            cfg.get('head_optimizer', 'adamw'), cfg['weight_decay'], cfg.get('momentum', .9))
        scheduler = make_scheduler(optimizer, cfg.get('head_scheduler', 'none'),
                                   cfg['head_epochs'], cfg.get('scheduler_min_ratio', .1))
        meta['scheduler'] = cfg.get('head_scheduler', 'none')
        meta['scheduler_min_ratio'] = cfg.get('scheduler_min_ratio', .1)
        result = fit(head, tensor_loader(train_features, bs, cfg["seed"], True),
                     tensor_loader(val_features, bs, cfg["seed"]),
                     optimizer, cfg["head_epochs"], device, trial, meta, scheduler=scheduler)
        trials.append({"trial": number, "lr": lr, "batch_size": bs, **result})
        pd.DataFrame(trials).to_csv(folder / "search_results.csv", index=False)
    winner = select_trial(trials, cfg.get('selection_policy'))
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
    write_json(folder / "selection.json", {"criterion": "validation accuracy/loss bands then fit time",
               "head": winner, "finetune_policy": "configured unfreeze depth + classifier; best validation accuracy epoch"})
    fine_result, meta, all_fine_seconds = search_finetune(
        model, name, cfg, selected, data_dir, run, device,
        train_features, val_features, winner['batch_size'])
    fine_folder = folder / 'finetune'
    bs = winner['batch_size']
    sync(device)
    model_selection_wall_seconds = time.perf_counter()-model_start
    # Evaluate two predeclared conditions after their checkpoints are locked.
    rows = []
    for stage, stage_folder, result, metadata in [
        ("head", head_folder, winner, head_meta),
        ("finetune", fine_folder, fine_result, meta)]:
        saved = torch.load(stage_folder / "best.pt", map_location=device, weights_only=True)
        restored = build_model(name, pretrained=False, dropout=saved['metadata'].get('dropout', 0.)).to(device)
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
                     "best_val_accuracy": result["best_val_accuracy"], 'best_val_loss':result['best_val_loss'],
                     'optimizer':metadata.get('optimizer', 'adamw'),
                     'unfreeze':metadata.get('unfreeze', 'head'), 'scheduler':metadata.get('scheduler', 'none'),
                     'execution':'pooled_feature_cache' if stage=='head' else metadata['execution'],
                     'all_head_search_seconds':sum(t['train_seconds'] for t in trials),
                     'all_finetune_search_seconds':all_fine_seconds if stage=='finetune' else 0.,
                     'selected_head_fit_seconds':winner['train_seconds'],
                     'model_selection_wall_seconds':model_selection_wall_seconds,
                     'train_pass_seconds':result.get('train_pass_seconds'),
                     'validation_seconds':result.get('validation_seconds'),
                     'train_images_per_second':result.get('train_images_per_second'),
                     'peak_cuda_memory_mib':result.get('peak_cuda_memory_mib'),
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


def run_suite(config_path, work_dir=None, device_override="auto", seed_override=None):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if seed_override is not None:
        cfg['seed'] = seed_override
    if cfg.get("dataset") != "CIFAR10":
        raise ValueError("This pipeline requires a CIFAR10 config; legacy FashionMNIST runs are read-only.")
    fine_candidates(cfg)
    if cfg.get('head_optimizer', 'adamw') not in ('adamw', 'sgd'):
        raise ValueError('head_optimizer must be adamw or sgd')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device_override=='auto' else device_override)
    if cfg.get('require_cuda') and device.type != 'cuda':
        raise ValueError('This tuning config requires CUDA; use --device cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available in this interpreter')
    torch.set_num_threads(cfg["threads"])
    seed_all(cfg["seed"])
    work = Path(work_dir or ROOT).resolve()
    torch.hub.set_dir(str(work / "data" / "pretrained"))
    data_dir = resolve_data_dir(work)
    run = work / "outputs" / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + cfg["profile"] + "_" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    # Preserve the executed presentation before a later run changes its cell outputs.
    previous = work / 'outputs' / 'selected_run.json'
    if previous.exists():
        previous_run = Path(json.loads(previous.read_text(encoding='utf-8'))['run'])
        if not previous_run.is_absolute():
            previous_run = work / previous_run
        if not previous_run.is_dir():
            raise ValueError(f'Previous selected run is missing: {previous_run}')
        for notebook in (ROOT / 'notebooks').glob('*.ipynb'):
            content = notebook.read_bytes()
            snapshot = previous_run / 'notebook_snapshots' / file_hash(notebook) / notebook.name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if not snapshot.exists():
                snapshot.write_bytes(content)
    write_json(run / "config.json", cfg)
    write_json(run / "environment.json", {"python": sys.version, "torch": torch.__version__,
               "torchvision": __import__("torchvision").__version__, "device": str(device),
               "platform": platform.platform(), "cpu_count": os.cpu_count(),
               'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
               'cuda':torch.version.cuda, 'cudnn':torch.backends.cudnn.version(),
               'timing_policy':'synchronized train/validation passes; fit wall includes logging/checkpoint I/O' })
    environment = json.loads((run/'environment.json').read_text(encoding='utf-8'))
    environment['packages'] = {}
    for package in ['torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn', 'tensorboard', 'pillow']:
        installed = package_metadata.metadata(package)
        environment['packages'][package] = {'version': package_metadata.version(package),
                                            'requires_python': installed.get('Requires-Python')}
    environment['reproducibility'] = {
        'seed': cfg['seed'], 'split_seed': cfg.get('split_seed', cfg['seed']),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cross_device_equivalence': 'not established; verify on target GPU'}
    write_json(run/'environment_snapshot.json', environment)
    write_json(work/'outputs/logs/environment_snapshot.json', {**environment, 'run': str(run)})
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
                selected = prepare_indices(data_dir, run, cfg, split_dir=work/'outputs/eda')
                export_summary(data_dir, run, selected)
                write_json(run / "data_summary.json", {"selected_counts": {k: len(v) for k, v in selected.items()},
                    'split_seed':cfg.get('split_seed', cfg['seed']), 'training_seed':cfg['seed'],
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
    parser.add_argument('--seed', type=int, help='Override initialization/shuffle seed; tuning config keeps split_seed fixed')
    parser.add_argument('--dry-run', action='store_true', help='Validate candidates and show budget without loading data/weights')
    args = parser.parse_args()
    if args.dry_run:
        cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
        candidates = fine_candidates(cfg)
        print(json.dumps({'profile':cfg['profile'], 'models':cfg['models'],
            'head_batch_sizes':cfg['batch_sizes'], 'head_epochs_per_trial':cfg['head_epochs'],
            'head_trials_per_model':len(cfg['learning_rates'])*len(cfg['batch_sizes']),
            'head_scheduler':cfg.get('head_scheduler', 'none'),
            'scheduler_min_ratio':cfg.get('scheduler_min_ratio', .1),
            'fine_trials_per_model':len(candidates), 'fine_epochs_per_trial':cfg['finetune_epochs'],
            'early_stopping':stopping_policy(cfg),
            'dropout':cfg.get('dropout', 0.), 'fine_train_augmentation':cfg.get('train_augmentation',False),
            'split_seed':cfg.get('split_seed',cfg['seed']), 'training_seed':args.seed if args.seed is not None else cfg['seed'],
            'cache_frozen_prefix':cfg.get('cache_frozen_prefix',False), 'candidates':candidates}, indent=2))
        return
    run_suite(args.config, args.work_dir, args.device, args.seed)


if __name__ == "__main__":
    main()

"""Bounded calibration on real CIFAR-10; never update the selected lab run.

Run with the repository .venv interpreter. This is an internal measurement,
not a candidate training config or a final accuracy benchmark.
"""
import argparse
import copy
import gc
from pathlib import Path
import platform
import signal
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
LAB = PROJECT/'labs/lab02'
sys.path.insert(0, str(LAB))

import numpy as np
import torch
from torchvision.datasets import CIFAR10

from src.datasets.classification import balanced_subset, image_loader
from src.utils.artifacts import write_json
from src.datasets.eda import locked_split
from src.training.trainer import cached_features, fit, pass_epoch, seed_all, tensor_loader
from src.models.backbones import build_model, classifier, configure_trainable, training_mode, CachedTail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-seconds', type=int, default=600)
    parser.add_argument('--batch-size', type=int, choices=(64, 128), default=64)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 1800:
        raise ValueError('Calibration must be bounded to at most 30 minutes')
    batch_size = args.batch_size
    destination = Path(__file__).parent/('measurements' if batch_size == 64 else 'measurements_batch128')
    if destination.exists():
        raise FileExistsError('Measurements already exist; preserve them instead of overwriting')
    destination.mkdir()
    started = time.perf_counter()
    report = {'purpose': 'CPU timing calibration, not final training or test evaluation',
              'python': platform.python_version(), 'torch': str(torch.__version__),
              'device': 'cpu', 'threads': 4, 'train_samples': 640, 'val_samples': 160,
              'image_size': 64, 'batch_size': batch_size, 'seed': 42, 'models': {}, 'status': 'running'}

    def timeout(signum, frame):
        raise TimeoutError('Calibration time budget reached')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds)
    torch.set_num_threads(report['threads'])
    device = torch.device('cpu')
    torch.hub.set_dir(str(LAB/'data/pretrained'))
    cfg = dict(seed=42, feature_batch_size=64, image_size=64, cache_frozen_prefix=True)
    data_dir = LAB/'data'
    pool = CIFAR10(str(data_dir), train=True, download=False)
    splits = locked_split(LAB/'outputs/eda', np.asarray(pool.targets, dtype=np.int64), 42)
    selected = {key: balanced_subset(splits[key], pool.targets, report[f'{key}_samples'], 42)
                for key in ('train', 'val')}
    del pool
    np.savez(destination/'selected_indices.npz', **selected)
    try:
        for name in ('resnet18', 'densenet121'):
            print(f'{name}: loading verified pretrained weights', flush=True)
            seed_all(42)
            folder = destination/name
            folder.mkdir()
            model = build_model(name).to(device)
            configure_trainable(model, name, 'head')
            # Warm up kernels without changing model/BN state; not counted in extraction.
            model.eval()
            with torch.inference_mode():
                model(torch.zeros(2, 3, 64, 64))
            train, tr_seconds = cached_features(model, name, data_dir, 'train', selected['train'],
                                                cfg, device, folder/'train_features.pt')
            val, va_seconds = cached_features(model, name, data_dir, 'val', selected['val'],
                                              cfg, device, folder/'val_features.pt')
            initial = copy.deepcopy(model.state_dict())
            head = classifier(model, name)
            # Sanity on real cached pretrained features, independent of timing epochs.
            x, y = train[0][:8], train[1][:8]
            opt = torch.optim.AdamW(head.parameters(), lr=.01)
            before = torch.nn.functional.cross_entropy(head(x), y).item()
            for step in range(100):
                opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(head(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite sanity loss')
                loss.backward()
                opt.step()
                with torch.no_grad():
                    final = torch.nn.functional.cross_entropy(head(x), y).item()
                    accuracy = (head(x).argmax(1) == y).float().mean().item()
                if final < .05 and accuracy == 1:
                    break
            sanity = dict(samples=8, before_loss=before, final_loss=final,
                          accuracy=accuracy, steps=step+1, passed=final < .05 and accuracy == 1,
                          scope='head-only on real cached pretrained features')
            if not sanity['passed']:
                raise RuntimeError(f'{name}: head-only sanity failed: {sanity}')
            model.load_state_dict(initial)
            head_optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=1e-4)
            head_result = fit(head, tensor_loader(train, batch_size, 42, True), tensor_loader(val, batch_size, 42),
                              head_optimizer, 1, device, folder/'head_timing',
                              {'model': name, 'stage': 'head', 'batch_size': batch_size, 'lr': .001})
            # Start both fine execution policies from identical weights.
            fine_initial = copy.deepcopy(model.state_dict())
            configure_trainable(model, name, 'finetune')
            policies = {}
            for policy in ('cached_frozen_prefix', 'full_image_forward'):
                model.load_state_dict(fine_initial)
                model.zero_grad(set_to_none=True)
                if policy == 'cached_frozen_prefix':
                    datasets = {}
                    for split, data in (('train', train), ('val', val)):
                        prefix = torch.from_numpy(np.load(folder/f'{split}_prefix.npy', mmap_mode='c'))
                        datasets[split] = tensor_loader((prefix, data[1], data[2]), batch_size, 42, split == 'train')
                    target = CachedTail(model, name)
                    # Check cached execution is equivalent before any optimizer update.
                    model.eval()
                    sample = next(iter(image_loader(data_dir, 'train', selected['train'][:2], name,
                                                   2, 42, image_size=64)))[0]
                    with torch.inference_mode():
                        direct = model(sample)
                        cached = target(datasets['train'].dataset.tensors[0][:2])
                    torch.testing.assert_close(direct, cached, atol=1e-4, rtol=1e-5)
                else:
                    target = model
                    datasets = {split: image_loader(data_dir, split, selected[split], name, batch_size, 42,
                                                   split == 'train', 64) for split in ('train', 'val')}
                head_ids = {id(p) for p in head.parameters()}
                groups = [dict(params=[p for p in model.parameters() if p.requires_grad and
                                       id(p) not in head_ids], lr=1e-5),
                          dict(params=head.parameters(), lr=1e-4)]
                optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
                policies[policy] = fit(target, datasets['train'], datasets['val'], optimizer, 1,
                                      device, folder/policy,
                                      {'model': name, 'stage': 'finetune', 'batch_size': batch_size,
                                       'lr': 1e-4, 'execution': policy}, training_mode)
                if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                           for p in model.parameters() if p.requires_grad and id(p) not in
                           {id(q) for q in head.parameters()}):
                    raise RuntimeError('No finite nonzero gradient in the unfrozen backbone')
                del datasets, optimizer, target
                gc.collect()
            # Measure independent validation inference throughput; do not touch test.
            loader = image_loader(data_dir, 'val', selected['val'], name, batch_size, 42, image_size=64)
            tick = time.perf_counter()
            pass_epoch(model, loader, device)
            inference_seconds = time.perf_counter()-tick
            report['models'][name] = dict(head_sanity=sanity, extraction_train_seconds=tr_seconds,
                extraction_val_seconds=va_seconds, head_epoch=head_result, fine_epochs=policies,
                inference_val_seconds=inference_seconds,
                extrapolation_note='Tiny CPU sample; thermal throttling, I/O and cache size may change speed')
            write_json(destination/'timings.json', report)
            print(f'{name}: calibration complete', flush=True)
            del model, head, initial, fine_initial, train, val, loader, opt, head_optimizer
            gc.collect()
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.perf_counter()-started
        write_json(destination/'timings.json', report)


if __name__ == '__main__':
    main()

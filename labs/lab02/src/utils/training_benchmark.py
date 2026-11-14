"""Bounded full-image training steps on disposable models, never saved back to weights.

Fresh optimizer at recipe initial learning rates; not resumed training or convergence.
Reuse the existing RSS sampler; memory and timing passes are independent.
"""
import gc
import math
import time
from pathlib import Path

from .inference_benchmark import _RSSSampler, timing_summary


def validate_settings(settings):
    """Reject invalid counts before loading checkpoints or creating output."""
    for key, minimum in [('batch_size', 1), ('warmup', 0), ('repeats', 1), ('threads', 1)]:
        value = settings[key]
        if type(value) is not int or value < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    if settings['batch_size'] != 1:
        raise ValueError('This protocol requires batch_size=1')
    interval = settings['sample_interval']
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval <= 0:
        raise ValueError('sample_interval must be finite and positive')
    if settings['device'] not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')


def prepare_training(checkpoint, config, name, device):
    """Restore freeze/BN/dropout from checkpoint metadata without pretrained downloads."""
    import torch
    from ..models.models import build_model, classifier, configure_trainable, training_mode
    from ..training.tuning import make_optimizer
    saved = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
    meta = saved['metadata']
    if meta.get('dataset') != 'CIFAR10' or meta.get('stage') != 'finetune' or meta.get('model') != name:
        raise ValueError('Expected matching CIFAR10 finetune model metadata')
    model = build_model(name, pretrained=False, dropout=meta.get('dropout', 0.))
    model.load_state_dict(saved['state_dict'])
    model = model.float().to(device)
    configure_trainable(model, name, 'finetune', meta['unfreeze'], config.get('max_trainable_fraction'))
    head = list(classifier(model, name).parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    optimizer = make_optimizer([dict(params=backbone, lr=meta['backbone_lr']),
                                dict(params=head, lr=meta['head_lr'])],
                               meta['optimizer'], config['weight_decay'], config.get('momentum', .9))
    training_mode(model)
    return model, optimizer, meta


def benchmark_steps(model, optimizer, images, targets, settings):
    """Measure one first memory step, warmups, then repeated timed training steps."""
    import torch
    from ..models.models import training_mode
    validate_settings(settings)
    if images.shape[0] != 1 or targets.shape != (1,) or images.device != targets.device:
        raise ValueError('Expected one preloaded image and one target on the same device')
    device = images.device
    if device.type != settings['device']:
        raise ValueError('Tensor device differs from benchmark settings')
    training_mode(model)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(images), targets)
        loss.backward()
        optimizer.step()
        return loss

    gc.collect()
    sync()
    import psutil
    baseline = psutil.Process().memory_info().rss
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with _RSSSampler(settings['sample_interval']) as sampler:
        loss = step()
        sync()
    allocated = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    reserved = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None
    for _ in range(settings['warmup']):
        loss = step()
    sync()
    times = []
    for _ in range(settings['repeats']):
        sync()
        start = time.perf_counter_ns()
        loss = step()
        sync()
        times.append((time.perf_counter_ns() - start) / 1e6)
    if not torch.isfinite(loss).item():
        raise RuntimeError('Training benchmark produced a nonfinite loss')
    return dict(timing_ms=timing_summary(times), samples_ms=times,
                memory=dict(baseline_rss_bytes=baseline, peak_rss_bytes=max(baseline, sampler.peak),
                            rss_increase_bytes=max(0, sampler.peak-baseline),
                            peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                            rss_samples=sampler.samples), last_loss=float(loss.detach()))


def measure_checkpoint(root, run, name, settings, data_dir):
    """Worker entry: use one fixed train image; hash source weights before and after."""
    import json
    import random
    import platform
    import sys
    import numpy as np
    import torch
    import torchvision
    from ..datasets.data import IndexedImages
    from ..models.models import parameter_counts
    from .delivery import file_hash
    validate_settings(settings)
    device = torch.device(settings['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing to substitute CPU')
    seed = settings['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(settings['threads'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    run = Path(run)
    config = json.loads((run/'config.json').read_text())
    checkpoint = run/name/'finetune/best.pt'
    before = file_hash(checkpoint)
    split_path = run/'selected_indices.npz'
    with np.load(split_path) as split:
        indices = split['train']
    position = settings['train_position']
    if type(position) is not int or not 0 <= position < len(indices):
        raise ValueError('train_position outside selected training split')
    model, optimizer, meta = prepare_training(checkpoint, config, name, device)
    if meta['data_sha256'] != file_hash(split_path):
        raise ValueError('Checkpoint selected training split checksum mismatch')
    dataset = IndexedImages(data_dir, 'train', indices[position:position+1], name,
                            meta['image_size'], augmentation=False)
    image, label, index = dataset[0]
    images = image.unsqueeze(0).to(device)
    targets = torch.tensor([label], dtype=torch.long, device=device)
    result = benchmark_steps(model, optimizer, images, targets, settings)
    if file_hash(checkpoint) != before:
        raise RuntimeError('Source checkpoint changed during benchmark')
    return dict(model=name, training_run=run.name, checkpoint_sha256=before,
                selected_indices_sha256=file_hash(split_path), train_index=index, label=int(label),
                input_tensor_sha256=__import__('hashlib').sha256(image.numpy().tobytes()).hexdigest(),
                metadata=meta, parameters=parameter_counts(model), batch_size=1, precision='float32',
                environment=dict(python=sys.version, platform=platform.platform(), torch=str(torch.__version__),
                    torchvision=str(torchvision.__version__), device=str(device), threads=torch.get_num_threads(),
                    device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
                    cuda=torch.version.cuda, deterministic_algorithms=True, tf32=False), **result)

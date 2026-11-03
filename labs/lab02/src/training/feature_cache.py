"""Extract deterministic feature/prefix caches and reload aligned tensor batches."""
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..datasets.data import image_loader
from ..models.models import features, last_block


def cached_features(model: torch.nn.Module, name: str, data_dir: Path, split: str,
                    indices: np.ndarray, cfg: dict, device: torch.device, dest: Path
                    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], float]:
    """Save CPU features (N, F), labels (N,) and original indices (N,) in loader order.

    Switch the model to eval mode and optionally persist last-block inputs as NPY.
    Always remove the capture hook and flush any mapped prefix, including on error.
    """
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
    if device.type == "cuda":
        torch.cuda.synchronize()
    result = (torch.cat(xs), torch.cat(ys), torch.cat(ids))
    torch.save(dict(zip(("features", "labels", "indices"), result)), dest)
    return result, time.perf_counter() - start


def tensor_loader(data: tuple[torch.Tensor, ...], batch: int, seed: int,
                  shuffle: bool = False) -> DataLoader:
    """Batch aligned feature/label/index tensors with a private seeded generator."""
    return DataLoader(TensorDataset(*data), batch_size=batch, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed))

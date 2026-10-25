"""Locked stratified split, balanced pilot subsets, and weight-specific input."""
import hashlib
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from .models import SPECS


def image_transform(name, image_size=224):
    """Preserve RGB. CPU profile changes spatial size, not ImageNet normalization."""
    weights_transform = SPECS[name][1].transforms()
    if image_size == 224:
        return weights_transform
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(weights_transform.mean, weights_transform.std),
    ])


def balanced_subset(indices, labels, limit, seed):
    indices = np.asarray(indices, dtype=np.int64)
    if not limit or limit >= len(indices):
        return indices
    chosen, _ = train_test_split(indices, train_size=limit, random_state=seed,
                                 stratify=np.asarray(labels)[indices])
    return np.sort(chosen)


def prepare_indices(data_dir, run_dir, cfg):
    pool = datasets.CIFAR10(str(data_dir), train=True, download=True)
    test = datasets.CIFAR10(str(data_dir), train=False, download=True)
    train_idx, val_idx = train_test_split(
        np.arange(len(pool)), test_size=5000, random_state=cfg["seed"],
        stratify=np.asarray(pool.targets))
    full = {"train": np.sort(train_idx), "val": np.sort(val_idx),
            "test": np.arange(len(test))}
    selected = {s: balanced_subset(v, pool.targets if s != "test" else test.targets,
                                  cfg[f"{s}_limit"], cfg["seed"])
                for s, v in full.items()}
    np.savez(Path(run_dir) / "split_indices.npz", **full)
    np.savez(Path(run_dir) / "selected_indices.npz", **selected)
    return selected


class IndexedImages(Dataset):
    def __init__(self, data_dir, split, indices, name, image_size=224):
        self.raw = datasets.CIFAR10(str(data_dir), train=split != "test")
        self.indices = np.asarray(indices)
        self.transform = image_transform(name, image_size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        original = int(self.indices[i])
        image, label = self.raw[original]
        return self.transform(image), label, original


def image_loader(data_dir, split, indices, name, batch, seed, shuffle=False, image_size=224):
    import torch
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(IndexedImages(data_dir, split, indices, name, image_size), batch_size=batch,
                      shuffle=shuffle, num_workers=0, generator=generator,
                      pin_memory=torch.cuda.is_available())


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_eda(data_dir, run_dir, selected):
    """Inspect train data only for image statistics; test contributes class counts only."""
    import json
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pool = datasets.CIFAR10(str(data_dir), train=True)
    test = datasets.CIFAR10(str(data_dir), train=False)
    destination = Path(run_dir) / 'eda'
    destination.mkdir()
    counts = {}
    for split in ('train', 'val', 'test'):
        labels = np.asarray((test if split == 'test' else pool).targets)[selected[split]]
        counts[split] = np.bincount(labels, minlength=10)
    pd.DataFrame(counts, index=pool.classes).to_csv(destination / 'class_counts.csv')
    sums, squares, pixels = np.zeros(3), np.zeros(3), 0
    for start in range(0, len(selected['train']), 512):
        values = pool.data[selected['train'][start:start+512]].astype(np.float64) / 255
        sums += values.sum(axis=(0, 1, 2))
        squares += (values**2).sum(axis=(0, 1, 2))
        pixels += values.shape[0] * 32 * 32
    means = sums / pixels
    stats = {'dataset': 'CIFAR10', 'original_shape': [32, 32, 3],
             'classes': pool.classes, 'train_mean_rgb': means.tolist(),
             'train_std_rgb': np.sqrt(squares/pixels-means**2).tolist(),
             'normalization_policy': 'Descriptive train statistics only; pretrained ImageNet mean/std used for model input'}
    (destination / 'summary.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(10, 5, figsize=(9, 15))
    labels = np.asarray(pool.targets)
    for label, row in enumerate(axes):
        chosen = selected['train'][labels[selected['train']] == label][:5]
        for ax, index in zip(row, chosen):
            ax.imshow(pool.data[index])
            ax.set(xticks=[], yticks=[])
        row[0].set_ylabel(pool.classes[label])
    fig.suptitle('CIFAR-10: five original RGB training images per class')
    fig.tight_layout()
    fig.savefig(destination / 'training_examples.png', dpi=120)
    plt.close(fig)

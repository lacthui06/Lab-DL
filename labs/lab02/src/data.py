"""Datasets/loaders for experiment and infer; split math is owned by src.eda.

Caller map: experiment.prepare_indices/image_loader; infer.image_transform;
tests/test_contracts.py. Artifact hashing and path resolution live in src.delivery.
"""
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from .models import SPECS
from .eda import locked_split


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


def prepare_indices(data_dir, run_dir, cfg, split_dir=None):
    pool = datasets.CIFAR10(str(data_dir), train=True, download=True)
    test = datasets.CIFAR10(str(data_dir), train=False, download=True)
    full = locked_split(Path(split_dir) if split_dir is not None else Path(run_dir)/'eda',
                        np.asarray(pool.targets,dtype=np.int64),cfg.get('split_seed',cfg['seed']))
    if len(full['test']) != len(test):
        raise ValueError('Locked split test size differs from official dataset')
    selected = {s: balanced_subset(v, pool.targets if s != "test" else test.targets,
                                  cfg[f"{s}_limit"], cfg.get('split_seed', cfg['seed']))
                for s, v in full.items()}
    np.savez(Path(run_dir) / "split_indices.npz", **full)
    np.savez(Path(run_dir) / "selected_indices.npz", **selected)
    return selected


class IndexedImages(Dataset):
    def __init__(self, data_dir, split, indices, name, image_size=224, augmentation=False):
        if augmentation and split != 'train':
            raise ValueError('Augmentation is only allowed for train')
        self.raw = datasets.CIFAR10(str(data_dir), train=split != "test")
        self.indices = np.asarray(indices)
        self.transform = image_transform(name, image_size)
        if augmentation:
            # Mild translations at original CIFAR resolution, then standard preprocessing.
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(p=.5), self.transform])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        original = int(self.indices[i])
        image, label = self.raw[original]
        return self.transform(image), label, original


def image_loader(data_dir, split, indices, name, batch, seed, shuffle=False, image_size=224,
                 augmentation=False):
    import torch
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(IndexedImages(data_dir, split, indices, name, image_size, augmentation), batch_size=batch,
                      shuffle=shuffle, num_workers=0, generator=generator,
                      pin_memory=torch.cuda.is_available())



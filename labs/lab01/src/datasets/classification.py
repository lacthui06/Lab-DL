"""FashionMNIST classification dataset and DataLoaders."""

from pathlib import Path
from typing import List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .base import TransformedSubset, load_split


def resolve_normalization_stats(eda_dir: Optional[Path] = None) -> Tuple[Tuple[float], Tuple[float]]:
    """Loads measured train-set normalization statistics or returns standard fallback."""
    if eda_dir is not None:
        cfg_path = Path(eda_dir) / "preprocessing_config.json"
        if cfg_path.exists():
            import json
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            norm = data.get("normalization", {})
            return tuple(norm.get("mean", [0.2860])), tuple(norm.get("std", [0.3530]))
    return (0.2860,), (0.3530,)


def get_transforms(
    augment: bool = False,
    mean: Optional[Tuple[float]] = None,
    std: Optional[Tuple[float]] = None,
) -> transforms.Compose:
    """Builds torchvision transformation pipeline."""
    if mean is None or std is None:
        try:
            from ..config import FASHION_MNIST_MEAN, FASHION_MNIST_STD
            mean = FASHION_MNIST_MEAN if mean is None else mean
            std = FASHION_MNIST_STD if std is None else std
        except Exception:
            mean = (0.2860,) if mean is None else mean
            std = (0.3530,) if std is None else std

    t_list = []
    if augment:
        t_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ])
    t_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transforms.Compose(t_list)


def get_dataloaders(
    data_dir: Path,
    split_dir: Optional[Path] = None,
    batch_size: int = 64,
    val_split: float = 0.1,
    augment: bool = False,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Constructs train, validation, and test DataLoaders for FashionMNIST."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]

    # 1. Download/load raw datasets
    raw_train_val = datasets.FashionMNIST(root=str(data_dir), train=True, download=True, transform=None)
    raw_test = datasets.FashionMNIST(root=str(data_dir), train=False, download=True, transform=None)

    # 2. Resolve split
    if split_dir is not None and (Path(split_dir) / "split_indices.npz").exists():
        train_idx, val_idx = load_split(split_dir, expected_pool_size=len(raw_train_val))
    else:
        from .base import make_stratified_split
        targets = [raw_train_val[i][1] for i in range(len(raw_train_val))]
        train_idx, val_idx = make_stratified_split(targets, val_split=val_split, seed=seed)

    mean, std = resolve_normalization_stats(split_dir)
    train_transform = get_transforms(augment=augment, mean=mean, std=std)
    eval_transform = get_transforms(augment=False, mean=mean, std=std)

    train_subset = TransformedSubset(Subset(raw_train_val, train_idx.tolist()), transform=train_transform)
    val_subset = TransformedSubset(Subset(raw_train_val, val_idx.tolist()), transform=eval_transform)
    test_dataset = TransformedSubset(Subset(raw_test, list(range(len(raw_test)))), transform=eval_transform)

    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, class_names

"""FashionMNIST classification dataset and DataLoaders."""

from pathlib import Path
from typing import List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .base import TransformedSubset, load_split
from ..utils.helpers import (
    AUGMENTATIONS_CONFIG_PATH,
    CLASS_NAMES,
    DATA_DIR,
    EDA_DIR,
    IMAGE_SIZE,
    load_yaml_config,
    resolve_normalization_stats,
)


def get_transforms(
    augment: bool = False,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    aug_config_path: Optional[Path] = None,
) -> transforms.Compose:
    """Builds torchvision transformation pipeline connected to config/augmentations.yaml."""
    if mean is None or std is None:
        measured_mean, measured_std, _ = resolve_normalization_stats()
        mean = measured_mean if mean is None else mean
        std = measured_std if std is None else std

    t_list = []
    if augment:
        cfg_path = Path(aug_config_path) if aug_config_path is not None else AUGMENTATIONS_CONFIG_PATH
        if cfg_path.exists():
            aug_cfg = load_yaml_config(cfg_path).get("train", {})
            hflip = aug_cfg.get("random_horizontal_flip", {})
            if hflip.get("enabled", False):
                prob = float(hflip.get("p", hflip.get("prob", 0.5)))
                t_list.append(transforms.RandomHorizontalFlip(p=prob))

            rot = aug_cfg.get("random_rotation", {})
            if rot.get("enabled", False):
                degrees = float(rot.get("degrees", 10))
                t_list.append(transforms.RandomRotation(degrees=degrees))

            crop = aug_cfg.get("random_crop", {})
            if crop.get("enabled", False):
                size = int(crop.get("size", IMAGE_SIZE))
                padding = int(crop.get("padding", 2))
                t_list.append(transforms.RandomCrop(size=size, padding=padding))
        else:
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
    data_dir: Optional[Path] = None,
    split_dir: Optional[Path] = None,
    batch_size: int = 64,
    val_split: float = 0.1,
    augment: bool = False,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Constructs train, validation, and test DataLoaders for FashionMNIST."""
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    split_dir = Path(split_dir) if split_dir is not None else EDA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download/load raw datasets
    raw_train_val = datasets.FashionMNIST(root=str(data_dir), train=True, download=True, transform=None)
    raw_test = datasets.FashionMNIST(root=str(data_dir), train=False, download=True, transform=None)

    # 2. Load locked split and measured normalization stats from EDA artifacts
    train_idx, val_idx = load_split(split_dir, expected_pool_size=len(raw_train_val))
    mean, std, _ = resolve_normalization_stats(split_dir)

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

    return train_loader, val_loader, test_loader, list(CLASS_NAMES)

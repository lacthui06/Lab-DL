"""Dataset and DataLoader module for FashionMNIST.
Implements transforms, strict train/val splitting without data leakage,
and clean PyTorch Dataset/DataLoader pipelines.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

from .config import (
    DATA_DIR,
    EDA_DIR,
    RANDOM_SEED,
    CLASS_NAMES,
    _resolve_normalization,
)
from .split import load_split


def get_transforms(augment: bool = False) -> transforms.Compose:
    """Returns torchvision transforms.
    
    Args:
        augment: If True, includes data augmentation (RandomHorizontalFlip, RandomRotation).
    
    Returns:
        transforms.Compose pipeline.
    """
    mean, std, _ = _resolve_normalization()
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class TransformedSubset(Dataset):
    """Wraps a torch.utils.data.Subset to apply custom transforms dynamically.
    Ensures Train set gets augmentations while Validation set stays unaugmented.
    """
    def __init__(self, subset: torch.utils.data.Subset, transform: transforms.Compose = None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image, label = self.subset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self) -> int:
        return len(self.subset)


def get_dataloaders(
    data_dir: Path = DATA_DIR,
    split_dir: Path = EDA_DIR,
    batch_size: int = 64,
    val_split: float = 0.1,
    augment: bool = False,
    num_workers: int = 0,
    seed: int = RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Loads FashionMNIST, applies the persisted stratified split, and returns DataLoaders.

    Args:
        data_dir: Directory where FashionMNIST raw data is stored.
        split_dir: Directory containing 'split_indices.npz' produced by EDA.
        batch_size: Batch size for training and evaluation.
        val_split: Expected validation proportion (0.1 = 10%).
        augment: Whether to apply data augmentation on the training set.
        num_workers: DataLoader worker processes.
        seed: Random seed for reproducibility.

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    # 1. Load official FashionMNIST sets
    raw_train_val = datasets.FashionMNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=None,
    )

    raw_test = datasets.FashionMNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=None,
    )

    # 2. Load the persisted stratified split (must exist from EDA)
    train_indices, val_indices = load_split(split_dir, expected_pool_size=len(raw_train_val))

    expected_val_size = int(len(raw_train_val) * val_split)
    if len(val_indices) != expected_val_size:
        raise ValueError(
            f"The persisted split reserves {len(val_indices)} validation samples but "
            f"val_split={val_split} on {len(raw_train_val)} images implies "
            f"{expected_val_size}. The caller and the split artifact disagree."
        )

    train_subset = Subset(raw_train_val, train_indices.tolist())
    val_subset = Subset(raw_train_val, val_indices.tolist())

    # 3. Apply appropriate transforms
    train_transform = get_transforms(augment=augment)
    eval_transform = get_transforms(augment=False)

    train_dataset = TransformedSubset(train_subset, transform=train_transform)
    val_dataset = TransformedSubset(val_subset, transform=eval_transform)
    test_dataset = TransformedSubset(raw_test, transform=eval_transform)

    # 4. Construct DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, CLASS_NAMES

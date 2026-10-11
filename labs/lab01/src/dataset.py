"""Dataset and DataLoader module for FashionMNIST.
Implements transforms, strict train/val splitting without data leakage,
and clean PyTorch Dataset/DataLoader pipelines.
"""

from pathlib import Path
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms
from PIL import Image

from .config import (
    DATA_DIR,
    RANDOM_SEED,
    FASHION_MNIST_MEAN,
    FASHION_MNIST_STD,
    CLASS_NAMES,
)


def get_transforms(augment: bool = False) -> transforms.Compose:
    """Returns torchvision transforms.
    
    Args:
        augment: If True, includes data augmentation (RandomHorizontalFlip, RandomRotation).
    
    Returns:
        transforms.Compose pipeline.
    """
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
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


class CustomFashionMNISTDataset(Dataset):
    """Demonstrates a standalone Custom PyTorch Dataset.
    Meets the educational learning goals:
    - __init__: stores metadata / tensors
    - __len__: returns dataset size
    - __getitem__: performs on-the-fly transformations (lazy pipeline)
    """
    def __init__(self, images: torch.Tensor, targets: torch.Tensor, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        target = self.targets[idx]
        
        # Convert torch tensor/numpy to PIL Image if transform expects PIL Image
        if isinstance(image, torch.Tensor):
            image = Image.fromarray(image.numpy(), mode="L")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image, mode="L")
            
        if self.transform is not None:
            image = self.transform(image)
            
        return image, target


def get_dataloaders(
    data_dir: Path = DATA_DIR,
    batch_size: int = 64,
    val_split: float = 0.1,
    augment: bool = False,
    seed: int = RANDOM_SEED,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Loads FashionMNIST, splits train/val, and returns DataLoaders.
    
    Args:
        data_dir: Directory where data is stored.
        batch_size: Mini-batch size.
        val_split: Fraction of training dataset reserved for validation (e.g. 0.1 = 10%).
        augment: Whether to apply data augmentation to training data.
        seed: Random seed for reproducible splitting.
        num_workers: Subprocesses for data loading.
        
    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    # 1. Load raw datasets without transforms (PIL images)
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
    
    # 2. Split train into train and validation sets reproducibility
    total_train = len(raw_train_val)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        raw_train_val, [train_size, val_size], generator=generator
    )
    
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


def get_kfold_pool(
    data_dir: Path = DATA_DIR,
    augment: bool = True,
    seed: int = RANDOM_SEED,
):
    """Builds the Stratified K-Fold training pool from the official train split.

    The full 60,000-image training partition is used as the cross-validation
    pool, leaving the official 10,000-image test split untouched as the final
    held-out benchmark. Two views over the same raw data are returned so that
    the training fold can be augmented while the validation fold stays clean.

    Args:
        data_dir: Directory where FashionMNIST is stored.
        augment: Whether the training-fold view applies data augmentation.
        seed: Seed reserved for reproducibility of any downstream sampling.

    Returns:
        (pool_train, pool_eval, targets, class_names)
        - pool_train: Dataset view with the training transform (augmented if requested).
        - pool_eval: Dataset view with the deterministic evaluation transform.
        - targets: 1D LongTensor of ground-truth labels for the 60,000 samples.
        - class_names: Ordered list of the 10 class names.
    """
    raw_pool = datasets.FashionMNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=None,
    )

    pool_train = TransformedSubset(raw_pool, transform=get_transforms(augment=augment))
    pool_eval = TransformedSubset(raw_pool, transform=get_transforms(augment=False))
    targets = raw_pool.targets.clone().detach().long()

    return pool_train, pool_eval, targets, CLASS_NAMES


def build_fold_loaders(
    pool_train: Dataset,
    pool_eval: Dataset,
    train_indices,
    val_indices,
    batch_size: int = 64,
    num_workers: int = 0,
):
    """Builds the train and validation DataLoaders for a single CV fold.

    Args:
        pool_train: Dataset view with the training transform.
        pool_eval: Dataset view with the evaluation transform.
        train_indices: Iterable of indices forming the training fold.
        val_indices: Iterable of indices forming the validation fold.
        batch_size: Mini-batch size.
        num_workers: Subprocesses for data loading.

    Returns:
        (fold_train_loader, fold_val_loader)
    """
    fold_train = Subset(pool_train, list(train_indices))
    fold_val = Subset(pool_eval, list(val_indices))

    fold_train_loader = DataLoader(
        fold_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    fold_val_loader = DataLoader(
        fold_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return fold_train_loader, fold_val_loader

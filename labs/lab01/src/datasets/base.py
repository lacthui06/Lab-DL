"""Base dataset wrapper and stratified split utilities."""

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms


class TransformedSubset(Dataset):
    """Wraps a torch.utils.data.Subset to apply transforms dynamically.
    Ensures Train set can receive augmentations while Validation stays unaugmented.
    """

    def __init__(self, subset: Subset, transform: Optional[transforms.Compose] = None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image, label = self.subset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self) -> int:
        return len(self.subset)


def make_stratified_split(
    targets: Sequence[int],
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates deterministic stratified split indices."""
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    dummy_x = np.zeros(len(targets))
    train_idx, val_idx = next(splitter.split(dummy_x, targets))
    return np.asarray(train_idx), np.asarray(val_idx)


def save_split(
    out_dir: Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    targets: Sequence[int],
    seed: int = 42,
    val_split: float = 0.1,
) -> Path:
    """Persists split indices and metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "split_indices.npz", train_indices=train_idx, val_indices=val_idx)

    meta_path = out_dir / "split_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": "split_v2_stratified",
            "val_split": val_split,
            "seed": seed,
            "train_count": len(train_idx),
            "val_count": len(val_idx),
            "total_count": len(train_idx) + len(val_idx),
        }, f, indent=2)
    return meta_path


def load_split(split_dir: Path, expected_pool_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Loads saved split indices from split_indices.npz."""
    npz_path = Path(split_dir) / "split_indices.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing required split artifact at: {npz_path}")

    data = np.load(npz_path)
    train_idx = data["train_indices"]
    val_idx = data["val_indices"]

    if expected_pool_size is not None:
        total = len(train_idx) + len(val_idx)
        if total != expected_pool_size:
            raise ValueError(f"Split artifact size mismatch: expected {expected_pool_size}, got {total}")

    return train_idx, val_idx

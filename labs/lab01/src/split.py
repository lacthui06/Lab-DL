"""Single-source-of-truth stratified split for Lab 01.

EDA, Feature Engineering and all three training runs must see exactly the same
train/validation partition, otherwise the runs are not comparable and the
normalization statistics measured on one partition would be applied to another.
This module is the only place where the split is created, persisted and reloaded.

Design notes:
- The official 10,000-image test partition is never touched here. Only the
  60,000-image original training partition is split.
- `sklearn.model_selection.train_test_split` with `stratify` is used instead of
  `torch.utils.data.random_split` because the latter does not preserve the class
  ratio: a seeded `random_split` of FashionMNIST produced validation folds with
  537 to 658 samples per class (a +/-10% deviation) instead of the expected 600.
- The indices are persisted so a later run can prove it used the same partition
  instead of silently re-deriving one.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .config import RANDOM_SEED, SPLIT_VERSION, VAL_SPLIT

SPLIT_FILENAME = "split_indices.npz"
SPLIT_METADATA_FILENAME = "split_metadata.json"


def make_stratified_split(
    targets,
    val_split: float = VAL_SPLIT,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Splits the original training partition into train and validation indices.

    Args:
        targets: 1D array-like of integer labels for the 60,000 original train
            samples, ordered as in the torchvision FashionMNIST dataset.
        val_split: Fraction reserved for validation (0.1 -> 6,000 samples).
        seed: Random state for the split. Must stay fixed across the whole lab.

    Returns:
        (train_indices, val_indices), both sorted 1D int64 arrays.

    Raises:
        ValueError: If the label array is empty, or if stratification would
            produce an empty class in either partition.
    """
    from sklearn.model_selection import train_test_split

    y = np.asarray(targets)
    if y.ndim != 1:
        raise ValueError(f"targets must be 1D, got shape {y.shape}")
    if y.size == 0:
        raise ValueError("targets is empty; the dataset was not loaded")

    indices = np.arange(y.size)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )

    # Guard: a stratified split must give every class at least one sample on
    # both sides. A silent fallback would let a broken split reach training.
    for name, idx in (("train", train_idx), ("val", val_idx)):
        counts = np.bincount(y[idx], minlength=int(y.max()) + 1)
        if counts.min() == 0:
            raise ValueError(
                f"Stratified split produced an empty class in the {name} partition: "
                f"{counts.tolist()}"
            )

    if np.intersect1d(train_idx, val_idx).size != 0:
        raise ValueError("Train and validation indices overlap")
    if len(train_idx) + len(val_idx) != y.size:
        raise ValueError(
            f"Split does not cover the pool: {len(train_idx)} + {len(val_idx)} != {y.size}"
        )

    return np.sort(train_idx).astype(np.int64), np.sort(val_idx).astype(np.int64)


def save_split(
    out_dir: Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    targets,
    seed: int = RANDOM_SEED,
    val_split: float = VAL_SPLIT,
    version: str = SPLIT_VERSION,
    extra: Optional[Dict] = None,
) -> Path:
    """Persists the split indices and their provenance metadata.

    The `.npz` file holds the indices used by the loaders; the `.json` file
    holds the metadata needed to prove which dataset, seed and ratio produced
    them. Both are written because indices without provenance cannot be audited.

    Args:
        out_dir: Destination directory (created if missing).
        train_idx: Training indices.
        val_idx: Validation indices.
        targets: Labels for the pool, used to record the per-class counts.
        seed: Seed recorded in the metadata.
        val_split: Validation fraction recorded in the metadata.
        version: Split version tag, e.g. "split_v2_stratified".
        extra: Additional metadata merged into the JSON record.

    Returns:
        Path to the written `.npz` file.
    """
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / SPLIT_FILENAME
    np.savez_compressed(npz_path, train_indices=train_idx, val_indices=val_idx)

    y = np.asarray(targets)
    metadata = {
        "split_version": version,
        "source_dataset": "FashionMNIST train=True (60,000 original train images)",
        "excluded_partition": "FashionMNIST train=False (10,000 official test images)",
        "seed": int(seed),
        "val_split": float(val_split),
        "shuffle": True,
        "stratified": True,
        "splitter": "sklearn.model_selection.train_test_split",
        "pool_size": int(y.size),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "train_class_counts": np.bincount(y[train_idx], minlength=10).tolist(),
        "val_class_counts": np.bincount(y[val_idx], minlength=10).tolist(),
    }
    if extra:
        metadata.update(extra)

    with open(out_dir / SPLIT_METADATA_FILENAME, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return npz_path


def load_split(
    split_dir: Path,
    expected_pool_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads previously persisted split indices.

    Args:
        split_dir: Directory containing `split_indices.npz`.
        expected_pool_size: When given, the reloaded indices must cover exactly
            this many samples. Guards against applying a split to the wrong pool.

    Returns:
        (train_indices, val_indices).

    Raises:
        FileNotFoundError: If the artifact is missing. The caller must not fall
            back to re-deriving a split, because a second derivation would make
            the run incomparable with EDA and with the other runs.
    """
    npz_path = Path(split_dir) / SPLIT_FILENAME
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Split artifact not found at {npz_path}. Run the EDA notebook first: "
            f"the split must be created once and reused by every later stage."
        )

    with np.load(npz_path) as data:
        train_idx = data["train_indices"]
        val_idx = data["val_indices"]

    if expected_pool_size is not None:
        covered = len(train_idx) + len(val_idx)
        if covered != expected_pool_size:
            raise ValueError(
                f"Split covers {covered} samples but the pool has {expected_pool_size}; "
                f"the artifact does not belong to this dataset."
            )
    return train_idx, val_idx


def load_split_metadata(split_dir: Path) -> Dict:
    """Loads the split provenance record written by `save_split`."""
    import json

    path = Path(split_dir) / SPLIT_METADATA_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Split metadata not found at {path}. Re-run the EDA notebook."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

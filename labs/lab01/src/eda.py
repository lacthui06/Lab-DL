"""Exploratory Data Analysis helpers for Lab 01 (FashionMNIST).

The notebook `notebooks/lab01_eda.ipynb` is the driver; this module holds the
computation so that the analysis is testable and so the same statistics can be
recomputed by the Feature Engineering notebook without a second implementation.

Scope follows docs/EDA1.md:
- Core: structural overview, integrity, class distribution, visual inspection,
  pixel intensity statistics.
- Extended (implemented): spatial analysis, inter-class similarity.
- Extended (deliberately not implemented): PCA, t-SNE/UMAP, outlier scoring,
  distribution-shift tests, traditional baseline. Reasons are recorded in the
  walkthrough rather than silently omitted.

Two rules from the spec are enforced in code rather than left to discipline:
- The official test partition is used only for structural and exact-duplicate
  checks, never for statistics that could inform a modelling decision.
- Pixel 0 is background, not a missing value, so foreground is defined by an
  explicit threshold instead of "pixel != 0".
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# A pixel is treated as foreground above this intensity (on the [0, 1] scale).
# The threshold is not 0 because 50.2% of FashionMNIST pixels are exactly zero:
# a "!= 0" rule would count faint anti-aliasing and JPEG-like noise at the edge
# of a garment as object pixels and would inflate every bounding box. 0.1 (about
# 26/255) keeps clearly-visible garment pixels while discarding near-black noise.
FOREGROUND_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_arrays(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reads the raw IDX files directly, bypassing torchvision transforms.

    The loaders in `src.dataset` return normalized float tensors, which are the
    wrong view for EDA: pixel range, dtype and zero-fraction must be measured on
    the raw uint8 data, not on a normalized tensor whose values leave [0, 1].

    Args:
        data_dir: Directory holding the FashionMNIST IDX files. torchvision
            nests them under `FashionMNIST/raw/`, so that layout is searched too.

    Returns:
        (X_train, y_train, X_test, y_test) where X_* is uint8 [N, 28, 28] and
        y_* is uint8 [N].

    Raises:
        FileNotFoundError: If any of the four IDX files is missing. A synthetic
            fallback would make every downstream statistic fictional.
    """
    import gzip

    root = Path(data_dir)

    def _resolve(name: str) -> Path:
        # Layouts produced by torchvision, in the order they are usually found.
        candidates = [
            root / name,
            root / "raw" / name,
            root / "FashionMNIST" / name,
            root / "FashionMNIST" / "raw" / name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Fall back to a recursive search so a differently nested copy still works.
        matches = sorted(root.glob(f"**/{name}"))
        if matches:
            return matches[0]

        raise FileNotFoundError(
            f"{name} not found under {root} (searched the flat, raw/, "
            f"FashionMNIST/ and FashionMNIST/raw/ layouts, plus a recursive glob). "
            f"Download FashionMNIST before running EDA."
        )

    def _read_images(name: str) -> np.ndarray:
        with gzip.open(_resolve(name), "rb") as f:
            blob = f.read()
        count = int.from_bytes(blob[4:8], "big")
        rows = int.from_bytes(blob[8:12], "big")
        cols = int.from_bytes(blob[12:16], "big")
        return np.frombuffer(blob, dtype=np.uint8, offset=16).reshape(count, rows, cols)

    def _read_labels(name: str) -> np.ndarray:
        with gzip.open(_resolve(name), "rb") as f:
            blob = f.read()
        return np.frombuffer(blob, dtype=np.uint8, offset=8).copy()

    return (
        _read_images("train-images-idx3-ubyte.gz"),
        _read_labels("train-labels-idx1-ubyte.gz"),
        _read_images("t10k-images-idx3-ubyte.gz"),
        _read_labels("t10k-labels-idx1-ubyte.gz"),
    )


# ---------------------------------------------------------------------------
# Step 1 - Structural overview
# ---------------------------------------------------------------------------

def structural_overview(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Builds one row per partition describing shape, dtype, range and memory.

    Args:
        X_train: Raw uint8 train images [N, H, W].
        y_train: Train labels.
        X_test: Raw uint8 test images.
        y_test: Test labels.
        class_names: Ordered class names, index i naming label i.

    Returns:
        DataFrame with one row per partition (train / test) plus a `summary`
        row holding the totals.
    """
    rows = []
    for name, X, y in (("original_train", X_train, y_train), ("original_test", X_test, y_test)):
        rows.append(
            {
                "partition": name,
                "num_images": int(len(X)),
                "image_height": int(X.shape[1]),
                "image_width": int(X.shape[2]),
                "num_channels": 1,  # grayscale, stored without a channel axis
                "array_shape": str(tuple(X.shape)),
                "dtype": str(X.dtype),
                "pixel_min": int(X.min()),
                "pixel_max": int(X.max()),
                "num_labels": int(len(y)),
                "label_min": int(y.min()),
                "label_max": int(y.max()),
                "num_classes_present": int(len(np.unique(y))),
                "num_class_names_defined": int(len(class_names)),
                "memory_MB": round(X.nbytes / (1024 ** 2), 3),
            }
        )

    overview = pd.DataFrame(rows)
    overview.loc[len(overview)] = {
        "partition": "summary",
        "num_images": int(len(X_train) + len(X_test)),
        "image_height": int(X_train.shape[1]),
        "image_width": int(X_train.shape[2]),
        "num_channels": 1,
        "array_shape": "n/a",
        "dtype": str(X_train.dtype),
        "pixel_min": int(min(X_train.min(), X_test.min())),
        "pixel_max": int(max(X_train.max(), X_test.max())),
        "num_labels": int(len(y_train) + len(y_test)),
        "label_min": int(min(y_train.min(), y_test.min())),
        "label_max": int(max(y_train.max(), y_test.max())),
        "num_classes_present": int(len(np.unique(np.concatenate([y_train, y_test])))),
        "num_class_names_defined": int(len(class_names)),
        "memory_MB": round((X_train.nbytes + X_test.nbytes) / (1024 ** 2), 3),
    }
    return overview


# ---------------------------------------------------------------------------
# Step 2 - Data integrity
# ---------------------------------------------------------------------------

def _md5_hashes(images: np.ndarray) -> List[str]:
    """Hashes each image by its exact byte content, for duplicate detection."""
    import hashlib

    flat = np.ascontiguousarray(images.reshape(len(images), -1))
    return [hashlib.md5(row.tobytes()).hexdigest() for row in flat]


def integrity_report(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    num_classes: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Runs the integrity checks and returns a summary plus the suspect records.

    The checks separate two ideas that are often conflated:
    - Index disjointness: the split does not reuse the same row twice.
    - Content duplication: two different rows hold identical pixels.
    A dataset can pass the first and fail the second, so both are measured.

    Args:
        X_train: Raw uint8 train images (the 60,000-image pool).
        y_train: Pool labels.
        X_test: Raw uint8 official test images.
        y_test: Official test labels.
        train_indices: Indices of the training partition.
        val_indices: Indices of the validation partition.
        num_classes: Expected number of classes.

    Returns:
        (summary_df, detail_dict). `detail_dict` maps a check name to the
        DataFrame of offending samples, each carrying source_dataset,
        original_index, split and label so a sample can be traced back.
    """
    details: Dict[str, pd.DataFrame] = {}
    split_of = np.full(len(y_train), "unassigned", dtype=object)
    split_of[train_indices] = "train"
    split_of[val_indices] = "val"

    def _frame(mask: np.ndarray, source: str = "original_train") -> pd.DataFrame:
        idx = np.flatnonzero(mask)
        return pd.DataFrame(
            {
                "source_dataset": source,
                "original_index": idx.astype(np.int64),
                "split": split_of[idx] if source == "original_train" else "test",
                "label": y_train[idx] if source == "original_train" else y_test[idx],
            }
        )

    checks: List[Dict] = []

    # --- Label validity -----------------------------------------------------
    bad_train_labels = (y_train < 0) | (y_train >= num_classes)
    bad_test_labels = (y_test < 0) | (y_test >= num_classes)
    details["invalid_train_labels"] = _frame(bad_train_labels)
    details["invalid_test_labels"] = _frame(bad_test_labels, source="original_test")
    checks.append(
        {
            "check": "labels_in_range",
            "expected": f"[0, {num_classes - 1}]",
            "num_flagged": int(bad_train_labels.sum() + bad_test_labels.sum()),
            "detail_file": "invalid_train_labels, invalid_test_labels",
        }
    )

    # --- Pixel range --------------------------------------------------------
    # uint8 cannot exceed [0, 255] by construction; the check is still recorded
    # because it would catch a cast to a wider dtype or a corrupted IDX header.
    out_train = int(((X_train < 0) | (X_train > 255)).sum())
    out_test = int(((X_test < 0) | (X_test > 255)).sum())
    checks.append(
        {
            "check": "pixel_range_0_255",
            "expected": "all pixels within [0, 255]",
            "num_flagged": out_train + out_test,
            "detail_file": "n/a",
        }
    )

    # --- NaN / Inf ----------------------------------------------------------
    # An integer array cannot hold NaN or Inf. The check is reported explicitly
    # so the absence of the problem is evidenced rather than assumed.
    non_finite_train = int((~np.isfinite(X_train.astype(np.float32))).sum())
    non_finite_test = int((~np.isfinite(X_test.astype(np.float32))).sum())
    checks.append(
        {
            "check": "nan_or_inf_pixels",
            "expected": "0 (uint8 storage cannot represent NaN/Inf)",
            "num_flagged": non_finite_train + non_finite_test,
            "detail_file": "n/a",
        }
    )

    # --- All-black and near-black images ------------------------------------
    train_max = X_train.reshape(len(X_train), -1).max(axis=1)
    test_max = X_test.reshape(len(X_test), -1).max(axis=1)
    all_black_train = train_max == 0
    all_black_test = test_max == 0
    details["all_black_train"] = _frame(all_black_train)
    details["all_black_test"] = _frame(all_black_test, source="original_test")
    checks.append(
        {
            "check": "all_black_images",
            "expected": "0 images with maximum pixel 0",
            "num_flagged": int(all_black_train.sum() + all_black_test.sum()),
            "detail_file": "all_black_train, all_black_test",
        }
    )

    # --- Split integrity ----------------------------------------------------
    overlap = int(np.intersect1d(train_indices, val_indices).size)
    covered = int(len(train_indices) + len(val_indices))
    checks.append(
        {
            "check": "split_indices_disjoint",
            "expected": "0 overlapping indices",
            "num_flagged": overlap,
            "detail_file": "n/a",
        }
    )
    checks.append(
        {
            "check": "split_coverage",
            "expected": f"{len(y_train)} samples covered exactly once",
            "num_flagged": int(abs(covered - len(y_train))),
            "detail_file": "n/a",
        }
    )

    # --- Exact duplicates (content, not index) ------------------------------
    train_hashes = _md5_hashes(X_train)
    test_hashes = _md5_hashes(X_test)
    hash_counts: Dict[str, List[int]] = {}
    for i, h in enumerate(train_hashes):
        hash_counts.setdefault(h, []).append(i)

    dup_groups = {h: idx for h, idx in hash_counts.items() if len(idx) > 1}
    dup_rows = []
    for h, idx in dup_groups.items():
        labels = sorted(set(int(y_train[i]) for i in idx))
        for i in idx:
            dup_rows.append(
                {
                    "source_dataset": "original_train",
                    "original_index": int(i),
                    "split": split_of[i],
                    "label": int(y_train[i]),
                    "hash": h,
                    "group_size": len(idx),
                    "group_labels": str(labels),
                    "label_conflict_within_group": len(labels) > 1,
                }
            )
    details["exact_duplicate_train"] = pd.DataFrame(
        dup_rows,
        columns=[
            "source_dataset", "original_index", "split", "label", "hash",
            "group_size", "group_labels", "label_conflict_within_group",
        ],
    )

    # Cross-partition duplicates: identical content in train and in test. This
    # is a leakage signal, not a modelling convenience, so it is reported even
    # when the count is zero. The official test set is never modified.
    train_hash_set = set(train_hashes)
    cross_rows = []
    for j, h in enumerate(test_hashes):
        if h in train_hash_set:
            cross_rows.append(
                {
                    "source_dataset": "original_test",
                    "original_index": int(j),
                    "split": "test",
                    "label": int(y_test[j]),
                    "hash": h,
                    "matching_train_indices": str(hash_counts[h]),
                }
            )
    details["exact_duplicate_train_test"] = pd.DataFrame(
        cross_rows,
        columns=[
            "source_dataset", "original_index", "split", "label", "hash",
            "matching_train_indices",
        ],
    )

    dup_train_rows = int(sum(len(idx) - 1 for idx in dup_groups.values()))
    checks.append(
        {
            "check": "exact_duplicate_within_train",
            "expected": "0 duplicated images (content hash)",
            "num_flagged": dup_train_rows,
            "detail_file": "exact_duplicate_train",
        }
    )
    checks.append(
        {
            "check": "exact_duplicate_train_vs_test",
            "expected": "0 identical images across partitions",
            "num_flagged": len(cross_rows),
            "detail_file": "exact_duplicate_train_test",
        }
    )

    # --- Near-duplicates ----------------------------------------------------
    # Not implemented: an all-pairs near-duplicate search over 60,000 images is
    # O(n^2) without an index and is out of scope for this lab. Recorded as an
    # explicit gap instead of being silently skipped.
    checks.append(
        {
            "check": "near_duplicates",
            "expected": "not implemented (out of scope; see walkthrough limitations)",
            "num_flagged": -1,
            "detail_file": "n/a",
        }
    )

    return pd.DataFrame(checks), details


# ---------------------------------------------------------------------------
# Step 3 - Class distribution
# ---------------------------------------------------------------------------

def class_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Counts and percentages per class for the train and validation partitions.

    The official test partition is intentionally excluded: the spec forbids
    using it for anything that could inform a modelling decision before the
    configuration is locked. Its counts are reported separately by the caller
    as a single aggregate check.

    Args:
        y_train: Labels of the training partition.
        y_val: Labels of the validation partition.
        class_names: Ordered class names.

    Returns:
        Long-format DataFrame: one row per (split, class).
    """
    rows = []
    for split_name, y in (("train", y_train), ("val", y_val)):
        counts = np.bincount(y, minlength=len(class_names))
        total = int(counts.sum())
        for i, name in enumerate(class_names):
            rows.append(
                {
                    "split": split_name,
                    "label": i,
                    "class_name": name,
                    "count": int(counts[i]),
                    "percent": round(100.0 * counts[i] / total, 4) if total else 0.0,
                    "split_total": total,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4 - Visual inspection
# ---------------------------------------------------------------------------

def sample_images_per_class(
    X: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str],
    n_per_class: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Selects a reproducible random sample of image indices per class.

    The first N rows of the dataset are not used: they are ordered, so the head
    of the file is not representative and would make the figure unreproducible
    under any other ordering.

    Args:
        X: Images [N, H, W].
        y: Labels [N].
        class_names: Ordered class names.
        n_per_class: Images to draw per class.
        seed: Seed for the draw.

    Returns:
        Integer array of shape [num_classes, n_per_class] holding pool indices.

    Raises:
        ValueError: If a class holds fewer than `n_per_class` samples.
    """
    rng = np.random.default_rng(seed)
    picks = []
    for i in range(len(class_names)):
        idx = np.flatnonzero(y == i)
        if idx.size < n_per_class:
            raise ValueError(
                f"Class {i} ({class_names[i]}) has {idx.size} samples but "
                f"{n_per_class} were requested"
            )
        picks.append(rng.choice(idx, size=n_per_class, replace=False))
    return np.vstack(picks)


def class_mean_images(
    X: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str],
) -> np.ndarray:
    """Computes the per-class mean image on the [0, 1] scale.

    Args:
        X: Images [N, H, W] as uint8 or float.
        y: Labels [N].
        class_names: Ordered class names.

    Returns:
        Float array [num_classes, H, W] in [0, 1].
    """
    scaled = X.astype(np.float32) / 255.0 if X.dtype == np.uint8 else X.astype(np.float32)
    return np.stack(
        [scaled[y == i].mean(axis=0) for i in range(len(class_names))]
    )


# ---------------------------------------------------------------------------
# Step 5 - Pixel intensity statistics
# ---------------------------------------------------------------------------

def compute_normalization_stats(X: np.ndarray) -> Dict[str, float]:
    """Computes the global mean and population standard deviation for Normalize.

    The statistics are pooled over every pixel of every image in the given
    partition, using population variance (ddof=0), which is what
    `torchvision.transforms.Normalize` assumes. Two shortcuts are explicitly
    rejected because they produce a different number:
    - averaging the per-image standard deviations gives ~0.3205, not the global
      standard deviation;
    - using ddof=1 gives the sample standard deviation, which is not the
      population quantity the transform expects.

    Args:
        X: Images [N, H, W] as uint8 or float on the [0, 1] scale.

    Returns:
        Dict with mean, std, num_images, num_pixels and a `per_image_std_mean`
        diagnostic so the difference from the rejected shortcut stays visible.
    """
    scaled = X.astype(np.float32) / 255.0 if X.dtype == np.uint8 else X.astype(np.float32)
    flat = scaled.reshape(len(scaled), -1)

    return {
        "mean": float(flat.mean()),
        "std": float(flat.std(ddof=0)),
        "num_images": int(len(flat)),
        "num_pixels": int(flat.size),
        # Diagnostic only: this is the quantity that must NOT be used as std.
        "per_image_std_mean": float(flat.std(axis=1, ddof=1).mean()),
        "per_image_std_ddof0_mean": float(flat.std(axis=1, ddof=0).mean()),
        "per_image_mean_mean": float(flat.mean(axis=1).mean()),
        "zero_pixel_fraction": float((flat == 0).mean()),
    }


def pixel_statistics_by_class(
    X: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Per-class pixel statistics on the [0, 1] scale.

    Args:
        X: Images [N, H, W].
        y: Labels [N].
        class_names: Ordered class names.

    Returns:
        DataFrame with one row per class: global mean/std, zero fraction, and
        the distribution of the per-image brightness (mean intensity).
    """
    scaled = X.astype(np.float32) / 255.0 if X.dtype == np.uint8 else X.astype(np.float32)
    rows = []
    for i, name in enumerate(class_names):
        sub = scaled[y == i]
        flat = sub.reshape(len(sub), -1)
        per_image_mean = flat.mean(axis=1)
        rows.append(
            {
                "label": i,
                "class_name": name,
                "count": int(len(sub)),
                "pixel_mean": round(float(flat.mean()), 6),
                "pixel_std": round(float(flat.std(ddof=0)), 6),
                "zero_pixel_fraction": round(float((flat == 0).mean()), 6),
                "image_brightness_mean": round(float(per_image_mean.mean()), 6),
                "image_brightness_std": round(float(per_image_mean.std(ddof=0)), 6),
                "image_brightness_min": round(float(per_image_mean.min()), 6),
                "image_brightness_max": round(float(per_image_mean.max()), 6),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 6 - Spatial analysis
# ---------------------------------------------------------------------------

def spatial_features(
    X: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str],
    threshold: float = FOREGROUND_THRESHOLD,
) -> pd.DataFrame:
    """Per-image spatial descriptors derived from a foreground mask.

    Args:
        X: Images [N, H, W] as uint8 or float on [0, 1].
        y: Labels [N].
        class_names: Ordered class names.
        threshold: Foreground intensity threshold on the [0, 1] scale.

    Returns:
        DataFrame with one row per image: foreground_ratio, bbox_height,
        bbox_width, bbox_aspect_ratio, bbox_area_ratio, and an empty_mask flag
        for images whose foreground is empty at this threshold.
    """
    scaled = X.astype(np.float32) / 255.0 if X.dtype == np.uint8 else X.astype(np.float32)
    mask = scaled > threshold

    foreground_ratio = mask.mean(axis=(1, 2))
    rows_any = mask.any(axis=2)
    cols_any = mask.any(axis=1)
    bbox_height = rows_any.sum(axis=1)
    bbox_width = cols_any.sum(axis=1)

    height = scaled.shape[1]
    width = scaled.shape[2]
    empty = (bbox_height == 0) | (bbox_width == 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        aspect = np.where(bbox_height > 0, bbox_width / bbox_height, np.nan)

    return pd.DataFrame(
        {
            "source_dataset": "original_train",
            "original_index": np.arange(len(scaled), dtype=np.int64),
            "split": "unassigned",  # filled by the caller from the split artifact
            "label": y.astype(np.int64),
            "class_name": [class_names[int(v)] for v in y],
            "foreground_threshold": threshold,
            "foreground_ratio": foreground_ratio.astype(np.float64),
            "bbox_height": bbox_height.astype(np.int64),
            "bbox_width": bbox_width.astype(np.int64),
            "bbox_aspect_ratio": aspect.astype(np.float64),
            "bbox_area_ratio": (bbox_height * bbox_width / (height * width)).astype(np.float64),
            "empty_foreground": empty,
        }
    )


def mean_pixel_map(X: np.ndarray) -> np.ndarray:
    """Mean image over a set of images, on the [0, 1] scale."""
    scaled = X.astype(np.float32) / 255.0 if X.dtype == np.uint8 else X.astype(np.float32)
    return scaled.mean(axis=0)


def pixel_variance_map(X: np.ndarray) -> np.ndarray:
    """Per-pixel population variance over a set of images, on the [0, 1] scale."""
    scaled = X.astype(np.float32) / 255.0 if X.dtype == np.uint8 else X.astype(np.float32)
    return scaled.var(axis=0, ddof=0)


def zero_pixel_frequency_map(X: np.ndarray) -> np.ndarray:
    """Fraction of images in which each pixel is exactly 0.

    Reported alongside the variance map because "this pixel is always background"
    and "this pixel is highly variable" are different statements: a border pixel
    can be zero in 99% of images and still carry the largest variance.
    """
    return (X == 0).mean(axis=0)


# ---------------------------------------------------------------------------
# Step 8 - Inter-class similarity
# ---------------------------------------------------------------------------

def class_similarity_matrix(
    mean_images: np.ndarray,
    class_names: Sequence[str],
    metric: str = "cosine",
) -> pd.DataFrame:
    """Similarity between per-class mean images.

    Both metrics are offered because they answer slightly different questions:
    cosine compares the shape of the intensity profile after removing magnitude,
    correlation additionally removes the per-image mean level. The result is a
    hypothesis generator only — the spec forbids concluding that two classes are
    hard to separate from mean-image similarity alone, because averaging
    destroys the within-class variation that actually drives confusion.

    Args:
        mean_images: Array [num_classes, H, W].
        class_names: Ordered class names.
        metric: 'cosine' or 'correlation'.

    Returns:
        Square DataFrame indexed and columned by class name.

    Raises:
        ValueError: If the metric is unsupported.
    """
    flat = mean_images.reshape(len(mean_images), -1).astype(np.float64)

    if metric == "cosine":
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("A class mean image is all zeros; cosine is undefined")
        normalised = flat / norms
        sim = normalised @ normalised.T
    elif metric == "correlation":
        centred = flat - flat.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centred, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("A class mean image is constant; correlation is undefined")
        normalised = centred / norms
        sim = normalised @ normalised.T
    else:
        raise ValueError(f"Unsupported metric '{metric}'. Choose 'cosine' or 'correlation'.")

    return pd.DataFrame(sim, index=list(class_names), columns=list(class_names))


def most_similar_pairs(similarity: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Extracts the highest-scoring off-diagonal class pairs from a similarity matrix.

    Args:
        similarity: Square DataFrame from `class_similarity_matrix`.
        top_k: Number of pairs to return.

    Returns:
        DataFrame with class_a, class_b, similarity, sorted descending. Each
        unordered pair appears once.
    """
    names = list(similarity.index)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append(
                {
                    "class_a": names[i],
                    "class_b": names[j],
                    "similarity": round(float(similarity.iloc[i, j]), 6),
                }
            )
    out = pd.DataFrame(pairs).sort_values("similarity", ascending=False)
    return out.head(top_k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
# The plotting helpers live next to the computations so that a figure and the
# number behind it cannot drift apart. Each one takes already-computed arrays or
# DataFrames and writes a PNG; none of them reads the dataset, so the notebook
# controls what is measured and these functions only render it.

def plot_sample_grid(
    X: np.ndarray,
    picks: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str],
    save_path: Optional[Path] = None,
):
    """Renders the sampled images as a class-by-sample grid.

    Args:
        X: Images [N, H, W].
        picks: Index array [num_classes, n_per_class] from `sample_images_per_class`.
        y: Labels [N].
        class_names: Ordered class names.
        save_path: Optional destination for the PNG.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    num_classes, n_per_class = picks.shape
    fig, axes = plt.subplots(
        num_classes, n_per_class, figsize=(n_per_class * 0.95, num_classes * 1.15)
    )
    axes = np.atleast_2d(axes)

    for row in range(num_classes):
        for col in range(n_per_class):
            ax = axes[row, col]
            ax.imshow(X[picks[row, col]], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(
                    f"{row}: {class_names[row]}",
                    fontsize=9,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=52,
                )

    fig.suptitle(
        f"FashionMNIST: {n_per_class} sampled images per class (train pool, seed-fixed draw)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.01, 0, 1, 0.97))
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    return fig


def plot_class_mean_images(
    mean_images: np.ndarray,
    class_names: Sequence[str],
    save_path: Optional[Path] = None,
):
    """Renders one mean image per class in a single row of subplots.

    Args:
        mean_images: Array [num_classes, H, W] on the [0, 1] scale.
        class_names: Ordered class names.
        save_path: Optional destination for the PNG.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    num_classes = len(mean_images)
    fig, axes = plt.subplots(1, num_classes, figsize=(num_classes * 1.9, 2.6))
    axes = np.atleast_1d(axes)

    for i, ax in enumerate(axes):
        ax.imshow(mean_images[i], cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"{i}: {class_names[i]}", fontsize=8)
        ax.axis("off")

    fig.suptitle(
        "Mean image per class (train partition, [0, 1] scale). Averaging smooths "
        "away the within-class variation that drives confusion.",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    return fig


def plot_class_distribution(
    distribution: pd.DataFrame,
    save_path: Optional[Path] = None,
):
    """Bar chart of per-class counts for train and validation.

    Args:
        distribution: Long-format DataFrame from `class_distribution`.
        save_path: Optional destination for the PNG.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    splits = list(distribution["split"].unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(6.5 * len(splits), 4.2))
    axes = np.atleast_1d(axes)

    for ax, split_name in zip(axes, splits):
        sub = distribution[distribution["split"] == split_name]
        ax.bar(sub["class_name"], sub["count"], color="#4c72b0")
        ax.set_title(f"{split_name} partition (n={int(sub['split_total'].iloc[0])})", fontsize=11)
        ax.set_ylabel("Number of images")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        for x, (count, pct) in enumerate(zip(sub["count"], sub["percent"])):
            ax.annotate(
                f"{int(count)}\n{pct:.2f}%",
                xy=(x, count),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        ax.margins(y=0.18)

    fig.suptitle(
        "Class distribution (measured, not assumed balanced)", fontsize=12, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    return fig


def plot_pixel_histogram(
    X_train: np.ndarray,
    X_val: np.ndarray,
    save_path: Optional[Path] = None,
    bins: int = 64,
):
    """Overlaid pixel-intensity histograms for train and validation.

    Both partitions are drawn on the [0, 1] scale with the same bin edges so the
    two distributions are directly comparable. The spike at 0 is the background,
    which is the reason the foreground threshold is not simply "pixel != 0".

    Args:
        X_train: Training images [N, H, W].
        X_val: Validation images [N, H, W].
        save_path: Optional destination for the PNG.
        bins: Histogram bins over [0, 1].

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    train_flat = (X_train.astype(np.float32) / 255.0).ravel()
    val_flat = (X_val.astype(np.float32) / 255.0).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    axes[0].hist(train_flat, bins=bins, range=(0.0, 1.0), color="#4c72b0", alpha=0.85)
    axes[0].set_yscale("log")
    axes[0].set_title("Train pixel intensity (log count)", fontsize=11)
    axes[0].set_xlabel("Pixel value on [0, 1]")
    axes[0].set_ylabel("Count (log)")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    edges = np.linspace(0.0, 1.0, bins + 1)
    axes[1].hist(train_flat, bins=edges, density=True, histtype="step", linewidth=2,
                 label="train", color="#4c72b0")
    axes[1].hist(val_flat, bins=edges, density=True, histtype="step", linewidth=2,
                 label="val", color="#dd8452")
    axes[1].set_yscale("log")
    axes[1].set_title("Train vs validation density (log)", fontsize=11)
    axes[1].set_xlabel("Pixel value on [0, 1]")
    axes[1].set_ylabel("Density (log)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        "Pixel intensity distribution on the [0, 1] scale (raw uint8 / 255)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    return fig


def plot_brightness_distribution(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_names: Sequence[str],
    save_path: Optional[Path] = None,
):
    """Per-class distribution of per-image mean intensity.

    Args:
        X_train: Training images [N, H, W].
        y_train: Training labels [N].
        class_names: Ordered class names.
        save_path: Optional destination for the PNG.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    scaled = X_train.astype(np.float32) / 255.0
    brightness = scaled.reshape(len(scaled), -1).mean(axis=1)

    data = [brightness[y_train == i] for i in range(len(class_names))]

    fig, ax = plt.subplots(figsize=(12, 4.6))
    box = ax.boxplot(
        data,
        tick_labels=[f"{i}\n{name}" for i, name in enumerate(class_names)],
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.35},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#4c72b0")
        patch.set_alpha(0.65)
    ax.set_title(
        "Per-image brightness (mean pixel intensity) by class, train partition",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("Mean pixel value on [0, 1]")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    return fig


def plot_heatmaps(
    mean_map: np.ndarray,
    variance_map: np.ndarray,
    zero_map: np.ndarray,
    save_path: Optional[Path] = None,
):
    """Renders the mean, variance and zero-frequency maps side by side.

    Args:
        mean_map: Mean image [H, W] on [0, 1].
        variance_map: Per-pixel variance [H, W].
        zero_map: Fraction of images where the pixel is exactly 0 [H, W].
        save_path: Optional destination for the PNG.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    panels = [
        (mean_map, "Mean pixel intensity", "gray"),
        (variance_map, "Pixel-wise variance", "viridis"),
        (zero_map, "Fraction of images with pixel == 0", "magma"),
    ]
    for ax, (data, title, cmap) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Spatial structure of the train partition", fontsize=12, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    return fig


def plot_spatial_by_class(
    spatial: pd.DataFrame,
    class_names: Sequence[str],
    save_path: Optional[Path] = None,
):
    """Box plots of the spatial descriptors, grouped by class.

    Args:
        spatial: DataFrame from `spatial_features`.
        class_names: Ordered class names.
        save_path: Optional destination for the PNG.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    metrics = [
        ("foreground_ratio", "Foreground ratio"),
        ("bbox_height", "Bounding-box height (px)"),
        ("bbox_width", "Bounding-box width (px)"),
        ("bbox_aspect_ratio", "Bounding-box aspect ratio (w/h)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()

    for ax, (column, label) in zip(axes, metrics):
        data = [
            spatial.loc[spatial["label"] == i, column].dropna().to_numpy()
            for i in range(len(class_names))
        ]
        box = ax.boxplot(
            data,
            tick_labels=[f"{i}\n{n}" for i, n in enumerate(class_names)],
            patch_artist=True,
            showfliers=False,
        )
        for patch in box["boxes"]:
            patch.set_facecolor("#55a868")
            patch.set_alpha(0.65)
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(
        f"Spatial descriptors by class (foreground threshold = {FOREGROUND_THRESHOLD})",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    return fig


def plot_similarity_heatmap(
    similarity: pd.DataFrame,
    metric: str,
    save_path: Optional[Path] = None,
):
    """Heatmap of the inter-class mean-image similarity matrix.

    Args:
        similarity: Square DataFrame from `class_similarity_matrix`.
        metric: Metric name, used in the title.
        save_path: Optional destination for the PNG.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    sns.heatmap(
        similarity,
        annot=True,
        fmt=".3f",
        cmap="rocket" if metric == "cosine" else "vlag",
        vmin=float(similarity.to_numpy().min()),
        vmax=1.0,
        square=True,
        cbar_kws={"label": f"{metric} similarity"},
        annot_kws={"size": 7},
        ax=ax,
    )
    ax.set_title(
        f"Inter-class mean-image {metric} similarity (train partition)\n"
        f"Hypothesis generator only: averaging hides within-class variation",
        fontsize=11,
        fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)

    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    return fig

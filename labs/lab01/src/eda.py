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
# Scaling
# ---------------------------------------------------------------------------

def to_scaled(images: np.ndarray) -> np.ndarray:
    """Converts uint8 images on [0, 255] to float32 on [0, 1].

    This is the single definition of the scaling step on the analysis path.
    `ToTensor()` performs the same division inside the training pipeline, and
    the Feature Engineering notebook calls this function instead of repeating
    the arithmetic. Splitting 255 in several places is a latent defect: change
    one copy and the reported statistics drift with no exception raised.

    Args:
        images: Array of any shape holding uint8 or float values.

    Returns:
        float32 array with the same shape. uint8 input is divided by 255;
        float input is assumed already scaled and only cast.

    Raises:
        TypeError: If the dtype is neither uint8 nor a floating type. Other
            integer types are ambiguous and are rejected on purpose.
    """
    if images.dtype == np.uint8:
        return images.astype(np.float32) / 255.0
    if np.issubdtype(images.dtype, np.floating):
        return images.astype(np.float32)
    raise TypeError(
        f"Expected uint8 or float input, got {images.dtype}."
    )


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
        A `num_flagged` of None means the check was not run, which is a
        different statement from zero.
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

    def _record(name: str, expected: str, flagged, detail: str = "n/a") -> None:
        """Appends one check result. Kept in one place so the column set cannot
        drift between checks, which is how the -1/None disagreement arose."""
        checks.append(
            {
                "check": name,
                "expected": expected,
                "num_flagged": flagged,
                "detail_file": detail,
            }
        )

    # --- Label validity -----------------------------------------------------
    bad_train = (y_train < 0) | (y_train >= num_classes)
    bad_test = (y_test < 0) | (y_test >= num_classes)
    details["invalid_train_labels"] = _frame(bad_train)
    details["invalid_test_labels"] = _frame(bad_test, source="original_test")
    _record(
        "labels_in_range",
        f"[0, {num_classes - 1}]",
        int(bad_train.sum() + bad_test.sum()),
        "invalid_train_labels, invalid_test_labels",
    )

    # --- Pixel range --------------------------------------------------------
    # uint8 cannot exceed [0, 255] by construction; the check is still recorded
    # because it would catch a cast to a wider dtype or a corrupted IDX header.
    out_of_range = int(((X_train < 0) | (X_train > 255)).sum())
    out_of_range += int(((X_test < 0) | (X_test > 255)).sum())
    _record("pixel_range_0_255", "all pixels within [0, 255]", out_of_range)

    # --- NaN / Inf ----------------------------------------------------------
    # An integer array cannot hold NaN or Inf. The check is reported explicitly
    # so the absence of the problem is evidenced rather than assumed.
    non_finite = int((~np.isfinite(X_train.astype(np.float32))).sum())
    non_finite += int((~np.isfinite(X_test.astype(np.float32))).sum())
    _record(
        "nan_or_inf_pixels",
        "0 (uint8 storage cannot represent NaN/Inf)",
        non_finite,
    )

    # --- All-black images ---------------------------------------------------
    all_black_train = X_train.reshape(len(X_train), -1).max(axis=1) == 0
    all_black_test = X_test.reshape(len(X_test), -1).max(axis=1) == 0
    details["all_black_train"] = _frame(all_black_train)
    details["all_black_test"] = _frame(all_black_test, source="original_test")
    _record(
        "all_black_images",
        "0 images with maximum pixel 0",
        int(all_black_train.sum() + all_black_test.sum()),
        "all_black_train, all_black_test",
    )

    # --- Split integrity ----------------------------------------------------
    overlap = int(np.intersect1d(train_indices, val_indices).size)
    _record("split_indices_disjoint", "0 overlapping indices", overlap)
    covered = int(len(train_indices) + len(val_indices))
    _record(
        "split_coverage",
        f"{len(y_train)} samples covered exactly once",
        int(abs(covered - len(y_train))),
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
    cross_rows = [
        {
            "source_dataset": "original_test",
            "original_index": int(j),
            "split": "test",
            "label": int(y_test[j]),
            "hash": h,
            "matching_train_indices": str(hash_counts[h]),
        }
        for j, h in enumerate(test_hashes)
        if h in train_hash_set
    ]
    details["exact_duplicate_train_test"] = pd.DataFrame(
        cross_rows,
        columns=[
            "source_dataset", "original_index", "split", "label", "hash",
            "matching_train_indices",
        ],
    )

    _record(
        "exact_duplicate_within_train",
        "0 duplicated images (content hash)",
        int(sum(len(idx) - 1 for idx in dup_groups.values())),
        "exact_duplicate_train",
    )
    _record(
        "exact_duplicate_train_vs_test",
        "0 identical images across partitions",
        len(cross_rows),
        "exact_duplicate_train_test",
    )

    # --- Near-duplicates ----------------------------------------------------
    # Not implemented: an all-pairs near-duplicate search over 60,000 images is
    # O(n^2) without an index and is out of scope for this lab. Recorded as an
    # explicit gap with a None count, never as a number, so it cannot be read as
    # "checked and clean".
    _record(
        "near_duplicates",
        "not implemented (out of scope; see walkthrough limitations)",
        None,
    )

    return pd.DataFrame(checks), details


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
    scaled = to_scaled(X)
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
    scaled = to_scaled(X)
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
    scaled = to_scaled(X)
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
    scaled = to_scaled(X)
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
    scaled = to_scaled(X)
    return scaled.mean(axis=0)


def pixel_variance_map(X: np.ndarray) -> np.ndarray:
    """Per-pixel population variance over a set of images, on the [0, 1] scale."""
    scaled = to_scaled(X)
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

    train_flat = to_scaled(X_train).ravel()
    val_flat = to_scaled(X_val).ravel()

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

    scaled = to_scaled(X_train)
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

# ---------------------------------------------------------------------------
# Step 13 - Handoff tables
# ---------------------------------------------------------------------------

def _flagged_count(integrity_summary: pd.DataFrame, check_name: str):
    """Reads the flagged count for one check from the integrity summary.

    Returns None when the check is absent, which means "not run". Returning a
    number such as -1 would be indistinguishable from a real count once written
    to JSON or CSV, and a reader would have no way to tell the two apart.
    """
    row = integrity_summary.loc[
        integrity_summary["check"] == check_name, "num_flagged"
    ]
    if not len(row):
        return None
    value = row.iloc[0]
    return None if pd.isna(value) else int(value)


def build_findings(
    split_meta: Dict,
    train_stats: Dict,
    spatial: pd.DataFrame,
    integrity_summary: pd.DataFrame,
    top_cosine: pd.DataFrame,
    top_correlation: pd.DataFrame,
    legacy_std: float,
    foreground_threshold: float,
    test_size: int,
) -> Dict:
    """Assembles the measured numbers into one JSON-serialisable dictionary.

    The commentary in the walkthrough is only allowed to quote numbers that
    appear here, so a claim cannot drift away from the artifact it came from.

    Args:
        split_meta: Metadata returned by `load_split_metadata`.
        train_stats: Output of `compute_normalization_stats` on train.
        spatial: Output of `spatial_features` on train.
        integrity_summary: Summary table from `integrity_report`.
        top_cosine: Output of `most_similar_pairs` for the cosine matrix.
        top_correlation: Output of `most_similar_pairs` for the correlation matrix.
        legacy_std: Standard deviation previously hard-coded in the config.
        foreground_threshold: Threshold used to define foreground.
        test_size: Number of images in the held-out test partition.

    Returns:
        Flat dictionary containing only JSON-safe scalar values.
    """
    train_counts = split_meta["train_class_counts"]
    val_counts = split_meta["val_class_counts"]

    return {
        "split_version": split_meta["split_version"],
        "train_size": int(split_meta["train_size"]),
        "val_size": int(split_meta["val_size"]),
        "test_size_held_out": int(test_size),
        "train_class_count_min": int(min(train_counts)),
        "train_class_count_max": int(max(train_counts)),
        "val_class_count_min": int(min(val_counts)),
        "val_class_count_max": int(max(val_counts)),
        "train_pixel_mean": round(float(train_stats["mean"]), 6),
        "train_pixel_std_ddof0": round(float(train_stats["std"]), 6),
        "train_per_image_std_mean": round(float(train_stats["per_image_std_mean"]), 6),
        "legacy_config_std": float(legacy_std),
        "zero_pixel_fraction_train": round(float(train_stats["zero_pixel_fraction"]), 6),
        "foreground_threshold": foreground_threshold,
        "foreground_ratio_mean": round(float(spatial["foreground_ratio"].mean()), 6),
        "empty_foreground_images": int(spatial["empty_foreground"].sum()),
        "exact_duplicate_within_train": _flagged_count(integrity_summary, "exact_duplicate_within_train"),
        "exact_duplicate_train_vs_test": _flagged_count(integrity_summary, "exact_duplicate_train_vs_test"),
        "all_black_images": _flagged_count(integrity_summary, "all_black_images"),
        "top_similar_pair_cosine": (
            f"{top_cosine.iloc[0]['class_a']} / {top_cosine.iloc[0]['class_b']} = "
            f"{top_cosine.iloc[0]['similarity']:.4f}"
        ),
        "top_similar_pair_correlation": (
            f"{top_correlation.iloc[0]['class_a']} / "
            f"{top_correlation.iloc[0]['class_b']} = "
            f"{top_correlation.iloc[0]['similarity']:.4f}"
        ),
    }


# Template rows for the EDA-to-FE handoff table. Placeholders in braces are
# filled from `findings` by `build_handoff_table`. The `status` column accepts
# exactly three values:
#   measured   - computed in this pass and reproducible from the artifacts
#   hypothesis - plausible but unconfirmed; the trained confusion matrix decides
#   not_run    - deliberately skipped, with the reason recorded in `proposal`
_HANDOFF_ROWS = [
    (
        "Split stratified {train_size:,}/{val_size:,}, moi lop "
        "{train_class_count_min:,}/{val_class_count_min:,}",
        "index-only, raw",
        "split_metadata.json, class_distribution.csv",
        "Dung dung artifact nay cho EDA, FE va moi run; khong chia lai",
        "assert min==max class count trong moi partition",
        "measured",
    ),
    (
        "Global train pixel mean={train_pixel_mean:.6f}, "
        "std(ddof=0)={train_pixel_std_ddof0:.6f}",
        "train, scaled [0,1], truoc augmentation",
        "normalization_stats_measurement.csv",
        "FE do lai tren dung artifact nay va ghi preprocessing_config.json",
        "std cua tensor sau Normalize phai ~1.0 (tolerance 0.02)",
        "measured",
    ),
    (
        "Config cu dung std={legacy_config_std:.4f}, bang mean cua per-image std "
        "({train_per_image_std_mean:.6f}), khong phai global std",
        "train, scaled [0,1]",
        "normalization_stats_measurement.csv (cot per_image_std_mean)",
        "Thay bang global std do tren split moi; version hoa preprocessing",
        "test_normalization_yields_unit_variance",
        "measured",
    ),
    (
        "Khong co exact duplicate trong train pool va khong co train-test "
        "overlap ({exact_duplicate_train_vs_test})",
        "raw, md5 tren byte",
        "integrity_report.csv",
        "Khong loai bo mau nao; giu nguyen dataset goc",
        "hash lai trong FE guard, phai van bang 0",
        "measured",
    ),
    (
        "Pixel 0 chiem {zero_pixel_fraction_train:.4f} tong pixel",
        "train, scaled [0,1]",
        "pixel_statistics.csv, pixel_histogram.png",
        "Foreground phai dung nguong, khong dung pixel != 0",
        "foreground_ratio khong doi nghia khi threshold thay doi nho",
        "measured",
    ),
    (
        "Cap lop giong nhau nhat (cosine): {top_similar_pair_cosine}",
        "train, mean image",
        "class_similarity_matrix.csv, class_similarity_heatmap.png",
        "Chi ghi la gia thuyet ve cap lop de nham lan",
        "confusion matrix phai xac nhan hoac bac bo",
        "hypothesis",
    ),
    (
        "Handcrafted features (Step 7)",
        "n/a",
        "n/a",
        "Khong dua vao model; chi dung de mo ta",
        "n/a",
        "not_run",
    ),
    (
        "PCA / t-SNE (Step 9)",
        "n/a",
        "n/a",
        "Khong thuc hien: khong phuc vu quyet dinh nao cho cac run MLP",
        "n/a",
        "not_run",
    ),
    (
        "Outlier scoring (Step 10)",
        "n/a",
        "n/a",
        "Khong thuc hien: spec cam loai bo mau chi vi bi danh dau outlier",
        "n/a",
        "not_run",
    ),
    (
        "Distribution shift train/val (Step 11)",
        "n/a",
        "pixel_histogram.png (so sanh dinh tinh)",
        "Khong chay KS test: hai tap cung mot pool, khac biet chu yeu la "
        "sampling noise",
        "n/a",
        "not_run",
    ),
    (
        "Traditional baseline LR/kNN (Step 12)",
        "n/a",
        "n/a",
        "Khong thuc hien: baseline cua lab la BaselineMLP",
        "n/a",
        "not_run",
    ),
]

_HANDOFF_COLUMNS = [
    "finding", "split_view", "evidence", "proposal", "validation_check", "status",
]


def build_handoff_table(findings: Dict, train_stats: Dict) -> pd.DataFrame:
    """Builds the EDA-to-Feature-Engineering handoff table.

    Each row carries the finding, the split and view it was measured on, the
    artifact that proves it, the proposed action, and the check that would
    falsify it. Rows live in `_HANDOFF_ROWS`; this function only fills the
    measured values into them.

    Args:
        findings: Output of `build_findings`.
        train_stats: Output of `compute_normalization_stats` on train. Accepted
            for call-site compatibility; the values it would supply are already
            inside `findings`.

    Returns:
        DataFrame with one row per finding, ready to write to CSV.
    """
    rows = [
        dict(zip(_HANDOFF_COLUMNS, (row[0].format(**findings),) + tuple(row[1:])))
        for row in _HANDOFF_ROWS
    ]
    return pd.DataFrame(rows, columns=_HANDOFF_COLUMNS)


def build_data_review(integrity_summary: pd.DataFrame) -> pd.DataFrame:
    """Turns the integrity summary into an explicit keep/drop decision table.

    The EDA pass only reports. This function records what is done about each
    reported issue, so a later reader can tell "nothing was found" apart from
    "something was found and deliberately kept".

    Args:
        integrity_summary: Summary table from `integrity_report`.

    Returns:
        DataFrame with issue, count, decision, and rationale. A `count` of
        `None` means the check was not run, which is different from zero.
    """
    dup_train = _flagged_count(integrity_summary, "exact_duplicate_within_train")
    dup_cross = _flagged_count(integrity_summary, "exact_duplicate_train_vs_test")
    all_black = _flagged_count(integrity_summary, "all_black_images")

    return pd.DataFrame([
        {
            "issue": "exact_duplicate_within_train",
            "count": dup_train,
            "decision": "keep" if dup_train == 0 else "keep_and_flag",
            "rationale": "Khong co duplicate nao bi phat hien; khong co gi de loai.",
        },
        {
            "issue": "exact_duplicate_train_vs_test",
            "count": dup_cross,
            "decision": "keep",
            "rationale": (
                "Khong co overlap. Official test set khong duoc thay doi trong "
                "moi truong hop."
            ),
        },
        {
            "issue": "all_black_images",
            "count": all_black,
            "decision": "keep" if all_black == 0 else "flag_for_manual_review",
            "rationale": (
                "Anh toan den se khong hoc duoc, nhung chi loai sau khi kiem tra "
                "thu cong."
            ),
        },
        {
            "issue": "near_duplicates",
            "count": None,
            "decision": "not_checked",
            "rationale": "Chua thuc hien: O(n^2) tren 60,000 anh, ngoai pham vi lab.",
        },
    ])


def plot_augmentation_grid(
    X: np.ndarray,
    picks: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str],
    transform,
    save_path: Optional[Path] = None,
):
    """Shows the same images before and after the training transform.

    Visual confirmation matters here: a rotation that is too large or a flip
    that changes the meaning of the object is easy to miss in code and obvious
    in a figure.

    Args:
        X: Images [N, H, W] as uint8.
        picks: Indices of the images to display, as returned by
            `sample_images_per_class`.
        y: Labels [N].
        class_names: Ordered class names.
        transform: A torchvision transform that accepts a uint8 HxW array.
        save_path: Optional destination for the PNG.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    flat = picks.reshape(-1)
    n_rows = picks.shape[0]
    n_cols = picks.shape[1] if picks.ndim > 1 else len(flat)
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(1.7 * n_cols, 3.4 * n_rows))
    axes = np.atleast_2d(axes)

    for row in range(n_rows):
        for col in range(n_cols):
            idx = picks[row, col] if picks.ndim > 1 else picks[col]
            augmented = transform(X[idx])
            if hasattr(augmented, "numpy"):
                augmented = augmented.numpy().squeeze()

            for offset, image, label in (
                (0, X[idx], "goc"),
                (1, augmented, "sau transform"),
            ):
                ax = axes[row * 2 + offset, col]
                ax.imshow(image, cmap="gray", vmin=0, vmax=1 if offset else 255)
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(label, fontsize=8)
                if offset == 0:
                    ax.set_title(class_names[int(y[idx])], fontsize=8)

    fig.suptitle("Anh truoc va sau augmentation (train only)", fontsize=11)
    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    return fig

"""Exploratory Data Analysis and data measurement module for Lab 01 (FashionMNIST)."""

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from .base import load_split, make_stratified_split, save_split
from ..utils.helpers import (
    CLASS_NAMES,
    DATA_DIR,
    EDA_DIR,
    PREPROC_VERSION,
    RANDOM_SEED,
    SPLIT_VERSION,
    VAL_SPLIT,
    _resolve_normalization,
)
from ..utils.plots import plot_eda_figures

# A pixel is treated as foreground above this intensity (on the [0, 1] scale).
FOREGROUND_THRESHOLD = 0.1


def to_scaled(images: np.ndarray) -> np.ndarray:
    """Converts uint8 images on [0, 255] to float32 on [0, 1]."""
    if images.dtype == np.uint8:
        return images.astype(np.float32) / 255.0
    if np.issubdtype(images.dtype, np.floating):
        return images.astype(np.float32)
    raise TypeError(f"Expected uint8 or float input, got {images.dtype}.")


def compute_normalization_stats(X: np.ndarray) -> Dict[str, float]:
    """Computes global mean and population standard deviation for Normalize."""
    scaled = to_scaled(X)
    flat = scaled.reshape(len(scaled), -1)

    return {
        "mean": float(flat.mean()),
        "std": float(flat.std(ddof=0)),
        "num_images": int(len(flat)),
        "num_pixels": int(flat.size),
        "per_image_std_mean": float(flat.std(axis=1, ddof=1).mean()),
        "per_image_std_ddof0_mean": float(flat.std(axis=1, ddof=0).mean()),
        "per_image_mean_mean": float(flat.mean(axis=1).mean()),
        "zero_pixel_fraction": float((flat == 0).mean()),
    }


def load_raw_arrays(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reads raw FashionMNIST IDX files directly, returning (X_train, y_train, X_test, y_test)."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    try:
        from torchvision import datasets
        datasets.FashionMNIST(root=str(root), train=True, download=True)
        datasets.FashionMNIST(root=str(root), train=False, download=True)
    except Exception:
        pass

    def _resolve(name: str) -> Path:
        for candidate in [root / name, root / "raw" / name, root / "FashionMNIST" / name, root / "FashionMNIST" / "raw" / name]:
            if candidate.exists():
                return candidate
        matches = sorted(root.glob(f"**/{name}"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"{name} not found under {root}.")

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


def get_structural_overview(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Core EDA Step 1: Structural overview of shapes, data types, pixel ranges, and memory."""
    return pd.DataFrame([
        {
            "Tập dữ liệu": part,
            "Số lượng ảnh": len(x),
            "Kích thước (HxW)": f"{x.shape[1]}x{x.shape[2]}",
            "Kênh màu": 1,
            "Kiểu dữ liệu (dtype)": str(x.dtype),
            "Dải điểm ảnh": f"[{x.min()}, {x.max()}]",
            "Dải nhãn (Labels)": f"[{y.min()}, {y.max()}]",
            "Số lớp": len(np.unique(y)),
            "Dung lượng RAM (MB)": round(x.nbytes / (1024 ** 2), 2),
        }
        for part, x, y in [("train", X_train, y_train), ("test", X_test, y_test)]
    ])


def check_data_integrity(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Core EDA Step 5: Data integrity verification (missing values, corruption, exact duplicates)."""
    train_has_nan = np.isnan(X_train).any() or np.isnan(y_train).any()
    test_has_nan = np.isnan(X_test).any() or np.isnan(y_test).any()
    pixels_in_bounds = (X_train.min() >= 0) and (X_train.max() <= 255) and (X_test.min() >= 0) and (X_test.max() <= 255)
    labels_in_bounds = (y_train.min() >= 0) and (y_train.max() <= 9) and (y_test.min() >= 0) and (y_test.max() <= 9)

    train_flat = np.ascontiguousarray(X_train.reshape(len(X_train), -1))
    test_flat = np.ascontiguousarray(X_test.reshape(len(X_test), -1))
    train_hash_list = [hashlib.md5(row.tobytes()).hexdigest() for row in train_flat]
    test_hash_list = [hashlib.md5(row.tobytes()).hexdigest() for row in test_flat]
    train_hashes, test_hashes = set(train_hash_list), set(test_hash_list)

    train_dups = len(train_hash_list) - len(train_hashes)
    test_dups = len(test_hash_list) - len(test_hashes)
    overlap = len(train_hashes.intersection(test_hashes))

    return pd.DataFrame([
        {"Hạng mục kiểm tra": "Giá trị thiếu (NaN / Null)", "Kết quả mong đợi": "Không có", "Thực tế": "Không có" if not (train_has_nan or test_has_nan) else "Có NaN", "Trạng thái": "PASSED" if not (train_has_nan or test_has_nan) else "FAILED"},
        {"Hạng mục kiểm tra": "Dải giá trị điểm ảnh [0, 255]", "Kết quả mong đợi": "[0, 255] uint8", "Thực tế": f"Train [{X_train.min()}, {X_train.max()}], Test [{X_test.min()}, {X_test.max()}]", "Trạng thái": "PASSED" if pixels_in_bounds else "FAILED"},
        {"Hạng mục kiểm tra": "Dải nhãn phân loại [0, 9]", "Kết quả mong đợi": "10 lớp [0, 9]", "Thực tế": f"Train [{y_train.min()}, {y_train.max()}], Test [{y_test.min()}, {y_test.max()}]", "Trạng thái": "PASSED" if labels_in_bounds else "FAILED"},
        {"Hạng mục kiểm tra": "Trùng lặp nội bộ tập Train", "Kết quả mong đợi": "0 mẫu trùng", "Thực tế": f"{train_dups} mẫu trùng", "Trạng thái": "PASSED" if train_dups == 0 else "WARNING"},
        {"Hạng mục kiểm tra": "Trùng lặp nội bộ tập Test", "Kết quả mong đợi": "0 mẫu trùng", "Thực tế": f"{test_dups} mẫu trùng", "Trạng thái": "PASSED" if test_dups == 0 else "WARNING"},
        {"Hạng mục kiểm tra": "Rò rỉ trùng lặp Train vs Test", "Kết quả mong đợi": "0 mẫu trùng", "Thực tế": f"{overlap} mẫu trùng", "Trạng thái": "PASSED" if overlap == 0 else "WARNING"},
    ])


def class_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Core EDA Step 3: Counts and percentages per class for train and val splits."""
    rows = []
    for split_name, y in (("train", y_train), ("val", y_val)):
        counts = np.bincount(y, minlength=len(class_names))
        total = int(counts.sum())
        for i, name in enumerate(class_names):
            rows.append({
                "split": split_name,
                "label": i,
                "class_name": name,
                "count": int(counts[i]),
                "percent": round(100.0 * counts[i] / total, 4) if total else 0.0,
                "split_total": total,
            })
    return pd.DataFrame(rows)


def spatial_features(
    X: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str],
    threshold: float = FOREGROUND_THRESHOLD,
) -> pd.DataFrame:
    """Core EDA Step 4: Per-image spatial descriptors derived from foreground mask."""
    scaled = to_scaled(X)
    mask = scaled > threshold

    foreground_ratio = mask.mean(axis=(1, 2))
    rows_any = mask.any(axis=2)
    cols_any = mask.any(axis=1)
    bbox_h = rows_any.sum(axis=1)
    bbox_w = cols_any.sum(axis=1)
    h, w = scaled.shape[1], scaled.shape[2]

    with np.errstate(divide="ignore", invalid="ignore"):
        aspect = np.where(bbox_h > 0, bbox_w / bbox_h, np.nan)

    return pd.DataFrame({
        "source_dataset": "original_train",
        "original_index": np.arange(len(scaled), dtype=np.int64),
        "split": "unassigned",
        "label": y.astype(np.int64),
        "class_name": [class_names[int(v)] for v in y],
        "foreground_threshold": threshold,
        "foreground_ratio": foreground_ratio.astype(np.float64),
        "bbox_height": bbox_h.astype(np.int64),
        "bbox_width": bbox_w.astype(np.int64),
        "bbox_aspect_ratio": aspect.astype(np.float64),
        "bbox_area_ratio": (bbox_h * bbox_w / (h * w)).astype(np.float64),
        "empty_foreground": (bbox_h == 0) | (bbox_w == 0),
    })


def class_similarity_matrix(
    mean_images: np.ndarray,
    class_names: Sequence[str],
    metric: str = "cosine",
) -> pd.DataFrame:
    """Similarity matrix between per-class mean images."""
    flat = mean_images.reshape(len(mean_images), -1).astype(np.float64)

    if metric == "cosine":
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Zero vector encountered in class mean.")
        sim = (flat / norms) @ (flat / norms).T
    elif metric == "correlation":
        centred = flat - flat.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centred, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Constant vector encountered in class mean.")
        sim = (centred / norms) @ (centred / norms).T
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return pd.DataFrame(sim, index=list(class_names), columns=list(class_names))


def most_similar_pairs(similarity: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Extracts highest-scoring off-diagonal class pairs from similarity matrix."""
    names = list(similarity.index)
    pairs = [
        {"class_a": names[i], "class_b": names[j], "similarity": round(float(similarity.iloc[i, j]), 6)}
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]
    return pd.DataFrame(pairs).sort_values("similarity", ascending=False).head(top_k).reset_index(drop=True)


def compute_pixel_statistics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_idx: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Core EDA Step 5: Global and per-class pixel intensity statistics."""
    scaled = to_scaled(X_train[train_idx])
    flat = scaled.reshape(len(scaled), -1)

    rows = []
    rows.append({
        "Phạm vi / Lớp": "Toàn bộ tập Train",
        "Số lượng mẫu": len(scaled),
        "Mean": round(float(flat.mean()), 4),
        "Std": round(float(flat.std(ddof=0)), 4),
        "Median": round(float(np.median(flat)), 4),
        "Tỷ lệ pixel 0": round(float((flat == 0).mean()), 4),
        "Min": round(float(flat.min()), 4),
        "Max": round(float(flat.max()), 4),
    })

    y_subset = y_train[train_idx]
    for c_idx, name in enumerate(class_names):
        c_mask = y_subset == c_idx
        c_flat = flat[c_mask]
        rows.append({
            "Phạm vi / Lớp": f"Lớp {c_idx} ({name})",
            "Số lượng mẫu": int(c_mask.sum()),
            "Mean": round(float(c_flat.mean()), 4),
            "Std": round(float(c_flat.std(ddof=0)), 4),
            "Median": round(float(np.median(c_flat)), 4),
            "Tỷ lệ pixel 0": round(float((c_flat == 0).mean()), 4),
            "Min": round(float(c_flat.min()), 4),
            "Max": round(float(c_flat.max()), 4),
        })

    return pd.DataFrame(rows)


def compute_handcrafted_features(
    X: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str],
    threshold: float = FOREGROUND_THRESHOLD,
) -> pd.DataFrame:
    """Core EDA Step 7: Summary distribution of handcrafted spatial and intensity features per class."""
    scaled = to_scaled(X)
    mask = scaled > threshold

    fg_ratios = mask.mean(axis=(1, 2))
    rows_any = mask.any(axis=2)
    cols_any = mask.any(axis=1)
    bbox_h = rows_any.sum(axis=1)
    bbox_w = cols_any.sum(axis=1)
    h, w = scaled.shape[1], scaled.shape[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        aspect = np.where(bbox_h > 0, bbox_w / bbox_h, np.nan)
    area_ratio = bbox_h * bbox_w / (h * w)

    diff_h = np.abs(np.diff(scaled, axis=1))[:, :, :-1]
    diff_w = np.abs(np.diff(scaled, axis=2))[:, :-1, :]
    edge_density = ((diff_h + diff_w) > 0.15).mean(axis=(1, 2))

    img_means = scaled.mean(axis=(1, 2))
    img_stds = scaled.std(axis=(1, 2))

    rows = []
    for c_idx, name in enumerate(class_names):
        c_mask = y == c_idx
        rows.append({
            "label": c_idx,
            "class_name": name,
            "mean_intensity": round(float(img_means[c_mask].mean()), 4),
            "std_intensity": round(float(img_stds[c_mask].mean()), 4),
            "foreground_ratio_mean": round(float(fg_ratios[c_mask].mean()), 4),
            "bbox_height_mean": round(float(bbox_h[c_mask].mean()), 2),
            "bbox_width_mean": round(float(bbox_w[c_mask].mean()), 2),
            "bbox_aspect_ratio_mean": round(float(np.nanmean(aspect[c_mask])), 4),
            "bbox_area_ratio_mean": round(float(area_ratio[c_mask].mean()), 4),
            "edge_density_mean": round(float(edge_density[c_mask].mean()), 4),
        })

    return pd.DataFrame(rows)


def lock_eda_fe_artifacts(
    X_train: np.ndarray,
    y_train: np.ndarray,
    out_dir: Path,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[Path, Path, Dict[str, float]]:
    """Generates and locks official EDA & Feature Engineering artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx = make_stratified_split(y_train, val_split=val_split, seed=seed)
    split_path = save_split(out_dir, train_idx, val_idx, targets=y_train, seed=seed, val_split=val_split)

    stats = compute_normalization_stats(X_train[train_idx])
    preproc_path = out_dir / "preprocessing_config.json"
    with open(preproc_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": PREPROC_VERSION,
            "dataset": "FashionMNIST",
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "normalization": {
                "mean": [round(stats["mean"], 6)],
                "std": [round(stats["std"], 6)],
                "formula": "Normalize(mean, std) where x in [0, 1]",
            },
            "diagnostics": {
                "per_image_std_mean": round(stats["per_image_std_mean"], 6),
                "zero_pixel_fraction": round(stats["zero_pixel_fraction"], 6),
            }
        }, f, indent=2)

    return split_path, preproc_path, stats


def run_eda_and_fe_pipeline(
    data_dir: Path = DATA_DIR,
    eda_dir: Path = EDA_DIR,
    val_split: float = VAL_SPLIT,
    seed: int = RANDOM_SEED,
    class_names: Sequence[str] = CLASS_NAMES,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Runs full Core EDA (9 canonical steps) & Feature Engineering pipeline and locks artifacts."""
    eda_dir = Path(eda_dir)
    tables_dir = eda_dir / "tables"
    figures_dir = eda_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Nạp dữ liệu thô
    X_train, y_train, X_test, y_test = load_raw_arrays(data_dir)

    # Step 1: Structural Overview
    overview_df = get_structural_overview(X_train, y_train, X_test, y_test, class_names)
    overview_df.to_csv(tables_dir / "structural_overview.csv", index=False, encoding="utf-8")
    overview_df.to_csv(eda_dir / "structural_overview.csv", index=False, encoding="utf-8")

    # Step 2: Data Integrity Check
    integrity_df = check_data_integrity(X_train, y_train, X_test, y_test)
    integrity_df.to_csv(tables_dir / "integrity_report.csv", index=False, encoding="utf-8")
    integrity_df.to_csv(eda_dir / "integrity_report.csv", index=False, encoding="utf-8")

    # Step 9 (Part 1): Khóa Stratified Split và tiền xử lý
    split_path, preproc_path, train_stats = lock_eda_fe_artifacts(X_train, y_train, eda_dir, val_split, seed)
    train_idx, val_idx = load_split(eda_dir, expected_pool_size=len(X_train))

    # Step 3: Class Distribution
    dist_df = class_distribution(y_train[train_idx], y_train[val_idx], class_names)
    dist_df.to_csv(tables_dir / "class_distribution.csv", index=False, encoding="utf-8")
    dist_df.to_csv(eda_dir / "class_distribution.csv", index=False, encoding="utf-8")

    # Step 5: Pixel Intensity Statistics
    pixel_stats_df = compute_pixel_statistics(X_train, y_train, train_idx, class_names)
    pixel_stats_df.to_csv(tables_dir / "pixel_statistics.csv", index=False, encoding="utf-8")
    pixel_stats_df.to_csv(eda_dir / "pixel_statistics.csv", index=False, encoding="utf-8")

    # Step 6: Spatial Analysis Descriptors
    spatial_df = spatial_features(X_train[train_idx], y_train[train_idx], class_names)
    spatial_df.to_csv(tables_dir / "spatial_features.csv", index=False, encoding="utf-8")
    spatial_df.to_csv(eda_dir / "spatial_features.csv", index=False, encoding="utf-8")

    # Step 7: Handcrafted Feature Analysis
    handcrafted_df = compute_handcrafted_features(X_train[train_idx], y_train[train_idx], class_names)
    handcrafted_df.to_csv(tables_dir / "handcrafted_features.csv", index=False, encoding="utf-8")
    handcrafted_df.to_csv(eda_dir / "handcrafted_features.csv", index=False, encoding="utf-8")

    # Step 8: Inter-Class Similarity Matrix
    mean_images = np.array([X_train[train_idx[y_train[train_idx] == c]].mean(axis=0) for c in range(len(class_names))])
    sim_df = class_similarity_matrix(mean_images, class_names, metric="cosine")
    sim_df.to_csv(tables_dir / "class_similarity_matrix.csv", encoding="utf-8")
    sim_df.to_csv(eda_dir / "class_similarity_matrix.csv", encoding="utf-8")
    similar_pairs_df = most_similar_pairs(sim_df, top_k=5)

    # Step 3, 4, 5, 6, 8: Trực quan hóa toàn bộ 7 figures vào eda_dir/figures
    figures = plot_eda_figures(
        X_train=X_train,
        y_train=y_train,
        train_idx=train_idx,
        val_idx=val_idx,
        class_names=class_names,
        save_dir=figures_dir,
    )

    if verbose:
        print("--- 1. TỔNG QUAN CẤU TRÚC DỮ LIỆU THÔ (STEP 1) ---")
        print(overview_df.to_string(index=False))
        print("\n--- 2. KIỂM TRA TÍNH TOÀN VẸN & CHỐNG RÒ RỈ (STEP 2) ---")
        print(integrity_df.to_string(index=False))
        print(f"\n[ARTIFACTS LOCKED] {split_path.name} & {preproc_path.name}")
        print("\n--- 5. THỐNG KÊ ĐIỂM ẢNH (STEP 5) ---")
        print(pixel_stats_df.head(4).to_string(index=False))
        print("\n--- 7. ĐẶC TRƯNG THỦ CÔNG THEO LỚP (STEP 7) ---")
        print(handcrafted_df.head(4).to_string(index=False))
        print("\n--- 8. CÁC CẶP LỚP CÓ ĐỘ TƯƠNG ĐỒNG COSINE CAO NHẤT (STEP 8) ---")
        print(similar_pairs_df.to_string(index=False))
        print(f"\n[HOÀN TẤT ĐẦY ĐỦ 9 BƯỚC EDA] Đã tạo 7 bảng CSV tại {tables_dir} và 7 biểu đồ PNG tại {figures_dir}")

    return {
        "overview_df": overview_df,
        "integrity_df": integrity_df,
        "dist_df": dist_df,
        "pixel_stats_df": pixel_stats_df,
        "spatial_df": spatial_df,
        "handcrafted_df": handcrafted_df,
        "sim_df": sim_df,
        "similar_pairs_df": similar_pairs_df,
        "split_path": split_path,
        "preproc_path": preproc_path,
        "train_stats": train_stats,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "num_test": len(X_test),
        "figures": figures,
    }


def verify_pipeline_handoff(
    train_loader,
    val_loader,
    eda_dir: Path = EDA_DIR,
) -> bool:
    """Verifies 4 core criteria before training handoff."""
    eda_dir = Path(eda_dir)
    split_meta_path = eda_dir / "split_metadata.json"
    preproc_path = eda_dir / "preprocessing_config.json"
    split_npz_path = eda_dir / "split_indices.npz"

    assert split_meta_path.exists() and preproc_path.exists() and split_npz_path.exists(), "[LỖI] Thiếu artifact bàn giao!"

    with open(split_meta_path, "r", encoding="utf-8") as f:
        split_meta = json.load(f)
    with open(preproc_path, "r", encoding="utf-8") as f:
        preproc_cfg = json.load(f)

    meta_split_version = split_meta.get("split_version") or split_meta.get("version")
    assert meta_split_version == SPLIT_VERSION, f"split_version mismatch: {meta_split_version} != {SPLIT_VERSION}"
    assert preproc_cfg.get("version") == PREPROC_VERSION, "preprocessing version mismatch"

    current_mean, _, current_source = _resolve_normalization()
    assert current_source.startswith("artifact:"), "NORMALIZATION_SOURCE must start with artifact:"
    expected_mean = preproc_cfg["normalization"]["mean"][0]
    assert round(current_mean[0], 4) == round(expected_mean, 4), f"Mean mismatch: {current_mean[0]} != {expected_mean}"
    expected_train_size = split_meta.get("train_size") or split_meta.get("train_count")
    expected_val_size = split_meta.get("val_size") or split_meta.get("val_count")
    assert len(train_loader.dataset) == expected_train_size, f"train_loader size mismatch: {len(train_loader.dataset)} != {expected_train_size}"
    assert len(val_loader.dataset) == expected_val_size, f"val_loader size mismatch: {len(val_loader.dataset)} != {expected_val_size}"

    print("[GUARD ASSERT PASSED] Toàn bộ 4 tiêu chí bàn giao EDA đã được xác thực 100%!")
    return True

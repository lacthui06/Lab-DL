"""Datasets package initialization: stratified split, features, preparation, and EDA."""

from .base import (
    load_split,
    make_stratified_split,
    save_split,
)
from .eda import (
    FOREGROUND_THRESHOLD,
    check_data_integrity,
    class_distribution,
    class_similarity_matrix,
    compute_handcrafted_features,
    compute_normalization_stats,
    compute_pixel_statistics,
    get_structural_overview,
    load_raw_arrays,
    lock_eda_fe_artifacts,
    most_similar_pairs,
    run_eda_and_fe_pipeline,
    spatial_features,
    to_scaled,
)
from .features import FEATURE_NAMES, extract_features
from .preparation import prepare_data

__all__ = [
    "load_split",
    "make_stratified_split",
    "save_split",
    "FEATURE_NAMES",
    "extract_features",
    "prepare_data",
    "FOREGROUND_THRESHOLD",
    "to_scaled",
    "compute_normalization_stats",
    "compute_pixel_statistics",
    "compute_handcrafted_features",
    "load_raw_arrays",
    "get_structural_overview",
    "check_data_integrity",
    "class_distribution",
    "spatial_features",
    "class_similarity_matrix",
    "most_similar_pairs",
    "lock_eda_fe_artifacts",
    "run_eda_and_fe_pipeline",
]

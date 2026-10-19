"""Datasets package initialization: base subsets, classification dataloaders, and EDA."""

from .base import (
    TransformedSubset,
    load_split,
    make_stratified_split,
    save_split,
)
from .classification import (
    get_dataloaders,
    get_transforms,
    resolve_normalization_stats,
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
    verify_pipeline_handoff,
)

__all__ = [
    # Base
    "TransformedSubset",
    "load_split",
    "make_stratified_split",
    "save_split",
    # Classification
    "get_dataloaders",
    "get_transforms",
    "resolve_normalization_stats",
    # EDA
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
    "verify_pipeline_handoff",
]

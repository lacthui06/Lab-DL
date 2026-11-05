"""Utility namespace; importing IO or costs does not load plotting or training."""
from importlib import import_module

_MODULE_EXPORTS = {
    ".constants": (
        "CHECKSUMS", "CLASSES", "DEFAULT_IMAGE_SIZE", "NUM_CHANNELS", "NUM_CLASSES", "RANDOM_SEED",
    ),
    ".delivery": (
        "audit_snapshot", "complete_delivery", "file_hash", "finalize_run",
        "iter_source_files", "resolve_data_dir", "resolve_run", "write_json",
    ),
    ".costs": ("cost_table", "validation_recommendation"),
    ".notebook_views": (
        "architecture_summary", "architecture_visual", "class_metrics", "context",
        "efficiency", "epoch_budget", "errors", "evidence", "fine_search", "learning_curves",
        "lr_search", "monitoring_status", "quality_cost", "roc_pr", "roundtrip",
    ),
    ".reporting": (
        "export_evaluation", "plot_learning_curves", "save_figure",
        "summarize_suite", "write_final_report",
    ),
}
_EXPORTS = {name: module for module, names in _MODULE_EXPORTS.items() for name in names}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Resolve and cache the requested export; unknown names remain normal attribute errors."""
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value

"""Training namespace; load dependencies only when a public export is requested."""
from importlib import import_module

_MODULE_EXPORTS = {
    "..utils.costs": ("cost_table", "validation_recommendation"),
    ".feature_cache": ("cached_features", "tensor_loader"),
    ".experiment": (
        "ROOT", "Tee", "fit", "main", "pass_epoch", "predict",
        "run_model", "run_suite", "search_finetune", "seed_all", "sync",
    ),
    ".tuning": (
        "DEPTHS", "OPTIMIZERS", "fine_candidates", "make_optimizer",
        "make_scheduler", "select_trial", "stopping_policy",
    ),
}
_EXPORTS = {name: module for module, names in _MODULE_EXPORTS.items() for name in names}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Preserve package-level imports without eagerly loading the training engine."""
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value

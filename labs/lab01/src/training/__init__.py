"""Training package initialization: trainer, optimizer factory, and experiment runners."""

from .experiments import (
    compare_optimizers_wallclock,
    plot_roc_curves,
    predict_proba,
    run_optuna_optimizer_study,
    run_threshold_scan,
)
from .trainer import (
    Trainer,
    build_optimizer,
    evaluate,
    sanity_overfit_check,
    train_model,
    train_one_epoch,
)

__all__ = [
    # Trainer & Loops
    "Trainer",
    "build_optimizer",
    "sanity_overfit_check",
    "train_one_epoch",
    "evaluate",
    "train_model",
    # Experiments
    "predict_proba",
    "run_optuna_optimizer_study",
    "compare_optimizers_wallclock",
    "plot_roc_curves",
    "run_threshold_scan",
]

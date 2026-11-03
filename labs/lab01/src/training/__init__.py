"""Training package initialization: trainer, optimizer factory, comparison, and tuning."""

from .comparison import CONFIGS, loaders, run_model
from .trainer import (
    Trainer,
    build_optimizer,
    sanity_overfit_check,
)
from .tuning import (
    OPTIMIZERS,
    plot_best_optimizer_histories,
    refit_winner,
    run_fixed_lr,
    run_tuning,
    summarize_optimizer_runs,
)

__all__ = [
    "Trainer",
    "build_optimizer",
    "sanity_overfit_check",
    "CONFIGS",
    "loaders",
    "run_model",
    "OPTIMIZERS",
    "plot_best_optimizer_histories",
    "run_fixed_lr",
    "run_tuning",
    "refit_winner",
    "summarize_optimizer_runs",
]

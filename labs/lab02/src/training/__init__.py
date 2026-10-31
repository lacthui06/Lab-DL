"""Training and experiments subpackage for Lab 02."""
from .costs import (
    cost_table,
    validation_recommendation,
)
from .experiment import (
    ROOT,
    Tee,
    cached_features,
    fit,
    main,
    pass_epoch,
    predict,
    run_model,
    run_suite,
    search_finetune,
    seed_all,
    sync,
    tensor_loader,
)
from .tuning import (
    DEPTHS,
    OPTIMIZERS,
    fine_candidates,
    make_optimizer,
    make_scheduler,
    select_trial,
    stopping_policy,
)

__all__ = [
    "cost_table",
    "validation_recommendation",
    "ROOT",
    "Tee",
    "cached_features",
    "fit",
    "main",
    "pass_epoch",
    "predict",
    "run_model",
    "run_suite",
    "search_finetune",
    "seed_all",
    "sync",
    "tensor_loader",
    "DEPTHS",
    "OPTIMIZERS",
    "fine_candidates",
    "make_optimizer",
    "make_scheduler",
    "select_trial",
    "stopping_policy",
]

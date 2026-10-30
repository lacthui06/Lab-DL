"""Compatibility alias for src.optuna_dense; implementation: src.training.optuna_dense."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module(".training.optuna_dense", __package__)

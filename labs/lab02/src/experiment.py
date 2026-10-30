"""Compatibility alias for src.experiment; implementation: src.training.trainer."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module(".training.trainer", __package__)

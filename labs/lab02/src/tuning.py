"""Compatibility alias for src.tuning; implementation: src.training.tuning."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module(".training.tuning", __package__)

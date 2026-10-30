"""Compatibility alias for src.standard_suite; implementation: src.training.standard_suite."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module(".training.standard_suite", __package__)

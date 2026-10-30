"""Compatibility alias for src.eda; implementation: src.datasets.eda."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module(".datasets.eda", __package__)

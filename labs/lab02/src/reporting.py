"""Compatibility alias for src.reporting; implementation: src.utils.plots."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module(".utils.plots", __package__)

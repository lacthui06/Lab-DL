"""Compatibility alias for src.merged_results; implementation: src.utils.merged_results."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module(".utils.merged_results", __package__)

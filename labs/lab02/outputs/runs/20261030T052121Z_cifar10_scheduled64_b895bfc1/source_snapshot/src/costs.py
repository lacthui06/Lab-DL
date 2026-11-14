"""Compatibility alias for src.costs; implementation: src.utils.costs."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module(".utils.costs", __package__)

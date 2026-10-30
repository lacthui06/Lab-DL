"""Compatibility alias for src.notebook_views; implementation: src.utils.notebook_views."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module(".utils.notebook_views", __package__)

"""Compatibility alias for src.delivery; implementation: src.utils.artifacts."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module(".utils.artifacts", __package__)

"""Compatibility alias for src.data; implementation: src.datasets.classification."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module(".datasets.classification", __package__)



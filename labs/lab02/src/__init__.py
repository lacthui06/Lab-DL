"""Lab 02 source package.

Subpackages are intentionally loaded on demand so lightweight utilities do
not import PyTorch or plotting dependencies unnecessarily.
"""

__all__ = ["datasets", "models", "training", "utils"]

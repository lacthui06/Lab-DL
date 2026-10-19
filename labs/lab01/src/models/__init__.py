"""Models package initialization and factory."""

from typing import Dict, Sequence
import torch.nn as nn
from .mlp import FashionMLP


def build_model(
    model_type: str = "mlp",
    num_classes: int = 10,
    in_features: int = 28 * 28,
    hidden_dims: Sequence[int] = (256, 128),
) -> nn.Module:
    """Factory function to instantiate classification models."""
    if model_type.lower() in ["mlp", "fashion_mlp"]:
        return FashionMLP(in_features=in_features, hidden_dims=hidden_dims, num_classes=num_classes)
    raise ValueError(f"Unsupported model architecture: {model_type}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Returns total, trainable, and non-trainable parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
    }


__all__ = ["FashionMLP", "build_model", "count_parameters"]

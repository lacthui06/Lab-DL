"""Model architecture definition for FashionMNIST Classification.

Provides a single pure PyTorch Multi-Layer Perceptron (FashionMLP)
using only 3 basic layer types: nn.Flatten, nn.Linear, nn.ReLU.
No BatchNorm, No Dropout - keeping the architecture transparent, simple, and fast.
"""

from typing import Tuple, Sequence, Optional
import torch
import torch.nn as nn


class FashionMLP(nn.Module):
    """Pure Multi-Layer Perceptron using only 3 basic layer types.
    Default architecture:
        Flatten -> Linear(784, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, 10).
    """
    def __init__(
        self,
        in_features: int = 28 * 28,
        hidden_dims: Sequence[int] = (256, 128),
        num_classes: int = 10,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if hidden_dim is not None:
            hidden_dims = (hidden_dim,)

        layers = [nn.Flatten()]
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(model_type: str = "mlp", num_classes: int = 10) -> nn.Module:
    """Factory function to instantiate the single FashionMLP model.
    
    Args:
        model_type: Model identifier (e.g. 'mlp', 'fashion_mlp').
        num_classes: Number of classification categories.
        
    Returns:
        Instantiated FashionMLP nn.Module.
    """
    return FashionMLP(num_classes=num_classes)


def count_parameters(model: nn.Module) -> dict:
    """Returns total, trainable, and non-trainable parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
    }

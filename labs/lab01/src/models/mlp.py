"""Multi-Layer Perceptron architecture for FashionMNIST classification."""

from typing import Optional, Sequence
import torch
import torch.nn as nn


class FashionMLP(nn.Module):
    """Pure Multi-Layer Perceptron using only 3 basic layer types: Flatten, Linear, ReLU.
    
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

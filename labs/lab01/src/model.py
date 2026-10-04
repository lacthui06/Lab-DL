"""Model architecture definitions for FashionMNIST Classification.
Provides:
1. BaselineMLP: Simple 2-layer perceptron for initial baseline.
2. TunedMLP: Deep perceptron with BatchNorm and Dropout regularization.
3. FashionCNN: Convolutional Neural Network with 2 Conv blocks and fully-connected head.
"""

from typing import Tuple
import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """Simple 2-layer Multi-Layer Perceptron (Baseline).
    Input: [B, 1, 28, 28] -> Output: [B, 10]
    """
    def __init__(self, in_features: int = 28 * 28, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TunedMLP(nn.Module):
    """Deep Multi-Layer Perceptron with Batch Normalization and Dropout.
    Demonstrates regularization and deeper feature extraction for tabular/flattened data.
    """
    def __init__(
        self,
        in_features: int = 28 * 28,
        hidden_dims: Tuple[int, int] = (256, 128),
        num_classes: int = 10,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FashionCNN(nn.Module):
    """Convolutional Neural Network tailored for 28x28 grayscale images.
    Preserves 2D spatial relationships and extracts translation-invariant features.
    Architecture:
      Conv1 (32 filters) -> BN -> ReLU -> MaxPool (14x14)
      Conv2 (64 filters) -> BN -> ReLU -> MaxPool (7x7)
      FC (128 units) -> BN -> ReLU -> Dropout(0.3) -> Linear (10)
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10, dropout_rate: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Conv Block 1: 1x28x28 -> 32x28x28 -> 32x14x14
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 2: 32x14x14 -> 64x14x14 -> 64x7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(model_type: str, num_classes: int = 10) -> nn.Module:
    """Factory function to build models by name.
    
    Args:
        model_type: 'baseline_mlp', 'tuned_mlp', or 'cnn'.
        num_classes: Number of classification targets.
        
    Returns:
        Instantiated nn.Module.
    """
    model_type = model_type.lower()
    if model_type in ["baseline_mlp", "baseline"]:
        return BaselineMLP(num_classes=num_classes)
    elif model_type in ["tuned_mlp", "mlp"]:
        return TunedMLP(num_classes=num_classes)
    elif model_type in ["cnn", "fashioncnn", "fashion_cnn"]:
        return FashionCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from 'baseline_mlp', 'tuned_mlp', 'cnn'.")


def count_parameters(model: nn.Module) -> dict:
    """Returns total, trainable, and non-trainable parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
    }

"""Model architecture tests."""

import sys
from pathlib import Path
import pytest
import torch

LAB01_DIR = Path(__file__).resolve().parent.parent
if str(LAB01_DIR) not in sys.path:
    sys.path.insert(0, str(LAB01_DIR))

from src.models import build_model, count_parameters


def test_model_forward_shapes():
    dummy_input = torch.randn(4, 1, 28, 28)
    model = build_model("mlp", num_classes=10)
    output = model(dummy_input)
    assert output.shape == (4, 10), f"Output shape must be (4, 10), got {output.shape}"
    assert torch.isfinite(output).all(), "Model output contains NaN or Inf"


def test_parameter_counts():
    model = build_model("mlp", num_classes=10)
    params = count_parameters(model)
    assert params["total"] > 0, "Model has 0 parameters"
    assert params["trainable"] > 0, "Model has 0 trainable parameters"
    assert params["non_trainable"] == 0, "FashionMLP should have no non-trainable parameters"

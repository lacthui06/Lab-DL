"""Helper utilities for Lab 01: reproducibility, config loading, and checkpointing."""

import json
import os
import platform
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across random, numpy, and torch for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Loads a YAML configuration file safely."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        raise ImportError("Package 'pyyaml' is required to load YAML config. Install with: pip install pyyaml")


def save_checkpoint(
    model: nn.Module,
    save_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves model state dictionary along with metadata."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, str(save_path))


def load_checkpoint(
    model: nn.Module,
    load_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Loads weights into model from checkpoint file."""
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {load_path}")
    checkpoint = torch.load(str(load_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return checkpoint.get("metadata", {})


def collect_environment_snapshot(save_path: Optional[Path] = None) -> Dict[str, Any]:
    """Records the interpreter and library versions that produced a run."""
    import sys

    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except ImportError:
        sklearn_version = None

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    snapshot = {
        "python_version": sys.version.split()[0],
        "python_version_full": sys.version,
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "platform": platform.platform(),
        "libraries": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn_version,
        },
        "device": device_str,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.is_dir():
            save_path = save_path / "environment_snapshot.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)

    return snapshot

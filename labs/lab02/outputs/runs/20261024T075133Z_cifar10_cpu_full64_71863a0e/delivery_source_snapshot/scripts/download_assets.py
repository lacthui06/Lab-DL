"""Download official dataset and official ImageNet weights into the project."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torchvision.datasets import CIFAR10
from src.models import SPECS

root = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(root / "data" / "pretrained"))
for train in (True, False):
    CIFAR10(str(root / "data"), train=train, download=True)
for name, (_, weights) in SPECS.items():
    print("Downloading", name, flush=True)
    weights.get_state_dict(progress=True, check_hash=True)
print("Official assets ready.")

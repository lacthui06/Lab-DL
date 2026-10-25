"""Load a saved model and classify one official test sample or local image."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
import torch
from torchvision.datasets import CIFAR10
from src.data import image_transform
from src.models import build_model
from src.reporting import CLASSES

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--image")
parser.add_argument("--sample-index", type=int, default=0)
args = parser.parse_args()
torch.set_num_threads(4)
saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
if saved["metadata"].get("dataset") != "CIFAR10":
    raise ValueError("Use the legacy source_snapshot for a FashionMNIST checkpoint")
name = saved["metadata"]["model"]
model = build_model(name, pretrained=False)
model.load_state_dict(saved["state_dict"])
model.eval()
if args.image:
    image = Image.open(args.image).convert("RGB")
    actual = None
else:
    raw = CIFAR10(str(Path(__file__).resolve().parents[1] / "data"), train=False)
    image, actual = raw[args.sample_index]
with torch.inference_mode():
    probabilities = model(image_transform(name, saved["metadata"]["image_size"])(image).unsqueeze(0)).softmax(1)[0]
prediction = int(probabilities.argmax())
print(json.dumps({"model": name, "predicted": CLASSES[prediction],
    "probability": float(probabilities[prediction]),
    "actual": CLASSES[actual] if actual is not None else None,
    "probabilities": dict(zip(CLASSES, probabilities.tolist()))}, indent=2))

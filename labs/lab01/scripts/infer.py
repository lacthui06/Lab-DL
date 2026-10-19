"""CLI Entry point for single/sample image inference (scripts/infer.py)."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.datasets.classification import get_dataloaders
from src.models import build_model
from src.utils.helpers import load_checkpoint, load_yaml_config


def main():
    parser = argparse.ArgumentParser(description="Infer label for sample test image.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file.")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/fashion_mlp_best.pt", help="Path to checkpoint.")
    parser.add_argument("--sample_idx", type=int, default=0, help="Test sample index to infer.")
    args = parser.parse_args()

    cfg = load_yaml_config(PROJECT_ROOT / args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataset sample
    data_dir = PROJECT_ROOT / cfg["paths"]["data_dir"]
    split_dir = PROJECT_ROOT / cfg["paths"]["eda_dir"]
    _, _, test_loader, class_names = get_dataloaders(data_dir=data_dir, split_dir=split_dir, batch_size=1)

    # Load model
    model = build_model(
        model_type=cfg["model"]["architecture"],
        num_classes=cfg["dataset"]["num_classes"],
        hidden_dims=cfg["model"].get("hidden_dims", (256, 128)),
    )
    load_checkpoint(model, PROJECT_ROOT / args.checkpoint, device=device)
    model.eval()

    # Get sample
    for idx, (img, target) in enumerate(test_loader):
        if idx == args.sample_idx:
            with torch.inference_mode():
                out = model(img.to(device))
                probs = torch.softmax(out, dim=1)
                conf, pred = torch.max(probs, dim=1)
                pred_label = class_names[pred.item()]
                true_label = class_names[target.item()]
                print(f"[INFERENCE SAMPLE #{args.sample_idx}]")
                print(f"  - Predicted: {pred_label} (Confidence: {conf.item() * 100:.2f}%)")
                print(f"  - Ground Truth: {true_label}")
                print(f"  - Match: {'CORRECT' if pred.item() == target.item() else 'WRONG'}")
            break


if __name__ == "__main__":
    main()

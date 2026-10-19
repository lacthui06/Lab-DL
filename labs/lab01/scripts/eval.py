"""CLI Entry point for model evaluation (scripts/eval.py)."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from src.datasets.classification import get_dataloaders
from src.models import build_model
from src.training.trainer import Trainer
from src.utils.helpers import load_checkpoint, load_yaml_config
from src.utils.metrics import build_per_class_metrics
from src.utils.plots import plot_confusion_matrix, plot_prediction_grid


def main():
    parser = argparse.ArgumentParser(description="Evaluate FashionMLP model.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file.")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/fashion_mlp_best.pt", help="Path to checkpoint.")
    args = parser.parse_args()

    cfg = load_yaml_config(PROJECT_ROOT / args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. DataLoaders
    data_dir = PROJECT_ROOT / cfg["paths"]["data_dir"]
    split_dir = PROJECT_ROOT / cfg["paths"]["eda_dir"]
    _, _, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        split_dir=split_dir,
        batch_size=cfg["dataset"].get("batch_size", 64),
        num_workers=cfg["dataset"].get("num_workers", 2),
    )

    # 2. Model & Checkpoint
    model = build_model(
        model_type=cfg["model"]["architecture"],
        num_classes=cfg["dataset"]["num_classes"],
        hidden_dims=cfg["model"].get("hidden_dims", (256, 128)),
    )
    ckpt_path = PROJECT_ROOT / args.checkpoint
    meta = load_checkpoint(model, ckpt_path, device=device)
    print(f"[LOADED CHECKPOINT] {ckpt_path.name} (Metadata: {meta})")

    # 3. Evaluate
    trainer = Trainer(model, nn.CrossEntropyLoss(), None, device)
    test_loss, test_acc, preds, targets = trainer.evaluate(test_loader)
    print(f"\n[TEST EVALUATION] Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}%")

    # 4. Metrics table & Plots
    predictions_dir = PROJECT_ROOT / cfg["paths"]["predictions_dir"]
    predictions_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = build_per_class_metrics(targets, preds, class_names)
    print("\n--- BẢNG ĐÁNH GIÁ CHI TIẾT TỪNG LỚP (PER-CLASS METRICS) ---")
    print(metrics_df.to_string(index=False))

    plot_confusion_matrix(
        targets, preds, class_names,
        title="FashionMLP: Ma Trận Nhầm Lẫn (Tập Test)",
        save_path=predictions_dir / "confusion_matrix.png",
    )
    plot_prediction_grid(
        model, test_loader, class_names, device,
        num_images=16,
        save_path=predictions_dir / "predictions_sample.png",
    )
    print(f"[DONE] Confusion matrix -> {predictions_dir / 'confusion_matrix.png'}")
    print(f"[DONE] Predictions sample -> {predictions_dir / 'predictions_sample.png'}")


if __name__ == "__main__":
    main()

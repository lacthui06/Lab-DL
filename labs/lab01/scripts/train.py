"""CLI Entry point for model training (scripts/train.py)."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from src.datasets.classification import get_dataloaders
from src.models import build_model
from src.training.trainer import Trainer, build_optimizer
from src.utils.helpers import load_yaml_config, set_seed
from src.utils.plots import plot_loss_and_acc_curves


def main():
    parser = argparse.ArgumentParser(description="Train FashionMLP model.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file.")
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / args.config
    cfg = load_yaml_config(cfg_path)

    # 1. Environment & Seeds
    set_seed(cfg.get("project", {}).get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Computing device: {device}")

    # 2. DataLoaders
    data_dir = PROJECT_ROOT / cfg["paths"]["data_dir"]
    split_dir = PROJECT_ROOT / cfg["paths"]["eda_dir"]
    batch_size = cfg["dataset"].get("batch_size", 64)
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        split_dir=split_dir,
        batch_size=batch_size,
        num_workers=cfg["dataset"].get("num_workers", 2),
    )

    # 3. Model & Optimizer
    model = build_model(
        model_type=cfg["model"]["architecture"],
        num_classes=cfg["dataset"]["num_classes"],
        hidden_dims=cfg["model"].get("hidden_dims", (256, 128)),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        optimizer_name=cfg["training"]["optimizer"],
        model=model,
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 0.0),
    )

    epochs = cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    trainer = Trainer(model, criterion, optimizer, device, scheduler=scheduler)

    # 4. Fit & Save
    checkpoints_dir = PROJECT_ROOT / cfg["paths"]["checkpoints_dir"]
    predictions_dir = PROJECT_ROOT / cfg["paths"]["predictions_dir"]
    logs_dir = PROJECT_ROOT / cfg["paths"]["logs_dir"]
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[START TRAINING] Model: {cfg['model']['architecture']} | Epochs: {epochs} | LR: {cfg['training']['learning_rate']}")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_path=checkpoints_dir / "fashion_mlp_best.pt",
        verbose=True,
    )

    # 5. Save logs & Curves
    with open(logs_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_loss_and_acc_curves(
        history,
        title=f"FashionMLP ({cfg['training']['optimizer']})",
        save_path=predictions_dir / "learning_curves.png",
    )
    print(f"[DONE] Checkpoint saved -> {checkpoints_dir / 'fashion_mlp_best.pt'}")
    print(f"[DONE] Learning curves saved -> {predictions_dir / 'learning_curves.png'}")


if __name__ == "__main__":
    main()

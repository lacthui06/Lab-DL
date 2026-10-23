"""CLI training script for Lab 03."""
import argparse
import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import yaml
import torch
from labs.lab03.src.datasets.data_loader import load_imdb_dataset, tokenize_dataset, run_text_eda
from labs.lab03.src.models.classifier import load_classification_model
from labs.lab03.src.training.trainer import build_trainer, run_one_batch_sanity_check
from labs.lab03.src.utils.metrics import save_environment_snapshot


def parse_args():
    parser = argparse.ArgumentParser(description="Lab 03 Training Script")
    parser.add_argument("--config", type=str, default="labs/lab03/config/config.yaml")
    parser.add_argument("--exp_config", type=str, default="labs/lab03/config/experiments.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.exp_config, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)

    output_dir = cfg["paths"]["output_dir"]
    save_environment_snapshot(os.path.join(cfg["paths"]["log_dir"], "environment_snapshot.json"))

    print("Step 1: Loading IMDB dataset...")
    raw_datasets = load_imdb_dataset(cfg)
    
    print("Step 2: Running text EDA...")
    eda_stats = run_text_eda(raw_datasets, os.path.join(output_dir, "eda"))
    print(f"EDA Summary: Train={eda_stats['train']['num_samples']}, Val={eda_stats['validation']['num_samples']}")

    print("Step 3: Loading tokenizer and model...")
    model_name = exp_cfg["finetuning"]["pretrained_model"]
    model, tokenizer = load_classification_model(
        model_name=model_name,
        num_labels=cfg["dataset"]["num_labels"],
        id2label={int(k): v for k, v in cfg["dataset"]["id2label"].items()},
        label2id=cfg["dataset"]["label2id"]
    )

    print("Step 4: Running 1-batch sanity check...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sanity_loss = run_one_batch_sanity_check(model, tokenizer, device=device, steps=25)
    print(f"Sanity Check Final Loss: {sanity_loss:.4f}")

    print("Step 5: Tokenizing datasets...")
    tokenized_datasets = tokenize_dataset(raw_datasets, tokenizer, max_length=256)

    print("Step 6: Building Trainer and launching finetuning...")
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized_datasets=tokenized_datasets,
        exp_config=exp_cfg["finetuning"],
        output_dir=output_dir
    )

    train_result = trainer.train()
    print("Training finished successfully!")
    print(train_result.metrics)

    best_checkpoint = os.path.join(cfg["paths"]["checkpoint_dir"], "finetuned_distilbert")
    trainer.save_model(best_checkpoint)
    tokenizer.save_pretrained(best_checkpoint)
    print(f"Model saved to {best_checkpoint}")


if __name__ == "__main__":
    main()

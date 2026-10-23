"""CLI evaluation script for Lab 03."""
import argparse
import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import json
import yaml
import numpy as np
import torch
from transformers import DataCollatorWithPadding, Trainer
from labs.lab03.src.datasets.data_loader import load_imdb_dataset, tokenize_dataset
from labs.lab03.src.models.classifier import load_classification_model
from labs.lab03.src.utils.metrics import get_compute_metrics_fn, plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Lab 03 Evaluation Script")
    parser.add_argument("--config", type=str, default="labs/lab03/config/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    checkpoint = args.checkpoint if args.checkpoint else os.path.join(cfg["paths"]["checkpoint_dir"], "finetuned_distilbert")
    if not os.path.exists(checkpoint):
        checkpoint = cfg["model"]["pretrained_model_name"]
        print(f"Fine-tuned checkpoint not found. Evaluating baseline model: {checkpoint}")

    print(f"Loading model from {checkpoint}...")
    model, tokenizer = load_classification_model(
        model_name=checkpoint,
        num_labels=cfg["dataset"]["num_labels"],
        id2label={int(k): v for k, v in cfg["dataset"]["id2label"].items()},
        label2id=cfg["dataset"]["label2id"]
    )

    print("Loading test dataset...")
    raw_datasets = load_imdb_dataset(cfg)
    tokenized_datasets = tokenize_dataset(raw_datasets, tokenizer, max_length=256)

    trainer = Trainer(
        model=model,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=get_compute_metrics_fn()
    )

    print("Running evaluation on test split...")
    eval_results = trainer.evaluate(tokenized_datasets["test"])
    print("\n--- Test Evaluation Results ---")
    for k, v in eval_results.items():
        print(f"{k}: {v}")

    # Generate Confusion Matrix
    preds = trainer.predict(tokenized_datasets["test"])
    pred_labels = np.argmax(preds.predictions, axis=-1)
    true_labels = preds.label_ids

    cm_path = os.path.join(cfg["paths"]["predictions_dir"], "confusion_matrix.png")
    plot_confusion_matrix(
        y_true=true_labels,
        y_pred=pred_labels,
        class_names=["NEGATIVE", "POSITIVE"],
        output_path=cm_path,
        title="IMDB Test Set Confusion Matrix"
    )
    print(f"Saved confusion matrix plot to {cm_path}")

    # Save metrics JSON
    metrics_path = os.path.join(cfg["paths"]["output_dir"], "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved evaluation metrics to {metrics_path}")


if __name__ == "__main__":
    main()

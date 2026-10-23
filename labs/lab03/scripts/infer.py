"""CLI inference script for Lab 03 Sentiment Analysis."""
import argparse
import json
import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import torch
import yaml
from transformers import pipeline
from labs.lab03.src.models.classifier import load_classification_model, predict_sentiment_manual


def parse_args():
    parser = argparse.ArgumentParser(description="Sentiment Analysis Inference CLI")
    parser.add_argument("--text", type=str, required=True, help="Input text sequence")
    parser.add_argument("--config", type=str, default="labs/lab03/config/config.yaml", help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to fine-tuned model checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference on device: {device}")

    model_path = args.checkpoint if args.checkpoint else cfg["model"]["pretrained_model_name"]
    model, tokenizer = load_classification_model(
        model_name=model_path,
        num_labels=cfg["dataset"]["num_labels"],
        id2label={int(k): v for k, v in cfg["dataset"]["id2label"].items()},
        label2id=cfg["dataset"]["label2id"]
    )

    result = predict_sentiment_manual(args.text, model, tokenizer, device=device)
    print("\n--- Inference Result ---")
    print(f"Input Text: {result['text']}")
    print(f"Predicted Label: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities: {result['probabilities']}")
    print(f"Tokens: {result['tokens'][:15]}...")


if __name__ == "__main__":
    main()

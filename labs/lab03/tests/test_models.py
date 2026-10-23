"""Test model initialization and inference for Lab 03."""
import pytest
import torch
from labs.lab03.src.models.classifier import load_classification_model, predict_sentiment_manual


def test_model_loading_and_forward():
    model_name = "distilbert-base-uncased"
    model, tokenizer = load_classification_model(
        model_name,
        num_labels=2,
        id2label={0: "NEG", 1: "POS"},
        label2id={"NEG": 0, "POS": 1}
    )

    sample_text = "Testing Hugging Face transformer model."
    res = predict_sentiment_manual(sample_text, model, tokenizer, device="cpu")

    assert "predicted_label" in res
    assert "confidence" in res
    assert res["predicted_label"] in ["NEG", "POS"]
    assert len(res["probabilities"]) == 2

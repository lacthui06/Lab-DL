"""Test training sanity check and metric calculation for Lab 03."""
import pytest
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from labs.lab03.src.training.trainer import run_one_batch_sanity_check
from labs.lab03.src.utils.metrics import get_compute_metrics_fn


def test_compute_metrics():
    compute_fn = get_compute_metrics_fn()
    dummy_logits = torch.tensor([[2.0, -1.0], [-1.5, 3.0]]).numpy()
    dummy_labels = torch.tensor([0, 1]).numpy()

    metrics = compute_fn((dummy_logits, dummy_labels))
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_one_batch_sanity_overfit():
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    final_loss = run_one_batch_sanity_check(model, tokenizer, device=device, steps=20, lr=1e-4)

    # Sanity loss overfit should decrease significantly
    assert final_loss < 0.25

"""Test datasets loading and tokenization for Lab 03."""
import pytest
from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset
from labs.lab03.src.datasets.data_loader import clean_text, tokenize_dataset


def test_clean_text():
    raw = "<br /><p>This was <b>awesome</b>!</p>   "
    cleaned = clean_text(raw)
    assert "<" not in cleaned
    assert ">" not in cleaned
    assert "This was awesome!" == cleaned


def test_tokenization_output():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    raw_ds = DatasetDict({
        "train": Dataset.from_dict({
            "text": ["A great movie!", "A terrible movie."],
            "label": [1, 0]
        })
    })

    tokenized = tokenize_dataset(raw_ds, tokenizer, max_length=64)
    assert "input_ids" in tokenized["train"].column_names
    assert "attention_mask" in tokenized["train"].column_names
    assert len(tokenized["train"]["input_ids"]) == 2

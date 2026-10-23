"""Datasets and EDA utilities for Lab 03."""
from typing import Dict, Any, Tuple
import os
import re
import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def clean_text(text: str) -> str:
    """Basic text cleaning: remove HTML tags and multiple spaces."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_imdb_dataset(config: Dict[str, Any]) -> DatasetDict:
    """Load IMDB dataset with optional subsetting for fast local testing."""
    dataset_name = config.get("dataset", {}).get("name", "stanfordnlp/imdb")
    dataset = load_dataset(dataset_name)
    
    train_size = config.get("dataset", {}).get("train_sample_size")
    test_size = config.get("dataset", {}).get("test_sample_size")
    seed = config.get("seed", 42)

    if train_size and train_size < len(dataset["train"]):
        train_ds = dataset["train"].shuffle(seed=seed).select(range(train_size))
    else:
        train_ds = dataset["train"]

    if test_size and test_size < len(dataset["test"]):
        test_ds = dataset["test"].shuffle(seed=seed).select(range(test_size))
    else:
        test_ds = dataset["test"]

    # Split train further into train and validation
    eval_size = config.get("dataset", {}).get("eval_sample_size", 500)
    split_ds = train_ds.train_test_split(test_size=eval_size, seed=seed)

    return DatasetDict({
        "train": split_ds["train"],
        "validation": split_ds["test"],
        "test": test_ds
    })


def run_text_eda(dataset_dict: DatasetDict, output_dir: str) -> Dict[str, Any]:
    """Calculate character length, word count distributions, and label balance."""
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    stats = {}
    for split_name, split_data in dataset_dict.items():
        df = pd.DataFrame(split_data)
        df["char_len"] = df["text"].apply(len)
        df["word_count"] = df["text"].apply(lambda x: len(x.split()))
        
        summary = {
            "num_samples": len(df),
            "label_distribution": df["label"].value_counts().to_dict(),
            "word_count_mean": float(df["word_count"].mean()),
            "word_count_std": float(df["word_count"].std()),
            "word_count_median": float(df["word_count"].median()),
            "word_count_max": int(df["word_count"].max()),
        }
        stats[split_name] = summary
        
        # Save table summary
        df[["char_len", "word_count", "label"]].describe().to_csv(
            os.path.join(output_dir, "tables", f"eda_{split_name}_stats.csv")
        )

    return stats


def tokenize_dataset(
    dataset_dict: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 256
) -> DatasetDict:
    """Preprocess and tokenize dataset using Hugging Face map."""
    def preprocess_function(examples):
        cleaned_texts = [clean_text(t) for t in examples["text"]]
        return tokenizer(
            cleaned_texts,
            truncation=True,
            max_length=max_length,
            padding=False  # Leave dynamic padding to DataCollatorWithPadding
        )

    return dataset_dict.map(preprocess_function, batched=True)

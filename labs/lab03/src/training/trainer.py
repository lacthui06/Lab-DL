"""Training orchestration for Lab 03."""
from typing import Dict, Any, Tuple
import os
import torch
import torch.nn as nn
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase
)
from datasets import DatasetDict
from ..utils.metrics import get_compute_metrics_fn


def run_one_batch_sanity_check(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: str = "cpu",
    steps: int = 30,
    lr: float = 5e-5
) -> float:
    """Overfit on a tiny mini-batch (8 samples) to verify pipeline learnability."""
    model.to(device)
    model.train()

    dummy_texts = [
        "This film was brilliantly crafted and remarkably uplifting.",
        "Worst movie experience ever, complete waste of money.",
        "Superb acting performance with astonishing visuals.",
        "Dull, predictable, and totally uninspiring garbage.",
        "I adored every single minute of this magnificent movie.",
        "Painful to watch, terrible writing and horrific acting.",
        "Masterpiece of modern cinema, truly unforgettable.",
        "Disastrous plot and utterly cardboard characters."
    ]
    dummy_labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.long, device=device)

    inputs = tokenizer(dummy_texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["labels"] = dummy_labels

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    final_loss = 0.0
    for _ in range(steps):
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    return final_loss


def build_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    tokenized_datasets: DatasetDict,
    exp_config: Dict[str, Any],
    output_dir: str
) -> Trainer:
    """Build and configure standard Hugging Face Trainer."""
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        learning_rate=float(exp_config.get("learning_rate", 2e-5)),
        num_train_epochs=exp_config.get("num_train_epochs", 2),
        per_device_train_batch_size=exp_config.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=exp_config.get("per_device_eval_batch_size", 32),
        warmup_steps=exp_config.get("warmup_steps", 20),
        weight_decay=exp_config.get("weight_decay", 0.01),
        logging_steps=exp_config.get("logging_steps", 25),
        eval_strategy=exp_config.get("eval_strategy", "epoch"),
        save_strategy=exp_config.get("save_strategy", "epoch"),
        load_best_model_at_end=exp_config.get("load_best_model_at_end", True),
        metric_for_best_model=exp_config.get("metric_for_best_model", "accuracy"),
        fp16=exp_config.get("fp16", torch.cuda.is_available()),
        report_to="none",
        seed=exp_config.get("seed", 42)
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=get_compute_metrics_fn(),
    )

    return trainer

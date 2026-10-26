"""Trainer có đủ epoch, log theo step và checkpoint chọn chỉ trên validation."""
import math
import warnings
import torch
from copy import copy
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, TrainerCallback
from .metrics import compute_metrics

def make_trainer(model, tokenizer, ds, config, run_path):
    strategy = config.get("eval_strategy", "epoch")
    if strategy not in {"steps", "epoch"}:
        raise ValueError("eval_strategy must be steps or epoch")
    args = TrainingArguments(
        output_dir=str(run_path / "checkpoints"),
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        weight_decay=config["weight_decay"], warmup_ratio=config["warmup_ratio"],
        eval_strategy=strategy, save_strategy=strategy,
        eval_steps=config.get("eval_steps", 100), save_steps=config.get("eval_steps", 100),
        logging_strategy="steps" if strategy == "steps" else "epoch",
        logging_steps=config.get("logging_steps", 25), logging_nan_inf_filter=False,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        dataloader_num_workers=config.get("dataloader_num_workers", 0),
        dataloader_pin_memory=torch.cuda.is_available(),
        load_best_model_at_end=True, metric_for_best_model=config.get("metric_for_best_model", "accuracy"),
        greater_is_better=True, save_total_limit=2,
        save_only_model=config.get("save_only_model", False),
        fp16=bool(config["fp16"] and torch.cuda.is_available()),
        seed=config["seed"], data_seed=config["seed"], report_to="none")
    trainer = Trainer(model=model, args=args, processing_class=tokenizer,
                   train_dataset=ds["train"], eval_dataset=ds["validation"],
                   data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if args.fp16 else None), compute_metrics=compute_metrics)
    class TrainingMetrics(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            # Dừng khi loss không hữu hạn thay vì che NaN trong trung bình log.
            for name in ("loss", "eval_loss", "train_loss"):
                if logs and name in logs and not math.isfinite(float(logs[name])):
                    raise FloatingPointError(f"Nonfinite {name} at step {state.global_step}")

            if logs and logs.get("grad_norm") is not None and not math.isfinite(float(logs["grad_norm"])):
                if not args.fp16:
                    raise FloatingPointError(f"Nonfinite grad_norm at step {state.global_step}")
                # AMP có thể bỏ qua update khi scale quá lớn; giữ cờ minh bạch,
                # không serialize Infinity thành một số JSON không chuẩn.
                logs.update(grad_norm=None, amp_nonfinite_grad_norm=True)
                if state.log_history:
                    state.log_history[-1].update(grad_norm=None, amp_nonfinite_grad_norm=True)
                warnings.warn("AMP nonfinite gradient norm recorded; inspect skipped optimizer steps", RuntimeWarning)

        def on_step_end(self, args, state, control, **kwargs):
            if trainer.accelerator.optimizer_step_was_skipped:
                trainer.amp_skipped_optimizer_steps = getattr(trainer, "amp_skipped_optimizer_steps", 0) + 1

        def on_epoch_end(self, args, state, control, **kwargs):
            # evaluate() đặt lại cờ TrainerControl; bảo toàn lịch validation/save.
            scheduled = copy(control)
            trainer.evaluate(ds["train"], metric_key_prefix="train")
            # Đo validation/save cả cuối epoch, kể cả step cuối không chia hết 100.
            # Không early stopping: mọi cấu hình phải chạy đủ ngân sách epoch.
            scheduled.should_evaluate = True
            scheduled.should_save = True
            return scheduled
    trainer.add_callback(TrainingMetrics())
    return trainer

def sanity_overfit(model, tokenizer, steps=30, lr=1e-3):
    """Test-only gate; caller supplies a disposable model, never the training model."""
    texts = ["I loved this excellent movie", "A terrible boring film"] * 4
    device = next(model.parameters()).device
    batch = tokenizer(texts, padding=True, return_tensors="pt").to(device)
    labels = torch.tensor([1, 0] * 4, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = model(**batch, labels=labels).loss
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return float(model(**batch, labels=labels).loss)

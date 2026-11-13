"""Trainer với ngân sách epoch, early stopping và checkpoint chọn trên validation."""
import math
import warnings
import time
import inspect
import torch
from copy import copy
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, TrainerCallback, EarlyStoppingCallback
from .metrics import compute_metrics
from ..utils.runs import save_json

def make_trainer(model, tokenizer, ds, config, run_path):
    stopping = config.get('early_stopping', {})
    if stopping.get('enabled', False):
        patience, delta = stopping.get('patience', 2), stopping.get('min_delta', .001)
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError('Early-stopping patience must be a positive integer')
        if isinstance(delta, bool) or not isinstance(delta, (int,float)) or not math.isfinite(delta) or delta < 0:
            raise ValueError('Early-stopping min_delta must be finite and nonnegative')
        if config.get('metric_for_best_model', '').removeprefix('eval_') != 'loss':
            raise ValueError('Early stopping is predeclared on validation loss')
    use_cpu = config.get("device") == "cpu"
    strategy = config.get("eval_strategy", "epoch")
    if strategy not in {"steps", "epoch"}:
        raise ValueError("eval_strategy must be steps or epoch")
    sig = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = dict(
        output_dir=str(run_path / "checkpoints"),
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        weight_decay=config["weight_decay"],
        eval_strategy=strategy,
        save_strategy=strategy,
        eval_steps=config.get("eval_steps", 100),
        save_steps=config.get("eval_steps", 100),
        logging_strategy="steps" if strategy == "steps" else "epoch",
        logging_steps=config.get("logging_steps", 25),
        logging_nan_inf_filter=False,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        dataloader_num_workers=config.get("dataloader_num_workers", 0),
        dataloader_pin_memory=bool(torch.cuda.is_available() and not use_cpu),
        use_cpu=use_cpu,
        load_best_model_at_end=True,
        metric_for_best_model=config.get("metric_for_best_model", "accuracy"),
        greater_is_better=config.get("metric_for_best_model", "accuracy").removeprefix("eval_") != "loss",
        save_total_limit=2,
        disable_tqdm=config.get("epoch_console_summary", False),
        save_only_model=config.get("save_only_model", False),
        fp16=bool(config["fp16"] and torch.cuda.is_available() and not use_cpu),
        seed=config["seed"],
        data_seed=config["seed"],
        report_to="none"
    )
    if "warmup_ratio" in sig:
        kwargs["warmup_ratio"] = config.get("warmup_ratio", 0.0)
        kwargs["warmup_steps"] = config.get("warmup_steps", 0)
    else:
        ratio = config.get("warmup_ratio", 0.0)
        steps = config.get("warmup_steps", 0)
        kwargs["warmup_steps"] = ratio if ratio > 0 else steps
    args = TrainingArguments(**kwargs)
    trainer = Trainer(model=model, args=args, processing_class=tokenizer,
                   train_dataset=ds["train"], eval_dataset=ds["validation"],
                   data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if args.fp16 else None), compute_metrics=compute_metrics)
    class TrainingMetrics(TrainerCallback):
        def __init__(self):
            self.epoch_rows = []
            self.train_epoch_metrics = None
            self.epoch_started = None

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.epoch_started = time.perf_counter()
            self.train_epoch_metrics = None

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            # evaluate(train) cũng phát callback; chỉ ghép với validation cùng epoch.
            if not metrics or "eval_loss" not in metrics or self.train_epoch_metrics is None:
                return
            epoch = float(state.epoch)
            if not math.isclose(epoch, round(epoch), abs_tol=1e-8):
                return
            if any(r["epoch"] == round(epoch) for r in self.epoch_rows):
                return
            train = self.train_epoch_metrics
            row = {"epoch": int(round(epoch)), "step": state.global_step,
                "train_loss_eval": train["train_loss"], "train_accuracy": train["train_accuracy"],
                "validation_loss": metrics["eval_loss"], "validation_accuracy": metrics["eval_accuracy"],
                "validation_macro_f1": metrics["eval_macro_f1"],
                "learning_rate": trainer.lr_scheduler.get_last_lr()[0] if trainer.lr_scheduler else args.learning_rate,
                "epoch_runtime_seconds": time.perf_counter() - self.epoch_started if self.epoch_started else None}
            self.epoch_rows.append(row)
            save_json(run_path / "epoch_history.json", self.epoch_rows)
            if config.get("epoch_console_summary", False):
                print(f"Epoch {row['epoch']:02d}/{int(args.num_train_epochs):02d} | "
                      f"Train Loss (eval): {row['train_loss_eval']:.4f} - Acc: {row['train_accuracy']:.2%} | "
                      f"Val Loss: {row['validation_loss']:.4f} - Acc: {row['validation_accuracy']:.2%} | "
                      f"Val Macro-F1: {row['validation_macro_f1']:.4f} | LR: {row['learning_rate']:.7f}", flush=True)

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
            self.train_epoch_metrics = trainer.evaluate(ds["train"], metric_key_prefix="train")
            # Đo validation/save cả cuối epoch, kể cả step cuối không chia hết 100.
            # Validation tiếp theo quyết định early stopping nếu profile bật cơ chế đó.
            scheduled.should_evaluate = True
            scheduled.should_save = True
            return scheduled
    if config.get("epoch_console_summary", False):
        from transformers.trainer_callback import PrinterCallback
        trainer.remove_callback(PrinterCallback)
    if config.get('record_resources', False):
        from ..performance import EpochResources
        trainer.resource_callback = EpochResources(trainer, len(ds['train']), run_path)
        trainer.add_callback(trainer.resource_callback)
    trainer.add_callback(TrainingMetrics())
    if stopping.get('enabled', False):
        class ValidationEarlyStopping(EarlyStoppingCallback):
            def on_evaluate(self, args, state, control, metrics, **kwargs):
                # Không tính evaluate(train) thành một lần validation không cải thiện.
                if 'eval_loss' in metrics:
                    return super().on_evaluate(args, state, control, metrics, **kwargs)
        trainer.add_callback(ValidationEarlyStopping(patience, delta))
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

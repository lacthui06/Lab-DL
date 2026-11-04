"""Shared notebook/CLI workflow, so preprocessing and training never diverge."""
import json
import re
from pathlib import Path
import numpy as np
import torch
from transformers import set_seed
from .datasets.loader import load_data, tokenize_data
from .models.classifier import load_model, manual_predict, compare_inference, verify_saved_model
from .training.trainer import make_trainer
from .training.metrics import compute_metrics
from .training.thresholds import export_threshold_analysis, binary_metrics_at_threshold
from .decision import checkpoint_identity
from .utils.runs import ROOT, run_context, save_json
from .utils.plots import plot_lengths, plot_learning, plot_epoch_learning, plot_step_learning, plot_confusion, plot_metrics, plot_samples


def read_config(path=None):
    return json.loads(Path(path or ROOT / "config/config.json").read_text(encoding="utf-8"))

def seed_all(seed):
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def exercise1(config=None):
    config = config or read_config()
    seed_all(config["seed"])
    cfg = config["exercise1"]
    with run_context("exercise1", config) as out:
        model, tokenizer = load_model(cfg["model_name"], cfg["revision"])
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        high, low = compare_inference(model, tokenizer, cfg["sample_sentences"])
        encoded = tokenizer(cfg["sample_sentences"][0])
        breakdown = {"text": cfg["sample_sentences"][0], "tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"]),
                     "input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
        save_json(out / "tokenizer_breakdown.json", breakdown)
        save_json(out / "predictions.json", {"pipeline": high, "manual": low})
        print(json.dumps(breakdown, indent=2)); print(json.dumps(low, indent=2))
    return out

def prepare_data(config=None):
    config = config or read_config()
    cfg = config["exercise2"]
    ds = load_data(cfg)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], revision=cfg["model_revision"])
    return ds, tokenizer, tokenize_data(ds, tokenizer, cfg["max_length"])

def choose_samples(texts, labels, probs):
    preds = probs.argmax(-1)
    selected = []
    # Ưu tiên lỗi tự tin cao cùng ví dụ đúng của cả hai lớp.
    for correct in (True, False):
        indices = [i for i in range(len(labels)) if bool(preds[i] == labels[i]) == correct]
        indices.sort(key=lambda i: float(probs[i].max()), reverse=True)
        for label in (0, 1):
            chosen = [i for i in indices if labels[i] == label][:2]
            for i in chosen:
                selected.append({"index": i, "text": texts[i], "true_label": int(labels[i]),
                                 "predicted_label": int(preds[i]), "confidence": float(probs[i].max())})
    return selected

def predict_split(trainer, tokenized_split, raw_split, out, split):
    """Lưu mọi xác suất/nhãn và metrics runtime để tái lập sau khi chạy."""
    prediction = trainer.predict(tokenized_split, metric_key_prefix=split)
    logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
    labels = np.asarray(prediction.label_ids)
    probs = torch.tensor(logits).softmax(-1).numpy()
    filename = "all_predictions.json" if split == "test" else f"{split}_predictions.json"
    save_json(out / filename, [dict(index=i, text=t, true_label=int(y), predicted_label=int(p.argmax()),
        probabilities=p.tolist()) for i, (t, y, p) in enumerate(zip(raw_split["text"], labels, probs))])
    save_json(out / f"{split}_runtime_metrics.json", prediction.metrics)
    return logits, probs, labels


def select_threshold_for_validation(trainer, tokenized_validation, raw_validation, out, thresholds=None, checkpoint_path=None):
    """Chỉ hàm này được chọn ngưỡng; đầu vào luôn là split validation."""
    logits, probs, labels = predict_split(trainer, tokenized_validation, raw_validation, out, "validation")
    save_json(out / "validation_metrics.json", compute_metrics((logits, labels)))
    policy = export_threshold_analysis(labels, probs[:, 1], out / "validation_threshold",
                                      split="validation", thresholds=thresholds, select=True)
    policy["validation_prediction_file"] = "validation_predictions.json"
    policy["validation_dataset_fingerprint"] = getattr(raw_validation, "_fingerprint", None)
    if checkpoint_path is not None:
        policy["checkpoint_identity"] = checkpoint_identity(checkpoint_path)
    save_json(out / "decision_threshold.json", policy)
    return policy


def evaluate_model(trainer, tokenized_test, raw_test, out, decision_threshold=None):
    logits, probs, labels = predict_split(trainer, tokenized_test, raw_test, out, "test")
    metrics = compute_metrics((logits, labels))
    rows = choose_samples(raw_test["text"], labels, probs)
    save_json(out / "test_metrics.json", metrics)
    save_json(out / "sample_predictions.json", rows)
    plot_confusion(labels, probs.argmax(-1), out / "confusion_matrix.png")
    plot_metrics(metrics, out / "metrics_bar.png")
    plot_samples(rows, out / "sample_predictions.png")
    if decision_threshold is not None:
        threshold_metrics = binary_metrics_at_threshold(labels, probs[:, 1], decision_threshold)
        threshold_metrics.update(selection_split="validation", selection_performed_on_test=False)
        save_json(out / "test_threshold_metrics.json", threshold_metrics)
        folder = out / "test_threshold"; folder.mkdir(exist_ok=True)
        plot_confusion(labels, (probs[:, 1] >= decision_threshold).astype(int), folder / "confusion_matrix.png")
    print(json.dumps(metrics, indent=2))
    return metrics, rows

def error_hypothesis(text):
    lower = text.lower()
    clues = []
    if re.search(r"\b(but|although|however|despite|yet)\b", lower):
        clues.append("contrast or concession may mix positive and negative cues")
    if re.search(r"\b(not|never|no|hardly|barely)\b|n.t\b", lower):
        clues.append("negation scope may reverse the sentiment of nearby words")
    if re.search(r"\b(gang|pranksters|kill|war|crime)\b", lower):
        clues.append("negative plot vocabulary may describe subject matter rather than reviewer sentiment")
    if not clues:
        clues.append("implicit sentiment, irony or contextual references need a manual reading of this review")
    return "; ".join(clues) + ". This is a linguistic hypothesis, not a causal attribution."

def report_run(out, metrics, rows):
    correct = sum(r["true_label"] == r["predicted_label"] for r in rows)
    text = ["# Fine-tuning run report", "", f"Run: `{out.name}`", "",
            "Official splits: 8,530 train / 1,066 validation / 1,066 test.",
            "", "## Test metrics", "", "```json", json.dumps(metrics, indent=2), "```", "",
            "## Qualitative error review", "",
            "The examples below come from this run. Linguistic explanations are hypotheses, not measured model causes.", ""]
    for row in rows:
        if row["true_label"] != row["predicted_label"]:
            text += [f"- Review: {row['text']}",
                     f"  True={row['true_label']}, predicted={row['predicted_label']}, confidence={row['confidence']:.3f}.",
                     "  Hypothesis: " + error_hypothesis(row["text"])]
    if correct == len(rows): text += ["No misclassified examples were available in this run."]
    (out / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")

def finetune(config=None, allow_cpu=False, evaluate_test=True):
    config = config or read_config()
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("Full training requires a GPU by default. Use --allow-cpu only if you intentionally accept the cost.")
    seed_all(config["seed"])
    cfg = dict(config["exercise2"], seed=config["seed"])
    with run_context("finetuning" if evaluate_test else "candidate", config) as out:
        raw, tokenizer, tokenized = prepare_data(config)
        official_sizes = {k:len(raw[k]) for k in ("train","validation","test")}
        official_fingerprints = {k:raw[k]._fingerprint for k in ("train","validation","test")}
        if cfg.get("augmentation", {}).get("enabled", False):
            from .datasets.augmentation import augment_train, export_augmentation
            raw, augmented = augment_train(raw, cfg["augmentation"], config["seed"])
            export_augmentation(augmented, out)
            # Chỉ tokenize lại train sau augmentation; val/test giữ nguyên.
            from datasets import DatasetDict
            tokenized["train"] = tokenize_data(DatasetDict(train=raw["train"]), tokenizer, cfg["max_length"])["train"]
        lengths = [len(ids) for ids in tokenizer(list(raw["train"]["text"]), truncation=False)["input_ids"]]
        save_json(out / "data_statistics.json", {"split_sizes": official_sizes,
                  "effective_split_sizes": {k:len(raw[k]) for k in ("train","validation","test")},
                  "label_counts": {k: {str(label): int(sum(int(v) == label for v in raw[k]["label"])) for label in (0, 1)} for k in ("train", "validation", "test")},
                  "train_token_lengths": {"mean": float(np.mean(lengths)), "max": max(lengths),
                  "truncated_fraction": float(np.mean(np.array(lengths) > cfg["max_length"]))}})
        plot_lengths(lengths, cfg["max_length"], out / "token_length_distribution.png")
        model, _ = load_model(cfg["model_name"], cfg["model_revision"], finetuning=True,
                              overrides=cfg.get("model_overrides"))
        save_json(out / "provenance.json", {"model_commit": getattr(model.config, "_commit_hash", None),
                  "dataset_fingerprints": {k: raw[k]._fingerprint for k in ("train", "validation", "test")},
                  "official_dataset_fingerprints": official_fingerprints,
                  "effective_dropout": {key: getattr(model.config, key, None) for key in ("dropout", "attention_dropout", "seq_classif_dropout")}})
        trainer = make_trainer(model, tokenizer, tokenized, cfg, out)
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        trained = trainer.train()
        if abs(float(trainer.state.epoch) - float(cfg["num_train_epochs"])) > 1e-5:
            raise RuntimeError("Training did not finish the requested epoch budget")
        save_json(out / "selection_summary.json", {
            "requested_epochs": cfg["num_train_epochs"], "completed_epochs": trainer.state.epoch,
            "global_step": trainer.state.global_step, "early_stopping": False,
            "amp_skipped_optimizer_steps": getattr(trainer, "amp_skipped_optimizer_steps", 0),
            "successful_optimizer_updates": trainer.state.global_step - getattr(trainer, "amp_skipped_optimizer_steps", 0),
            "selection_split": "validation", "metric": trainer.args.metric_for_best_model,
            "best_metric": trainer.state.best_metric,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "test_evaluated": evaluate_test,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        })
        save_json(out / "train_metrics.json", trained.metrics)
        save_json(out / "trainer_history.json", trainer.state.log_history)
        plot_epoch_learning(trainer.state.log_history, out / "learning_curves.png")
        plot_step_learning(trainer.state.log_history, out / "step_learning_curves.png",
            best_step=int(Path(trainer.state.best_model_checkpoint).name.split("-")[-1]))
        trainer.save_model(str(out / "best_model"))
        tokenizer.save_pretrained(str(out / "best_model"))
        policy = None
        if cfg.get("threshold_selection", {}).get("enabled", True):
            # Cố định ngưỡng từ validation trước khi đọc kết quả test.
            policy = select_threshold_for_validation(trainer, tokenized["validation"], raw["validation"], out,
                                                     checkpoint_path=out / "best_model")
            save_json(out / "best_model/decision_threshold.json", policy)
        if not evaluate_test:
            # Không dự đoán test trong giai đoạn chọn cấu hình.
            checks = verify_saved_model(trainer.model, out / "best_model",
                                        [raw["validation"][0]["text"]], cfg["max_length"])
            save_json(out / "reload_check.json", checks)
            return out
        metrics, rows = evaluate_model(trainer, tokenized["test"], raw["test"], out,
                                      decision_threshold=policy["selected_threshold"] if policy else None)
        checks = verify_saved_model(trainer.model, out / "best_model",
                                    [raw["test"][0]["text"]], cfg["max_length"])
        save_json(out / "reload_check.json", checks)
        report_run(out, metrics, rows)
    return out


def recover_reload_failure(out):
    """Complete only the known post-training reload check, without retraining."""
    from filelock import FileLock
    from .utils.runs import hash_file, Tee
    from contextlib import redirect_stdout, redirect_stderr
    import sys
    out = Path(out).resolve()
    required = ["config.json", "train_metrics.json", "test_metrics.json", "all_predictions.json",
                "sample_predictions.json", "data_statistics.json", "console.log", "status.json",
                "learning_curves.png", "confusion_matrix.png", "metrics_bar.png",
                "token_length_distribution.png", "sample_predictions.png", "manifest.json"]
    for name in required:
        if not (out / name).is_file():
            raise RuntimeError(f"Cannot recover incomplete run: missing {name}")
    if json.loads((out / "status.json").read_text())["status"] != "failed":
        raise RuntimeError("Recovery is only for a failed reload-check run")
    if "Checkpoint reload predictions differ" not in (out / "console.log").read_text():
        raise RuntimeError("Run did not fail at the expected reload check")
    manifest = json.loads((out / "manifest.json").read_text())
    for name, digest in manifest.items():
        path = (out / name).resolve()
        if not path.is_relative_to(out) or hash_file(path) != digest:
            raise RuntimeError(f"Original run integrity check failed: {name}")
    cfg = json.loads((out / "config.json").read_text())
    train = json.loads((out / "train_metrics.json").read_text())
    if train["epoch"] != cfg["exercise2"]["num_train_epochs"]:
        raise RuntimeError("Training epochs did not finish")
    if len(json.loads((out / "all_predictions.json").read_text())) != 1066:
        raise RuntimeError("Expected predictions for all 1066 test examples")
    states = [json.loads(p.read_text()) for p in (out / "checkpoints").glob("checkpoint-*/trainer_state.json")]
    if not states:
        raise RuntimeError("No Trainer state available to identify the selected checkpoint")
    selected = Path(max(states, key=lambda state: state["global_step"])["best_model_checkpoint"])
    if not selected.is_dir():
        selected = out / "checkpoints" / selected.name
    reference, _ = load_model(str(selected))
    rows = json.loads((out / "sample_predictions.json").read_text())
    metrics = json.loads((out / "test_metrics.json").read_text())
    first = json.loads((out / "all_predictions.json").read_text())[0]["text"]
    # Kiểm tra trước khi đổi trạng thái/registry; giữ console.log gốc.
    checks = verify_saved_model(reference, out / "best_model", [first], cfg["exercise2"]["max_length"])
    with (out / "recovery.log").open("a", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            print("Verified exact weights against selected checkpoint:", selected.name)
            save_json(out / "reload_check.json", checks)
            report_run(out, metrics, rows)
            save_json(out / "recovery.json", {"original_status": "failed", "reason": "mixed execution contexts in old reload check",
                      "verified_against": selected.name, "retrained": False})
            print("Recovery completed without retraining:", out.name)
    save_json(out / "status.json", {"status": "completed", "recovered": True, "original_status": "failed"})
    save_json(out / "manifest.json", {str(p.relative_to(out)): hash_file(p)
              for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"})
    root = out.parents[2]
    registry = root / "config/workflow.json"
    with FileLock(str(root / "outputs/.registry.lock")):
        state = json.loads(registry.read_text())
        for run in state["runs"]:
            if run["id"] == out.name:
                run.update(status="completed", recovered=True, original_status="failed")
        state["active_runs"]["finetuning"] = out.name
        save_json(registry, state)
    return out

"""Chọn ngưỡng từ validation của checkpoint có sẵn, không huấn luyện lại."""
from pathlib import Path
import argparse
import sys
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs.lab03.src.workflow import read_config, seed_all, select_threshold_for_validation
from labs.lab03.src.models.classifier import load_model
from labs.lab03.src.datasets.loader import load_data, tokenize_data
from labs.lab03.src.utils.runs import run_context

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    if not args.model_path.is_dir(): parser.error("Restore best_model/ first; checkpoint is required")
    saved = args.model_path.resolve().parent / "config.json"
    cfg = read_config(saved if saved.exists() else None); seed_all(cfg["seed"])
    cfg["threshold_checkpoint"] = str(args.model_path.resolve())
    with run_context("threshold_selection", cfg) as out:
        raw = load_data(cfg["exercise2"])
        model, tok = load_model(str(args.model_path))
        # Chỉ tokenize và predict validation, không truy cập nội dung test để chọn ngưỡng.
        from datasets import DatasetDict
        validation = tokenize_data(DatasetDict(validation=raw["validation"]), tok, cfg["exercise2"]["max_length"])
        trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(out / "temporary"),
            per_device_eval_batch_size=cfg["exercise2"]["eval_batch_size"], report_to="none"),
            processing_class=tok, data_collator=DataCollatorWithPadding(tok))
        selected = select_threshold_for_validation(trainer, validation["validation"], raw["validation"], out,
                                                   checkpoint_path=args.model_path)
        print("Selected threshold from validation:", selected["selected_threshold"])
    print("Frozen policy:", out / "decision_threshold.json")

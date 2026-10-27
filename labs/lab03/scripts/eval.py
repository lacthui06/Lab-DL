from pathlib import Path
import sys
import argparse
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs.lab03.src.workflow import read_config, evaluate_model, seed_all
from labs.lab03.src.models.classifier import load_model
from labs.lab03.src.datasets.loader import tokenize_data, load_data
from labs.lab03.src.utils.runs import run_context
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--threshold-file", type=Path, help="Frozen validation-selected policy")
    args = parser.parse_args()
    if not args.model_path.is_dir(): parser.error("Fine-tuned checkpoint directory not found")
    saved_config = args.model_path.resolve().parent / "config.json"
    if (args.model_path/'training_config.json').is_file():
        saved_config = args.model_path/'training_config.json'
    cfg = read_config(saved_config if saved_config.is_file() else None)
    seed_all(cfg["seed"])
    cfg["evaluated_checkpoint"] = str(args.model_path.resolve())
    with run_context("evaluation", cfg) as out:
        raw = load_data(cfg["exercise2"])
        model, tok = load_model(str(args.model_path))
        tokenized = tokenize_data(raw, tok, cfg["exercise2"]["max_length"])
        trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(out / "temporary"),
            per_device_eval_batch_size=cfg["exercise2"]["eval_batch_size"], report_to="none"),
            processing_class=tok, data_collator=DataCollatorWithPadding(tok))
        from labs.lab03.src.decision import load_decision_policy
        from labs.lab03.src.utils.runs import save_json
        policy = load_decision_policy(args.model_path, args.threshold_file)
        if policy: save_json(out / "decision_threshold.json", policy)
        evaluate_model(trainer, tokenized["test"], raw["test"], out,
                       decision_threshold=policy["selected_threshold"] if policy else None)

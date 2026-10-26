from pathlib import Path
import sys
import argparse
import json
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs.lab03.src.workflow import read_config
from labs.lab03.src.models.classifier import load_model, manual_predict
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--threshold-file", type=Path, help="Frozen validation-selected policy")
    args = parser.parse_args()
    if args.threshold_file and not args.model_path:
        parser.error("--threshold-file requires --model-path")
    if args.model_path and not args.model_path.is_dir():
        parser.error("Checkpoint directory does not exist; refusing to fall back to an untuned model")
    saved_config = args.model_path.resolve().parent / "config.json" if args.model_path else None
    cfg = read_config(saved_config if saved_config and saved_config.is_file() else None)
    ex1 = cfg["exercise1"]
    model, tok = load_model(str(args.model_path) if args.model_path else ex1["model_name"],
                           None if args.model_path else ex1["revision"])
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    from labs.lab03.src.decision import load_decision_policy, apply_policy
    from labs.lab03.src.utils.runs import run_context, save_json
    policy = load_decision_policy(args.model_path, args.threshold_file) if args.model_path else None
    cfg["inference"] = {"text": args.text, "model": str(args.model_path) if args.model_path else ex1["model_name"],
                        "threshold": policy["selected_threshold"] if policy else None,
                        "threshold_file": str(args.threshold_file) if args.threshold_file else None}
    with run_context("inference", cfg) as out:
        if policy: save_json(out / "decision_threshold.json", policy)
        predictions = manual_predict(model, tok, [args.text],
            max_length=cfg["exercise2"]["max_length"] if args.model_path else 128)
        predictions = apply_policy(predictions, policy, model.config.id2label)
        save_json(out / "predictions.json", predictions)
        from labs.lab03.src.analysis import _table_image
        _table_image(predictions, ["text", "label", "score"], out / "inference.png", "Sentiment inference", [.74,.13,.13])
        print(json.dumps(predictions, indent=2))
    print("Inference artifacts:", out)

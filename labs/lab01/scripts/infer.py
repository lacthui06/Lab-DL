"""Current run-based infer entry point; supports both model names ('final', 'MainMLP', ...) and direct run paths."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference import infer_sample, resolve_run_path

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="final",
                   help="Model name ('final', 'SmallMLP', 'MainMLP', 'MainMLP_FE') to run inference with. Default: final")
    p.add_argument("--run", type=str, default=None,
                   help="Optional explicit path to run directory or best.pt (overrides --model).")
    p.add_argument("--data", type=Path, default=None,
                   help="Override prepared.npz path when moving machines")
    p.add_argument("--sample-idx", type=int, default=0,
                   help="Index of sample in test set to predict. Default: 0")
    args = p.parse_args()
    target = args.run if args.run is not None else args.model
    run_path = resolve_run_path(target)
    print(infer_sample(run_path, args.sample_idx, args.data))

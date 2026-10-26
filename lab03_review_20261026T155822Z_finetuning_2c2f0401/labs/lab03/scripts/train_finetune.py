from pathlib import Path
import sys
import argparse
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs.lab03.src.workflow import finetune
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    out = finetune(allow_cpu=args.allow_cpu)
    from labs.lab03.src.analysis import ensure_analysis
    print("Training run:", out)
    print("Visual analysis:", ensure_analysis(out))

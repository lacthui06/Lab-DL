from pathlib import Path
import sys
import argparse
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs.lab03.src.workflow import exercise1,read_config
from labs.lab03.src.utils.runs import ROOT
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'config/frozen_trial.json')
    args=parser.parse_args()
    out = exercise1(read_config(args.config))
    from labs.lab03.src.analysis import ensure_exercise1_analysis
    print("Exercise 1 run:", out)
    print("Visual analysis:", ensure_exercise1_analysis(out))

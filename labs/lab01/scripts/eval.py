"""Current run-based eval entry point; legacy YAML interface is in Git history (e0fd041)."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.inference import evaluate_run

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--data",type=Path,help="Override prepared.npz path when moving machines")
    args=p.parse_args()
    print(evaluate_run(args.run,args.data))

"""Train a current Lab01 configuration with run-based artifact tracking."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.workflow import load_workflow_config, model_stage

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices=["SmallMLP","MainMLP","MainMLP_FE"],default="MainMLP")
    p.add_argument("--config",type=Path)
    args=p.parse_args()
    print(model_stage(load_workflow_config(args.config),args.model,"train"))

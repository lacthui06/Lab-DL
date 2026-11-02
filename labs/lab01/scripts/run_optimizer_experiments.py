"""Run controlled optimizer experiments on local CPU/CUDA or Kaggle."""
import argparse
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.training.tuning import run_fixed_lr,run_tuning,refit_winner,summarize_optimizer_runs
from src.utils.runs import Run,write_json
from src.workflow import load_workflow_config


def main():
    config=load_workflow_config()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data",type=Path,help="prepared.npz from notebook 01")
    parser.add_argument("--epochs",type=int,default=config["epochs"])
    parser.add_argument("--trials",type=int,default=config["trials_per_optimizer"],help="Trials per optimizer")
    parser.add_argument("--seed",type=int,default=config["seed"])
    parser.add_argument("--batch-size",type=int,default=config["batch_size"])
    parser.add_argument("--threads",type=int,default=config["torch_threads"])
    args=parser.parse_args()
    if min(args.epochs,args.trials,args.batch_size,args.threads)<1:
        parser.error("epochs, trials, batch size and threads must be positive")
    torch.set_num_threads(args.threads)
    config=vars(args).copy();config["data"]=str(args.data.resolve())
    with Run("optimizer_experiment_suite",config) as run:
        paths={}
        fixed=run_fixed_lr(args.data,args.epochs,args.seed,args.batch_size)
        paths["fixed"]=str(fixed.resolve());write_json(run.path/"phase_paths.json",paths)
        tuning=run_tuning(args.data,args.trials,args.epochs,args.seed,args.batch_size)
        paths["tuning"]=str(tuning.resolve());write_json(run.path/"phase_paths.json",paths)
        final=refit_winner(args.data,tuning,args.epochs,args.seed,args.batch_size)
        paths["final"]=str(final.resolve());write_json(run.path/"phase_paths.json",paths)
        summary=summarize_optimizer_runs(fixed,tuning,final)
        paths["summary"]=str(summary.resolve());write_json(run.path/"phase_paths.json",paths)
        print("Suite completed:",run.path)


if __name__=="__main__":
    main()

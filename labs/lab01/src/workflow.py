"""Current Lab01 entry points: one config, explicit saved runs, shared train/view cells."""
import json
from pathlib import Path
import torch
from .utils.helpers import SOURCE_DIR, BASE_DIR
from .training.comparison import run_model
from .utils.comparison_plots import build_comparison
from .training.tuning import run_fixed_lr, run_tuning, refit_winner, summarize_optimizer_runs


def load_workflow_config(path=None):
    return json.loads(Path(path or SOURCE_DIR/"config/workflow.json").read_text(encoding="utf-8"))


def saved_run(config,key,name=None):
    value=config["runs"][key]
    if name is not None:value=value[name]
    path=Path(value)
    if not path.is_absolute():path=BASE_DIR/path
    if not path.is_dir():raise FileNotFoundError(f"Saved run unavailable: {path}. Copy artifacts or edit config/workflow.json.")
    return path


def validate_mode(mode):
    if mode not in ("view","train"):raise ValueError("MODE must be view or train")


def model_stage(config,name,mode="view"):
    validate_mode(mode)
    if mode=="view":return saved_run(config,"models",name)
    torch.set_num_threads(config["torch_threads"])
    return run_model(saved_run(config,"preparation")/"prepared.npz",name,
        epochs=config["epochs"],seed=config["seed"],batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],optimizer_name=config["optimizer"],
        weight_decay=config["weight_decay"],momentum=config["momentum"],
        model_config=config["models"][name])


def comparison_stage(config,paths,mode="view"):
    validate_mode(mode)
    if mode == "train":
        return build_comparison(paths)
    result = saved_run(config,"comparison")
    sources = json.loads((result/"config.json").read_text(encoding="utf-8"))["source_runs"]
    # Run IDs survive copying artifacts between Windows, Linux and Kaggle.
    source_ids = [p.replace("\\", "/").rstrip("/").split("/")[-1] for p in sources]
    if source_ids != [Path(p).name for p in paths]:
        raise ValueError("Saved comparison uses different model runs. Select matching runs in config/workflow.json or rebuild the comparison.")
    return result


def optimizer_stage(config,phase,mode="view",tuning=None,fixed=None,final=None):
    validate_mode(mode)
    if mode=="view":return saved_run(config,phase)
    torch.set_num_threads(config["torch_threads"])
    data=saved_run(config,"preparation")/"prepared.npz"
    args=dict(epochs=config["epochs"],seed=config["seed"],batch_size=config["batch_size"])
    if phase=="fixed":return run_fixed_lr(data,**args)
    if phase=="tuning":return run_tuning(data,trials_per_optimizer=config["trials_per_optimizer"],**args)
    if phase=="final":return refit_winner(data,tuning,**args)
    if phase=="optimizer_results":return summarize_optimizer_runs(fixed,tuning,final)
    raise ValueError(f"Unknown phase: {phase}")


def show_tables(run,names):
    import pandas as pd
    from IPython.display import display
    for name in names:
        print(name);display(pd.read_csv(Path(run)/name))


def show_images(run,names):
    from IPython.display import display,Image
    for name in names:display(Image(filename=str(Path(run)/name)))

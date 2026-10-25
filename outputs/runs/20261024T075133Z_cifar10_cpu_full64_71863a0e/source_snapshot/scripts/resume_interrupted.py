"""Resume a suite interrupted between architectures, preserving completed results."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contextlib
import json
import traceback
import numpy as np
import pandas as pd
import torch
from src.experiment import ROOT, run_model, seed_all, write_json, Tee
from src.data import file_hash
from src.reporting import summarize_suite

run = Path(sys.argv[1]).resolve()
cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
assert cfg.get("dataset") == "CIFAR10", "Legacy runs require their source_snapshot; do not mix datasets"
if json.loads((run / "status.json").read_text(encoding="utf-8"))["status"] == "completed":
    raise SystemExit("Run already completed")
work = run.parents[2]
torch.set_num_threads(cfg["threads"])
torch.hub.set_dir(str(work / "data" / "pretrained"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selected = dict(np.load(run / "selected_indices.npz"))
table = pd.read_csv(run / "comparison.csv") if (run / "comparison.csv").exists() else pd.DataFrame()
rows = table.to_dict("records")
try:
    with (run / "console.log").open("a", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print("Resumed after interruption; completed architectures retained.", flush=True)
            for name in cfg["models"]:
                if len(table) and len(table[table.model == name]) == 2:
                    print("Retained:", name, flush=True)
                    continue
                # Avoid silently overwriting any completed model-stage checkpoint.
                if (run / name / "finetune" / "best.pt").exists():
                    raise RuntimeError("Partial architecture has a fine-tune checkpoint; inspect before resuming")
                rows.extend(run_model(name, cfg, selected, work / "data", run, device))
                pd.DataFrame(rows).to_csv(run / "comparison.csv", index=False)
            summarize_suite(run, cfg, selected)
            write_json(run / "status.json", {"status": "completed", "resumed": True})
            write_json(work / "outputs" / "selected_run.json", {"run": str(run)})
            print("Completed:", run, flush=True)
except BaseException:
    write_json(run / "status.json", {"status": "failed", "traceback": traceback.format_exc()})
    raise
finally:
    write_json(run / "resume_source_hashes.json", {
        str(p.relative_to(ROOT)): file_hash(p) for p in (ROOT / "src").glob("*.py")})
    write_json(run / "manifest.json", {str(p.relative_to(run)): file_hash(p)
        for p in run.rglob("*") if p.is_file() and p.name != "manifest.json"})

"""Audit selected experiment outputs without GPU, network or model weights."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs.lab03.src.evidence import selected_run, audit_exercise1, audit_finetuning
from labs.lab03.src.utils.runs import save_json, ROOT, run_context

from labs.lab03.src.evidence import read_json
from labs.lab03.src.threshold_evidence import audit_threshold_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "validation/artifact_audit.json")
    args = parser.parse_args()
    result = {"exercise1": audit_exercise1(selected_run("exercise1")),
              "finetuning": audit_finetuning(selected_run("finetuning"))}
    kinds = ["threshold_selection", "evaluation", "inference"]
    state = read_json(ROOT / "config/workflow.json")
    if all(k in state["active_runs"] for k in kinds):
        result["threshold_experiment"] = audit_threshold_experiment(*(selected_run(k) for k in kinds))
    with run_context("artifact_audit", {"purpose": "verify persisted evidence without model execution"}) as out:
        save_json(out / "artifact_audit.json", result)
        print("Saved audit artifact:", out)
    save_json(args.output, result)
    print("Artifact audit: PASSED")
    print("Verified files:", sum(row["verified_files"] for row in result.values()))
    print("Recomputed test accuracy:", result["finetuning"]["metrics"]["accuracy"])
    print("Checkpoint availability:", result["finetuning"]["checkpoint_available_locally"])
    print("Audit report:", args.output)

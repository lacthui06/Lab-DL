"""Audit selected experiment outputs without GPU, network or model weights."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs.lab03.src.evidence import selected_run, audit_exercise1, audit_finetuning, verify_manifest
from labs.lab03.src.utils.runs import save_json, ROOT, run_context

from labs.lab03.src.threshold_analysis import matching_threshold_runs
from labs.lab03.src.threshold_evidence import audit_threshold_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "validation/artifact_audit.json")
    args = parser.parse_args()
    result = {"exercise1": audit_exercise1(selected_run("exercise1")),
              "finetuning": audit_finetuning(selected_run("finetuning"))}
    paths = matching_threshold_runs(selected_run("finetuning"))
    if paths:
        result["threshold_experiment"] = audit_threshold_experiment(
            paths["threshold_selection"], paths["evaluation"], paths["inference"])
    with run_context("artifact_audit", {"purpose": "verify persisted evidence without model execution"}) as out:
        save_json(out / "artifact_audit.json", result)
        print("Saved audit artifact:", out)
    save_json(args.output, result)
    print("Artifact audit: PASSED")
    sources = {selected_run("exercise1"), selected_run("finetuning")}
    if paths: sources.update(paths.values())
    print("Unique verified source files:", sum(verify_manifest(p)["verified_files"] for p in sources))
    print("Recomputed test accuracy:", result["finetuning"]["metrics"]["accuracy"])
    print("Checkpoint availability:", result["finetuning"]["checkpoint_available_locally"])
    print("Audit report:", args.output)

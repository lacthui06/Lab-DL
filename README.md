# Deep Learning Labs

Repository containing source code, engineering workflows, and practical lab exercises for Deep Learning coursework.

---

## Directory Structure

- `.agents/`: Skill framework and operational rules for AI Coding Agents.
- `docs/`: Consolidated reference documentation (EDA, Feature Engineering, Deep Learning pipeline, Data Augmentation, Training and Optimization).
- `labs/`: Course practical lab implementations.

---

## Environment

- **Python requirement:** `>=3.12, !=3.14.1`, derived from the pinned libraries (numpy 2.5.3 needs `>=3.12`; torchvision 0.29.1 excludes `3.14.1`). Verified local build: CPython **3.13.15** (64-bit). The requirement and the verified build are recorded as comments in `requirements.txt`, never as a `python==x.y.z` pip line (invalid pip syntax).
- **Dependency specifications:** see `requirements.txt` for the repository environment and [Lab01 requirements](labs/lab01/requirements.txt) for its supported dependency ranges. The root file mixes exact pins and lower bounds (including pandas and Optuna); it is not a complete lockfile. Actual installed versions belong to each run snapshot.
- **Run-time evidence:** current Lab01 runs write `outputs/runs/<run_id>/environment_snapshot.json` (legacy snapshots remain in `outputs/logs/`) with the interpreter version, platform, library versions, and the resolved compute device. The Kaggle interpreter is logged per run rather than assumed equal to local.

---

Lab03 and New Lab03 use their own pinned dependencies in `labs/lab03/requirements.txt` and `labs/new-lab03/requirements.txt`, with separate Python 3.12 environments; do not replace the Lab01 environment with these pins.

Lab03 retains the completed five-epoch experiment. New Lab03 is a separate three-epoch experiment in `labs/new-lab03/`, with two notebooks, inline visualizations and independent artifacts. Both use validation-only checkpoint and threshold selection.

## Lab Directory

- **Lab 01**: [FashionMNIST Classification with PyTorch](labs/lab01/README.md) — local experiments available; final PDF pending.
- **Lab 02**: Upcoming
- **Lab 03**: [Hugging Face Sentiment Analysis & Fine-tuning](labs/lab03/README.md) — completed five-epoch experiment.
- **New Lab 03**: [Three-epoch sentiment experiment](labs/new-lab03/README.md) — verified three-epoch experiment; test accuracy82.18%, separate notebooks and artifacts.
- **Lab 04**: Upcoming

---

## Lab Architecture (`labs/lab0X/`)

Each practical lab is organized into modular directories and a single living documentation file:

- `config/`: Current Lab01 configuration and explicit run registry in `workflow.json`; legacy YAML configurations remain for compatible APIs.
- `data/`: Experimental datasets (ignored by git).
- `notebooks/`: Jupyter Notebooks for interactive execution, analysis, and visualization.
- `src/`: Modular, production-ready Python subpackages (`datasets/`, `models/`, `training/`, `utils/`).
- `scripts/`: Command-line entry points (`train.py`, `eval.py`, `infer.py`).
- `tests/`: Mandatory TDD test suites (`test_datasets.py`, `test_models.py`, `test_training.py`).
- `outputs/`: Current per-run artifacts in `runs/<run_id>/` (config, logs, metrics, tables, figures, checkpoints); historical `checkpoints/`, `logs/`, `eda/`, and `predictions/` remain as evidence.
- `description_lab_0X.md`: Original assignment brief from instructor (input, kept verbatim).
- `walkthrough_lab_0X.md`: Single Living Walkthrough unifying 3 Core Rules (Section 1: Spec with brief interpretation (Mục 1.0) from `spec-driven-development`, Section 2: experiment methods and artifact-backed evidence from `walkthrough-and-experiment-tracking`, Section 3: Final Report with decision traceability from `documentation-and-adrs`).

Lab01 keeps software change journals and troubleshooting logs local, outside Git and the report. See the Lab01 README for the two-notebook workflow and published report evidence.

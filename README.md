# Deep Learning Labs

Repository containing source code, engineering workflows, and practical lab exercises for Deep Learning coursework.

---

## Directory Structure

- `.agent/`: Skill framework and operational rules for AI Coding Agents.
- `docs/`: Consolidated reference documentation (EDA, Feature Engineering, Deep Learning pipeline, Data Augmentation, Training and Optimization).
- `labs/`: Course practical lab implementations.

---

## Environment

- **Python requirement:** `>=3.12, !=3.14.1`, derived from the pinned libraries (numpy 2.5.3 needs `>=3.12`; torchvision 0.29.1 excludes `3.14.1`). Verified local build: CPython **3.13.15** (64-bit). The requirement and the verified build are recorded as comments in `requirements.txt`, never as a `python==x.y.z` pip line (invalid pip syntax).
- **Library pins:** see `requirements.txt` (torch 2.14.0, torchvision 0.29.1, numpy 2.5.3, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2, Pillow 12.3.0, pytest 9.1.1).
- **Run-time evidence:** each lab run writes `outputs/environment_snapshot.json` with the interpreter version, platform, library versions, and the resolved compute device. The Kaggle interpreter is logged per run rather than assumed equal to local.

---

## Lab Directory

- **Lab 01**: FashionMNIST Classification with PyTorch (Completed)
- **Lab 02**: Upcoming
- **Lab 03**: Upcoming
- **Lab 04**: Upcoming

---

## Lab Architecture (`labs/lab0X/`)

Each practical lab is organized into modular directories and a single living documentation file:

- `data/`: Experimental datasets (ignored by git).
- `notebooks/`: Jupyter Notebooks for interactive execution, analysis, and visualization.
- `src/`: Modular, production-ready Python components (`config.py`, `dataset.py`, `model.py`, `engine.py`, `utils.py`).
- `outputs/`: Artifact store for training checkpoints, evaluation logs, and high-resolution figures.
- `description_lab_0X.md`: Original assignment brief from instructor (input, kept verbatim).
- `walkthrough_lab_0X.md`: Single Living Walkthrough unifying 3 Core Rules (Section 1: Spec with brief interpretation (Mục 1.0) from `spec-driven-development`, Section 2: 5-Stage Changelog from `walkthrough-and-experiment-tracking`, Section 3: Final Report with decision traceability from `documentation-and-adrs`).

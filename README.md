# Deep Learning Labs

Repository containing source code, engineering workflows, and practical lab exercises for Deep Learning coursework.

---

## Directory Structure

- `agent/`: Skill framework and operational rules for AI Coding Agents.
- `workflows/`: Foundational Machine Learning workflows and playbooks (EDA, Feature Engineering).
- `dl_workflows/`: Deep Learning production playbooks (Data Augmentation, Training Loops, Optimization).
- `labs/`: Course practical lab implementations.

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
- `walkthrough_lab_0X.md`: Single Living Walkthrough unifying 3 Core Rules (Section 1: Spec from `spec-driven-development`, Section 2: 5-Stage Changelog from `walkthrough-and-experiment-tracking`, Section 3: Final Report from `documentation-and-adrs`).

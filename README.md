# Deep Learning Labs

- **Tác giả:** Thái Anh Lạc
- **Mô tả:** Repository chứa toàn bộ mã nguồn, quy trình kỹ thuật thực nghiệm và bài tập thực hành môn Học Sâu (Deep Learning).

---

## Directory Structure

- `.agents/`: Skill framework and operational rules for AI Coding Agents.
- `docs/`: Consolidated reference documentation (EDA, Feature Engineering, Deep Learning pipeline, Data Augmentation, Training and Optimization).
- `labs/`: Course practical lab implementations.

---

## Environment

- **Python and dependencies:** use the selected lab's requirements and README. Record the interpreter and installed libraries actually used in each run; do not assume the same environment across labs or between local and Kaggle.
- **Dependency specifications:** see [Lab01 requirements](labs/lab01/requirements.txt) and [Lab02 requirements](labs/lab02/requirements.txt). The root requirements are shared guidance, not proof that every lab has been verified in one environment.
- **Run-time evidence:** current Lab01 runs write `outputs/runs/<run_id>/environment_snapshot.json` (legacy snapshots remain in `outputs/logs/`) with the interpreter version, platform, library versions, and the resolved compute device. The Kaggle interpreter is logged per run rather than assumed equal to local.

---

## Lab Directory

- **Lab 01**: [FashionMNIST Classification with PyTorch](labs/lab01/README.md) — local experiments available; [Walkthrough](labs/lab01/reports/walkthrough_lab_01.md) & [HTML Report](labs/lab01/reports/walkthrough_lab_01.html).
- **Lab 02**: [CIFAR-10 Transfer Learning](labs/lab02/README.md) — ResNet18 and DenseNet121, EDA/modeling notebooks, experiments and saved results.
- **Lab 03**: [Hugging Face Sentiment Analysis & Fine-tuning](https://github.com/lacthui06/Lab-DL/tree/lab03/labs/lab03) — notebooks and experiment results on the `lab03` branch.
- **Lab 04**: Upcoming

---

## Lab Architecture (`labs/lab0X/`)

Each practical lab is organized into modular directories and a single living documentation file:

- `reports/`: Living walkthrough, interactive HTML report with collapsible code blocks, and exported PDF deliverables.
- `config/`: JSON or YAML configurations matching the actual pipeline; Lab01 uses `workflow.json` and `publication.json`, while Lab02 provides experiment profiles.
- `data/`: Experimental datasets (ignored by git).
- `models/`: Production-ready deliverable checkpoints (`*.pt`), `manifest.json` (audit provenance), and quick-load guide.
- `notebooks/`: Jupyter Notebooks for interactive execution, analysis, and visualization. All notebooks follow the mandatory Pipeline Execution Roadmap standard at the first cell.
- `src/`: Modular, production-ready Python subpackages (`datasets/`, `models/`, `training/`, `utils/`).
- `scripts/`: Actual command-line entry points; see each lab README for supported train, evaluation, inference and artifact commands.
- `tests/`: Mandatory TDD test suites (`test_datasets.py`, `test_models.py`, `test_training.py`).
- `outputs/`: Current per-run artifacts in `runs/<run_id>/` (config, logs, metrics, tables, figures, checkpoints); historical `checkpoints/`, `logs/`, `eda/`, and `predictions/` remain as evidence.
- `description_lab_0X.md`: Original assignment brief from instructor (input, kept verbatim).
- `walkthrough_lab_0X.md`: Single Living Walkthrough unifying 3 Core Rules (Section 1: Spec with brief interpretation (Mục 1.0) from `spec-driven-development`, Section 2: experiment methods and artifact-backed evidence from `walkthrough-and-experiment-tracking`, Section 3: Final Report with decision traceability from `documentation-and-adrs`).

Lab01 keeps software change journals and troubleshooting logs local, outside Git and the report. See the Lab01 README for the two-notebook workflow and published report evidence.

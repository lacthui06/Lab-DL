# Lab 03: Sentiment Analysis and Text Classification Finetuning with Hugging Face

Repository implementation of Lab 03 according to AGENTS.md constitution.

## Overview
- **Exercise 1:** Sentiment Analysis with Hugging Face pipeline and manual Tokenizer/Model inference.
- **Exercise 2:** Finetuning Pretrained Model (`distilbert-base-uncased`) on IMDB dataset with Hugging Face `Trainer` API.

## Structure
- `config/`: Configuration files (`config.yaml`, `augmentations.yaml`, `experiments.yaml`).
- `data/`: Dataset caches and local data directories.
- `notebooks/`: Interactive execution notebook (`lab03.ipynb`).
- `outputs/`: Metrics, checkpoints, evaluation curves, logs, and confusion matrix.
- `scripts/`: CLI execution tools (`train.py`, `eval.py`, `infer.py`).
- `src/`: Modular codebase (`datasets`, `models`, `training`, `utils`).
- `tests/`: TDD verification suite (`test_datasets.py`, `test_models.py`, `test_training.py`).
- `walkthrough_lab_03.md`: Technical report and experiment changelog.

## Quickstart
1. Run tests:
   ```bash
   pytest labs/lab03/tests/ -v
   ```
2. Run single sentence inference:
   ```bash
   python labs/lab03/scripts/infer.py --text "This movie was absolutely wonderful and thrilling!"
   ```
3. Run training:
   ```bash
   python labs/lab03/scripts/train.py
   ```
4. Run evaluation:
   ```bash
   python labs/lab03/scripts/eval.py
   ```

---
name: code-simplification
description: Refactor and streamline Data Science and Deep Learning codebases. Convert exploratory Jupyter Notebook code into modular, production-ready Python packages, eliminating global variables, hard-coded constants, and non-standard characters.
---

# code-simplification (DS & AI Edition)

## Overview
This skill refactors exploratory notebook scripts, scattered global variables, and ad-hoc logic into **clean, modular, production-grade Python code**. It prepares codebases for coursework submission or engineering integration while guaranteeing exact reproducibility of experimental results.

## When to Use
- Cleaning and organizing a codebase after successful model training prior to submission.
- Extracting modular reusable components from `.ipynb` notebooks into structured `.py` files under `src/`.
- Removing hard-coded hyperparameters and fragmented constants.

## The 6 Refactoring Steps

### 1. Eliminate Hard-Coded Constants and File Paths
- Centralize all hyperparameters, paths, and execution settings into structured configuration classes or YAML configs:
  ```python
  from dataclasses import dataclass

  @dataclass
  class TrainConfig:
      data_path: str = "data/raw_dataset.csv"
      batch_size: int = 32
      learning_rate: float = 1e-4
      epochs: int = 20
      random_seed: int = 42
  ```

### 2. Eliminate Global Variables
- Replace notebook-style global state with pure functions with explicit signatures:
  - `load_data(path: str) -> pd.DataFrame`
  - `preprocess_features(df: pd.DataFrame, is_train: bool) -> np.ndarray`
  - `train_one_epoch(model: nn.Module, loader: DataLoader, ...) -> float`

### 3. Maintain Single Source of Truth for Preprocessing
- Do not maintain duplicate preprocessing logic for training and evaluation. Encapsulate transformations in unified pipeline classes or Torchvision transforms to guarantee identical behavior during inference.

### 4. Add Type Hints and Precise Docstrings
- Include type annotations and tensor shape docstrings to eliminate ambiguity:
  ```python
  def forward(self, x: torch.Tensor) -> torch.Tensor:
      """
      Args:
          x: Input tensor of shape (batch_size, channels, height, width).
      Returns:
          Logits tensor of shape (batch_size, num_classes).
      """
  ```

### 5. Remove Dead Code and Unused Imports
- Purge unused imports, scratch `print()` statements, and deprecated experiment cells.

### 6. Enforce Strict No-Emoji Policy
- Prohibit emojis and decorative icons in:
  - Source code files (`.py`, `.sh`).
  - Terminal output statements (`print()`, loggers, exception messages).
  - Code comments and docstrings.
  - Markdown and code cells of Jupyter Notebooks (`.ipynb`).
- Express all statuses with standard technical text (e.g. `[INFO]`, `[SUCCESS]`, `[ERROR]`, `Section 1: ...`). This prevents `UnicodeEncodeError` issues across diverse operating systems (such as Windows CP1252) and maintains professional academic standards.

## Verification Criteria
- Re-executing refactored modules with fixed random seeds yields identical losses and evaluation metrics.
- Repository scans confirm zero emojis or non-standard graphical symbols across code and notebook files.

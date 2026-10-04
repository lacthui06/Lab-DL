---
name: planning-and-task-breakdown
description: Decompose technical specifications (Specs) into atomic, testable pipeline phases with dependency ordering, clear contracts, and verified acceptance criteria for Data Science and AI projects.
---

# planning-and-task-breakdown (DS & AI Edition)

## Overview
This skill translates a technical specification into an actionable **Machine Learning Implementation Plan**. It partitions complex workflows into modular, independently testable phases before connecting them into an end-to-end pipeline.

## When to Use
- Immediately following completion of `spec-driven-development`.
- Prior to creating or modifying modular Python packages in `src/`.

## Standard 6-Phase Pipeline Decomposition

### Phase 1: Data Ingestion and Validation
- Task 1.1: Load raw data, inspect initial dimensions, and verify integrity.
- Task 1.2: Implement schema and type assertions (checking target ranges, null constraints).

### Phase 2: Preprocessing and Feature Pipeline
- Task 2.1: Handle missing values and outliers.
- Task 2.2: Apply categorical encoding, feature normalization, or image augmentation.
- *Rule:* All transformation fits must be computed strictly on the training partition and transformed across validation/test partitions to prevent data leakage.

### Phase 3: Baseline Model Construction
- Task 3.1: Implement minimal heuristic or standard baseline (e.g. 2-layer MLP).
- Task 3.2: Record initial benchmark metrics to establish the comparative baseline.

### Phase 4: Core Model and Deep Learning Pipeline
- Task 4.1: Define model architecture (neural network layers, normalization, activation heads).
- Task 4.2: Construct training loop (forward, criterion, backward, optimizer, scheduler).
- Task 4.3: Integrate checkpointing (`best_model.pt`) and early stopping mechanisms.

### Phase 5: Evaluation and Diagnostic Error Analysis
- Task 5.1: Measure final metrics on the independent test set.
- Task 5.2: Render visualization artifacts (loss/accuracy curves, confusion matrix).
- Task 5.3: Inspect misclassified sample slices to determine failure modes.

### Phase 6: Packaging and Modularization
- Task 6.1: Organize reusable modules under `src/` (`config.py`, `dataset.py`, `model.py`, `engine.py`, `utils.py`).
- Task 6.2: Ensure clean interface boundaries and script reproducibility.

## Acceptance Criteria
- Every task in the plan must define:
  - Explicit inputs and outputs.
  - Verification assertions confirming task success before advancing to subsequent stages.

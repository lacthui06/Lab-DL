---
name: spec-driven-development
description: Author technical ML/DL specifications before implementing code. Automatically links and extracts standards from workflows/ML_flow.md (for Tabular ML) or dl_workflows/DL_flow.md (for Deep Learning). Directly maps to Section 1 of walkthrough_lab_0X.md.
---

# spec-driven-development (DS & AI Edition)

## Overview
This skill mandates that AI agents author a comprehensive **Machine Learning / Deep Learning Specification (ML/DL Spec)** prior to coding. It grounds the project in standardized data contracts, objective loss formulations, baseline architectures, and quantitative evaluation criteria.

## Workflow References
When writing specifications, agents **MUST** extract standards from the corresponding workflow playbook:
- **Tabular ML:** Reference [`workflows/ML_flow.md`](../../workflows/ML_flow.md) and [`workflows/eda_guide.md`](../../workflows/eda_guide.md) for EDA guidelines, normalization standards, categorical encoding, and baseline benchmarks.
- **Deep Learning (CV / NLP):** Reference [`dl_workflows/DL_flow.md`](../../dl_workflows/DL_flow.md) for input modalities, tensor dimensions, backbone selection, and specialized loss functions (e.g. Focal Loss, Label Smoothing).

---

## Standard ML/DL Spec Architecture (Integrated into Section 1 of `walkthrough_lab_0X.md`)

The specification is authored directly in **Section 1 of the single living document `walkthrough_lab_0X.md`** before writing code (governed by Rule 1 in the 3-Rule Architecture):

> Strictly prohibit creating a separate `SPEC.md` file. All problem formulations and baseline plans are integrated directly into Section 1 of `walkthrough_lab_0X.md`.

### 1. Problem Formulation and Modality
- Task category: Classification (Binary/Multiclass), Regression, Segmentation, Object Detection, Sequence Modeling.
- Input data modality: Tabular DataFrame, 2D Image tensor `[B, C, H, W]`, Text token sequence `[B, Seq_Len]`.

### 2. Data Contract and Split Strategy
- Data schema, data types, and class balance ratios.
- Leakage-free data partitioning strategies:
  - Independent Tabular/Image samples -> `StratifiedKFold` (preserves class ratios).
  - Time-series data -> `TimeSeriesSplit` (strictly chronological, never shuffled).
  - Grouped entities -> `GroupKFold` (prevents entity overlap across train and val).

### 3. Baseline Model and Target Metrics
- **Baseline Model:** Minimal viable benchmark to anchor comparative progress (e.g. Heuristic, Logistic Regression, or simple 2-layer MLP).
- **Primary Metric:** Macro F1-Score, ROC-AUC, mAP, IoU, RMSE.
- **Acceptance Threshold:** Minimum quantitative margin the advanced model must achieve over baseline to be deemed acceptable.

### 4. Loss Function and Optimization Design
- Loss function selection:
  - Standard multi-class classification -> `nn.CrossEntropyLoss(label_smoothing=0.1)`.
  - Severe class imbalance (> 1:10) -> `FocalLoss` or weighted `class_weights`.
- Optimizer (AdamW) and Learning Rate Scheduler (CosineAnnealingLR or OneCycleLR).

### 5. Execution Environment Constraints
- Compute target: Local CPU/GPU vs **Kaggle GPU (Tesla T4)**.
- Filesystem compatibility: Handling `/kaggle/input/` (Read-Only) and `/kaggle/working/` (Writable).
- Memory constraints: Enforcing VRAM budgets (< 15GB, AMP FP16 enabled).

---

## Anti-Rationalization
- [X] *"This lab exercise is short, implement code immediately"* -> **Rejected:** Implementing without a spec forfeits a baseline benchmark, risks selecting inappropriate metrics, and degrades final report rigor.

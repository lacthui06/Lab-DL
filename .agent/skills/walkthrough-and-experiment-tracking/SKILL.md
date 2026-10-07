---
name: walkthrough-and-experiment-tracking
description: Automatically track, record, and update full-flow experimental lineages in Section 2 of walkthrough_lab_0X.md following each training run. Capture architectural, hyperparameter, and metric deltas across all 5 pipeline stages.
---

# walkthrough-and-experiment-tracking

## Overview
This skill operates as the automated experiment lineage tracker for all deep learning coursework labs. After each training execution (Run #1 Baseline, Run #2 Tuned MLP, Run #3 FashionCNN), this skill mandates that agents document **changes across all 5 pipeline stages** directly in **Section 2 of the single living document `walkthrough_lab_0X.md`** (governed by Rule 2 in the 3-Rule Architecture).

> Standard: All experimental logs and hyperparameter adjustment rationales are consolidated directly into Section 2 of `walkthrough_lab_0X.md`. Do not create separate changelog files.

## When to Activate
- Following completion of any training run when evaluation metrics and loss values are produced.
- Whenever modifying any component of the pipeline (data transforms, backbone architecture, optimizer schedules).
- When synthesizing metrics and figures for coursework submission.

---

## The 4-Step Tracking Protocol

### Step 1: Capture Full-Flow State
Record the configuration snapshot for the current execution (`Run #X`):
1. **Data:** Dataset name, sample count, Train/Val/Test split ratios.
2. **Preprocessing & Augmentation:** Applied transformations (`RandomHorizontalFlip`, `Normalize`, etc.).
3. **Model:** Architecture name, total parameter count, depth/layers.
4. **Training Strategy:** Loss function, optimizer, learning rate, scheduler, batch size, epoch count.
5. **Evaluation Metrics:** Train Loss/Acc, Val Loss/Acc, Test Loss/Acc, Macro F1-Score, execution duration.

### Step 2: Log Technical Delta
Answer 3 mandatory questions in the **Changelog**:
- *Which pipeline stages changed compared to the previous run?* (Data, Preprocessing, Architecture, or Training Strategy?)
- *What is the technical rationale (Why)?* (Mitigating overfitting, accelerating convergence, or resolving gradient vanishing?)
- *What is the quantitative impact?* (Absolute accuracy gain, reduction in generalization gap, loss delta?)

### Step 3: Update Comparison Matrix
- Append an updated column for `Run #X` in the **Flow Comparison Matrix** in `walkthrough_lab_0X.md`.
- Explicitly highlight which candidate holds the **Best Checkpoint** record.

### Step 4: Kaggle Artifact Harvesting and Findings Synthesis
- **Harvesting from Kaggle:** When running on Kaggle GPU:
  - Notebook writes output artifacts to `/kaggle/working/outputs/`:
    - Model checkpoint: `outputs/checkpoints/best_model.pt`
    - Prediction grid (Input vs Predicted): `outputs/predictions_best.png`
    - Confusion matrix: `outputs/confusion_matrix.png`
    - Training curves: `outputs/loss_curves.png`
  - Synchronize via GitHub or download `outputs.zip` back to local `labs/lab0X/outputs/`.
- **Findings Synthesis:**
  - Read output artifacts and log verified empirical numbers into `walkthrough_lab_0X.md`.
  - Extract the key breakthrough factor, error slice patterns, and paths to saved model weights.

## Verification Criteria
- `walkthrough_lab_0X.md` is synchronized with exact empirical numbers, free of placeholder blanks or unverified claims.
- The living document provides an end-to-end audit trail ready for submission.

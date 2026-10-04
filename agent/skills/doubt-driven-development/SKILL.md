---
name: doubt-driven-development
description: Adversarial auditing mechanism activated when Machine Learning or Deep Learning models achieve suspiciously high performance (Accuracy > 98%), specifically detecting target leakage, train-test contamination, and latent overfitting.
---

# doubt-driven-development (DS & AI Edition)

## Overview
In AI and Machine Learning, the most dangerous failure mode is **False Confidence**. When a model achieves 99.9% accuracy on the first attempt, it almost certainly indicates a silent pipeline defect rather than algorithmic superiority. This skill forces the AI agent to act as a skeptical auditor, seeking to invalidate the experimental setup before accepting the reported performance.

## When to Activate
- Model achieves suspiciously high Accuracy, F1-Score, or ROC-AUC (>95%) during early training iterations.
- Prior to finalizing coursework submissions or production deployments.
- When validation metrics appear unrealistically optimal despite noisy input domains.

## The 4-Step Skeptical Audit Protocol

### Step 1: Data Leakage Audit
- **Target Leakage:**
  - Are any features in $X$ generated *after* target event $y$ occurred in the real-world timeline?
  - Does any identifier column (ID, row hash) correlate 1-to-1 with the target label?
- **Split Contamination:**
  - Was `fit_transform()` executed on the entire dataset *before* `train_test_split()`? (If yes -> SEVERE LEAKAGE: test statistics contaminated the training set).
- **Duplicate Sample Leakage:**
  - Are identical rows or duplicated images present in both training and test partitions?

### Step 2: Class Imbalance Trap Audit
- Inspect the **Confusion Matrix**:
  - In a dataset with 99 negative samples and 1 positive sample, a trivial majority classifier achieves 99% accuracy!
  - Mandatory checks: Macro F1-Score, Minority Class Precision and Recall, and Precision-Recall AUC (PR-AUC).

### Step 3: Overfitting and Generalization Gap Audit
- Compare learning curves:
  - Train Loss = 0.02 while Val Loss = 1.20 -> The network is memorizing training samples and failing to generalize.

### Step 4: Random Permutation Sanity Check
- **Label Permutation Test:**
  - Randomly shuffle the target column $y$ and retrain the model.
  - **Expected Outcome:** Performance drops to random chance (e.g. ~50% for binary classification, ~10% for 10-class FashionMNIST).
  - **Red Alert:** If the model still achieves high accuracy on randomly permuted labels -> Pipeline contains 100% confirmed data leakage bugs.

## Completion Criteria
- Model results are accepted as valid only after surviving all 4 audit checks without detecting leakage or evaluation artifacts.

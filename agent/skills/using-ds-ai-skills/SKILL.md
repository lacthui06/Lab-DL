---
name: using-ds-ai-skills
description: Orchestrate and route Data Science, Machine Learning, and Deep Learning tasks to the appropriate specialized skills and corresponding workflow playbooks within Lab DL.
---

# using-ds-ai-skills

## Overview
This skill acts as the master conductor, routing all tasks across Data Science, Machine Learning, and Deep Learning to standardized engineering skills and directly linking to domain playbooks in [`workflows/`](../../workflows/) and [`dl_workflows/`](../../dl_workflows/).

## When to Use
- Starting a new DS/AI development session.
- The user provides an exercise, dataset, or model training request.
- Deciding the immediate next step in the project lifecycle.

## Standard Execution Flow
1. **Task Classification:**
   - Tabular datasets -> Route to `workflows/ML_flow.md`.
   - Computer Vision / NLP / Deep Learning -> Route to `dl_workflows/DL_flow.md`.
2. **Initialization and Specification:**
   - Activate `idea-refine` / `interview-me` if requirements are underspecified.
   - Activate `spec-driven-development` to construct the technical specification and baseline setup in Section 1 of `walkthrough_lab_0X.md`.
3. **Implementation and Testing:**
   - Activate `planning-and-task-breakdown` to partition modular source code in `src/` and interactive notebooks in `notebooks/`.
   - Activate `test-driven-development` to verify data schemas and execute local 1-batch sanity overfit tests (~5 seconds).
4. **Execution and Harvesting Protocol (Zero Waste Protocol):**
   - **Locally:** Strictly prohibit executing full multi-epoch training runs on local CPU. Halt once modules and 1-batch sanity tests pass.
   - **Kaggle GPU:** Direct execution to Kaggle GPU (Tesla T4) for accelerated training. Notebook saves checkpoints (`best_model.pt`) and visual figures (predictions, confusion matrix, loss curves) to `/kaggle/working/outputs/`.
   - **Harvest:** Synchronize or download artifacts back to local `outputs/`.
5. **Documentation and Submission:**
   - Activate `walkthrough-and-experiment-tracking` to record all run iterations in Section 2 of `walkthrough_lab_0X.md`.
   - Activate `doubt-driven-development` to audit for data leakage before triggering `documentation-and-adrs` to finalize Section 3 of `walkthrough_lab_0X.md`.

## Verification Criteria
- Every execution phase aligns with the appropriate workflow playbook and produces verifiable records in the living walkthrough document.

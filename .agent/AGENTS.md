# AGENTS.md — Operational Constitution and Execution Protocols for AI Agents in DS & AI

This document serves as the mandatory technical constitution that all AI Coding Agents (Antigravity, Claude, Cursor, Copilot) must strictly follow when working on Data Science, Machine Learning, and Deep Learning projects within `Lab DL`.

---

## 1. Core Principles

1. **Spec & Baseline First:** Never implement or train complex architectures without first specifying a simple baseline model and defining clear evaluation metrics.
2. **Deep Workflow Playbook References:**
   - For Tabular ML problems: Mandatorily reference [`workflows/ML_flow.md`](../workflows/ML_flow.md) and [`workflows/eda_guide.md`](../workflows/eda_guide.md).
   - For Deep Learning (CV, NLP, Audio): Mandatorily reference [`dl_workflows/DL_flow.md`](../dl_workflows/DL_flow.md), [`dl_workflows/dl_data_and_augmentation_guide.md`](../dl_workflows/dl_data_and_augmentation_guide.md), and [`dl_workflows/dl_training_and_optimization_guide.md`](../dl_workflows/dl_training_and_optimization_guide.md).
3. **Strict Reproducibility:** Always fix random seeds (`random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`) across all pipeline executions.
4. **Mandatory 1-Batch Sanity Overfit Test:** Before submitting training jobs to remote GPU environments (e.g. Kaggle), the model must pass a fast local 1-batch overfit test driving loss close to zero (~5 seconds).
5. **Zero Waste Protocol:** Strictly prohibit executing full training runs with multiple epochs on local CPU machines. Doing so wastes time and user token quota. Local execution stops at module completion in `src/`, interactive notebook construction in `notebooks/`, and 1-batch sanity testing. Full training is executed on Kaggle GPU (Tesla T4) or equivalent compute.
6. **Single Living Walkthrough (3-Rule Consolidation):** All technical documentation for a lab must be unified into exactly ONE living document: `walkthrough_lab_0X.md`. This file maintains explicit bindings to 3 core rules:
   - **Section 1 (Spec & Baseline Setup):** Governed by `spec-driven-development` (problem statement, data contracts, baseline model, acceptance thresholds).
   - **Section 2 (Full-Flow Experiment Changelog):** Governed by `walkthrough-and-experiment-tracking` (iteration lineage, 5-stage pipeline deltas, hyperparameter adjustment rationales).
   - **Section 3 (Results & Final Technical Report):** Governed by `documentation-and-adrs` (comparative evaluation matrices, confusion matrix diagnostics, error analysis slices, conclusion findings).
   Do NOT generate separate `SPEC.md` or `REPORT.md` files.
7. **No Emojis/Icons Policy:** Strictly prohibit emojis or graphical icons in any source code (`.py`, `.sh`), `print()` statements, log formatters, docstrings, code comments, or Markdown cells within Jupyter Notebooks (`.ipynb`). All headers and status indicators must use clean, professional ASCII/UTF-8 technical text compatible across all terminal environments and CI/CD pipelines.

---

## 2. Intent-to-Skill Mapping

| User Intent | Activated Skill | Workflow Reference Document |
|---|---|---|
| Unclear requirements / Initial ideation | `idea-refine` -> `interview-me` | `DL_flow.md` Stage 1 |
| Starting a new lab / Task specification | `spec-driven-development` | Section 1 in `walkthrough_lab_0X.md` |
| Pipeline architecture / Task breakdown | `planning-and-task-breakdown` | `DL_flow.md` 7-Stage Architecture |
| Implementing Dataset, DataLoader, Model | `test-driven-development` | `dl_data_and_augmentation_guide.md` |
| Verifying PyTorch 2.x / Hugging Face APIs | `source-driven-development` | Official PyTorch Documentation |
| GPU CUDA OOM, NaN loss, Kaggle environment bugs | `debugging-and-error-recovery` | `dl_training_and_optimization_guide.md` |
| Training speed optimization, AMP FP16 | `performance-optimization` | `dl_training_and_optimization_guide.md` |
| Unusually high model accuracy (>98%) | `doubt-driven-development` | Data Leakage Audit |
| Packaging model into inference service | `api-and-interface-design` | `DL_flow.md` Stage 7 |
| Code refactoring, extracting modular `src/` | `code-simplification` | Clean Architecture Guidelines |
| Logging runs, comparing experimental deltas | `walkthrough-and-experiment-tracking` | Section 2 in `walkthrough_lab_0X.md` |
| Completing lab, generating final submission report | `documentation-and-adrs` | Section 3 in `walkthrough_lab_0X.md` |

---

## 3. Anti-Rationalization

- [X] *"Implement code directly without writing a spec"* -> **Rejected:** Violates engineering protocols; leaves no baseline or acceptance criteria for comparative analysis.
- [X] *"Split into separate SPEC.md and REPORT.md files"* -> **Rejected:** Causes documentation fragmentation and duplicate technical narratives. Consolidate all 3 components (Spec, Changelog, Report) into `walkthrough_lab_0X.md`.
- [X] *"Skip 1-batch overfit testing and train directly on Kaggle"* -> **Rejected:** Wastes GPU quotas and hours of compute if silent dimension or loss bugs exist.
- [X] *"Execute full multi-epoch training on local CPU"* -> **Rejected:** Exhausts local compute resources and token quota. Stop after local 1-batch sanity verification and migrate execution to Kaggle GPU.
- [X] *"Paste raw terminal log snippets into chat instead of documenting"* -> **Rejected:** Must update all 5 pipeline stages and quantitative metric deltas in `walkthrough_lab_0X.md`.
- [X] *"Add emojis or icons to notebooks or scripts for decoration"* -> **Rejected:** Violates professional standards and introduces character encoding issues on diverse runtime environments. Use pure standard technical text only.

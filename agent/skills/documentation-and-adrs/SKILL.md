---
name: documentation-and-adrs
description: Author Model Cards, technical evaluation reports, Architecture Decision Records (ADRs), and submission documentation for Data Science and AI projects. Maps directly to Section 3 of walkthrough_lab_0X.md.
---

# documentation-and-adrs (DS & AI Edition)

## Overview
This skill handles professional technical documentation for DS and AI projects, providing industry-standard artifacts: **Model Cards**, **Comparative Evaluation Reports**, and **Architecture Decision Records (ADRs)** to guarantee full reproducibility, interpretability, and academic rigor.

## When to Use
- Concluding experimental runs and generating final coursework or project reports.
- Documenting why specific architectures, loss functions, or hyperparameter schedules were selected over alternatives.
- Synthesizing confusion matrices and failure modes into actionable error analyses.

## Documentation Deliverables for Coursework Labs (Integrated into Section 3 of `walkthrough_lab_0X.md`)

For practical labs, all quantitative evaluations, diagnostic summaries, and technical decisions are directly consolidated into **Section 3 of the single living document `walkthrough_lab_0X.md`** (governed by Rule 3 in the 3-Rule Architecture):
- **Quantitative Evaluation & Comparison Matrix:** Head-to-head comparison table across all runs (Baseline vs Iterations) covering Test Loss, Test Accuracy, and Macro F1-Score.
- **Confusion Matrix Diagnostics:** Deep dive into high-confusion class pairs (e.g. Shirt vs T-shirt/Pullover).
- **Error Analysis Slices:** Visual and statistical inspection of misclassified samples, identifying root failure causes.
- **Key Technical Findings:** Explanations for performance breakthroughs, assessing the impact of regularization, learning rate schedulers, and data augmentation.
- **Checkpoint Verification (Save/Load):** Verification confirming that reloading `best_model.pt` reproduces 100% identical inference metrics.

> Strictly prohibit creating a separate `REPORT.md` file. All final findings are integrated directly into Section 3 of `walkthrough_lab_0X.md`.

### 2. Architecture Decision Records (ADRs)
When maintaining permanent engineering projects, record major architectural choices under `docs/adr/`:
- *ADR 001:* Why Focal Loss over Binary Cross-Entropy? (Rationale: Mitigate severe 95:5 class imbalance).
- *ADR 002:* Why ResNet over Vision Transformer (ViT)? (Rationale: Small sample size of 2,000 images causing ViT overfitting, plus constrained GPU memory).

### 3. Session Handoff Reports (`HANDOFF.md`)
Concise handoff summary for subsequent development sessions:
- List of generated source modules and output artifacts.
- File path to the best verified model checkpoint (`outputs/checkpoints/best_model.pt`).
- Single-command instructions to reproduce results:
  ```bash
  python train.py --config config.yaml
  python evaluate.py --checkpoint checkpoints/best_model.pt
  ```

## Verification Criteria
- All documented metrics match 100% with empirical logs and output confusion matrices; zero hallucinated or inaccurately rounded numbers.

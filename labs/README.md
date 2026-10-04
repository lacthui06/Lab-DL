# Deep Learning Lab Management

This directory manages all practical coursework labs for Deep Learning. The standard reference structure is demonstrated in **`lab01/`**:

```text
labs/
└── lab01/                      <-- Standard reference directory layout for each lab
    ├── walkthrough_lab_01.md   <-- Unified 3-Rule living document (Spec -> Changelog -> Report)
    ├── notebooks/              <-- Interactive Jupyter Notebooks
    ├── src/                    <-- Modular, reusable Python source components
    ├── data/                   <-- Local input datasets (git-ignored)
    └── outputs/                <-- Checkpoints, evaluation logs, and visualization figures
```

---

## Operating Guide for `walkthrough_lab_0X.md` (Unified 3-Rule Living Document)

`walkthrough_lab_0X.md` integrates three core development rules into a single living document. Do not create separate `SPEC.md` or `REPORT.md` files:

- **Section 1 (Spec & Baseline Setup):** Governed by the `spec-driven-development` rule — Defines problem formulation, data contracts, baseline model, and quantitative acceptance thresholds.
- **Section 2 (Full-Flow Experiment Changelog):** Governed by the `walkthrough-and-experiment-tracking` rule — Records each experimental iteration (Run #1 Baseline, Run #2 Tuned MLP, Run #3 FashionCNN), tracking changes across all 5 pipeline stages, rationale for hyperparameter adjustments, and quantitative metric impact.
- **Section 3 (Results & Final Technical Report):** Governed by the `documentation-and-adrs` rule — Provides comparative evaluation matrices, confusion matrix diagnostics, error analysis slices, and conclusions ready for submission.

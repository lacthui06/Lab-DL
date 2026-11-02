# Deep Learning Lab Management

This directory manages all practical coursework labs for Deep Learning. The standard reference structure is demonstrated in **`lab01/`**:

```text
labs/
|-- lab01/                      <-- Standard reference directory layout for each lab
|   |-- README.md               <-- Detailed guide: objective, structure, reproduce, results
|   |-- description_lab_0X.md   <-- Original assignment brief from instructor (input, verbatim)
|   |-- walkthrough_lab_01.md   <-- Unified 3-Rule living document (Spec -> Experimental evidence -> Report)
|   |-- config/                 <-- workflow.json and publication.json for current Lab01
|   |-- notebooks/              <-- EDA/preprocessing/FE and modeling/optimizer notebooks
|   |-- src/                    <-- Modular Python subpackages (datasets, models, training, utils)
|   |-- scripts/                <-- CLI entry points (train.py, eval.py, infer.py)
|   |-- tests/                  <-- Mandatory TDD tests (test_datasets, test_models, test_training)
|   |-- data/                   <-- Local input datasets (git-ignored)
|   +-- outputs/                <-- Checkpoints, evaluation logs, EDA artifacts, and figures
```

---

## Operating Guide for `walkthrough_lab_0X.md` (Unified 3-Rule Living Document)

`walkthrough_lab_0X.md` integrates three core development rules into a single living document. Do not create separate `SPEC.md` or `REPORT.md` files:

- **Section 1 (Spec & Baseline Setup):** Governed by the `spec-driven-development` rule — Records the confirmed brief interpretation (Mục 1.0), then defines problem formulation, data contracts, baseline model, and quantitative acceptance thresholds.
- **Section 2 (Experiment Methods & Evidence):** Governed by the `walkthrough-and-experiment-tracking` rule — Records each experimental iteration (for example, Run #1 Baseline then later tuned runs), tracking changes across all 5 pipeline stages, rationale for hyperparameter adjustments, and quantitative metric impact.
- **Section 3 (Results & Final Technical Report):** Governed by the `documentation-and-adrs` rule — Provides comparative evaluation matrices, confusion matrix interpretation, decision traceability back to Mục 1.0, and conclusions ready for submission.

For Lab01, software change journals and troubleshooting logs stay local, outside Git and the report. Current report evidence is in `lab01/outputs/published/`; original runs are retained locally.

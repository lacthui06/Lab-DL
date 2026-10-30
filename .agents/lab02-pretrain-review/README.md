# Lab02 pre-training review — 2026-10-27

Internal validation evidence for the approved checkpoint/monitoring and presentation revision.
No full training or GPU benchmark is authorized in this revision. Historical run artifacts remain unchanged.

Environment: macOS, Python 3.12.7, PyTorch 2.2.2, torchvision 0.17.2, CPU only.
TensorBoard 2.21.0 and Ruff 0.17.3 installed into the existing .venv for validation.
The historical requirements pins were not replaced with local versions.

Final validation:

- Full Lab02 suite: 55 passed, 2 skipped, 71.37 seconds; see tests.xml.
- Skips: raw CIFAR-10 split/label contract and finalizer integration require absent raw data.
- Epoch resume test matches uninterrupted model tensors exactly on a small CPU model with dropout,
  shuffled DataLoader, AdamW and cosine; scalar event steps remain 1, 2, 3, 4.
- CLI head-trial resume test reaches its epoch budget without falsely completing suite selection/test/report.
- Existing tests cover gradient/unfreeze, real-model one-batch overfit and synthetic fit/search/predict/report.
- Ruff F checks, pip check and git diff --check passed.
- Modeling notebook reexecuted with the existing execution CLI: 11/11 code cells, 11 embedded PNGs,
  no errors/empty outputs, maximum 7 lines per code cell. EDA notebook source/output untouched.
- Visually inspected revised parameter allocation, LR search, class heatmaps and embedded TensorBoard stages.
- No changes under labs/lab02/outputs/runs; historical metrics/snapshots/manifests preserved.

Initial failures and recovery:

- Missing tensorboard dependency: installed into .venv; no historical requirements pins modified.
- macOS GUI Matplotlib backend aborted headless pytest: reran with MPLBACKEND=Agg.
- Historical Windows hash-manifest paths failed on macOS: normalized separators during lookup,
  preserving manifest keys and snapshot bytes; added regression assertion.
- Kernel TCP ports/report writes needed sandbox approval for notebook/JUnit execution.

Limits:

- No CUDA available; no GPU benchmark or full dataset training performed.
- Resume is at completed-epoch boundaries within one trial; suite continuation between partial trials
  remains unsupported. Runtime/config/source must match. Mid-batch resume is unsupported.
- Environment snapshot writer records installed metadata for key packages; Linux/CUDA compatibility
  and the complete intersection of historical pinned Requires-Python constraints remain unverified.
- Existing untracked Lab01 spatial_features.csv was left untouched.

## Follow-up: approved evaluation layout

- Restored LR/batch line plot at the user's request; actual trial values unchanged.
- Summary plot/table includes loss, accuracy, macro precision, macro recall/TPR and macro F1.
- Removed duplicate recall/FPR bar chart from notebook rendering; retained the existing legacy PNG.
- Per-class heatmap uses Precision, Recall/TPR, F1, FPR; ROC/PR follows confusion/prediction examples.
- ROC/PR uses saved fine-tune PNG/CSV, averaging per-class AUC/AP scores only; no fresh inference.
- Timing moved to a separate section after ROC/PR.
- Reexecuted modeling notebook: 13/13 code cells, 12 embedded PNGs, no empty/error outputs, max 7 lines.
- Targeted notebook-review/cleanup suite: 24 passed, 1 raw-data-dependent skip, 34.23 seconds.
- Ruff F checks and git diff --check passed; original outputs/runs files remain untouched.
- Full training suite was not rerun for these presentation-only changes.

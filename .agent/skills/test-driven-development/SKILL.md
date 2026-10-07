---
name: test-driven-development
description: Apply specialized Test-Driven Development (TDD) protocols to Data Science and Deep Learning. Direct alignment with dl_workflows/dl_data_and_augmentation_guide.md to verify data integrity, leakage prevention, tensor dimensions, and 1-batch sanity overfitting.
---

# test-driven-development (DS & AI Edition)

## Overview
This skill adapts TDD principles to the stochastic nature of machine learning and deep learning. In deep learning engineering, automated testing is **the primary safeguard against silent bugs** (such as train-val contamination, silent tensor broadcasting errors, GPU memory leaks, or non-converging architectures).

## Workflow References
When implementing tests, agents **MUST** apply technical standards from:
- [`dl_workflows/dl_data_and_augmentation_guide.md`](../../dl_workflows/dl_data_and_augmentation_guide.md): File integrity scanning, lazy-loading verification, and DataLoader batch shape assertions.
- [`dl_workflows/dl_training_and_optimization_guide.md`](../../dl_workflows/dl_training_and_optimization_guide.md): Gradient flow assertions, clipping checks, and 1-batch sanity overfit testing.

---

## The 5 Mandatory Deep Learning Tests

### 1. Data Integrity and Leakage Prevention Test
- Scan and assert zero corrupted image or tabular files before feeding data into loaders.
- Verify partition disjointness: $\text{Train} \cap \text{Val} = \emptyset$ (no sample or identity overlap).
- Assert preprocessing fit boundaries: Preprocessing scalers or encoders must call `fit()` exclusively on the training partition.

### 2. Custom Dataset Lazy-Loading Test
- Assert that `Dataset.__init__` only stores index arrays or file paths, **never buffering raw image matrices directly into host RAM**.
- Access `dataset[0]` to verify types:
  ```python
  image, label = dataset[0]
  assert isinstance(image, torch.Tensor), "Output must be a Tensor"
  assert image.dtype == torch.float32, "Image tensor must be float32"
  ```

### 3. Tensor Dimension and Shape Test
- Pass a synthetic dummy batch through the forward path to catch dimension mismatches before full data iterations:
  ```python
  def test_forward_shape():
      model = build_model(num_classes=10)
      dummy = torch.randn(2, 1, 28, 28)
      out = model(dummy)
      assert out.shape == (2, 10), f"Incorrect output shape: {out.shape}"
  ```

### 4. Learning Capacity — 1-Batch Sanity Overfit Test (CRITICAL)
- Isolate a single mini-batch (4 to 8 samples) and train the model for 30 to 50 iterations on that exact batch.
- **Mandatory Criteria:** The training loss **MUST DROP CLOSE TO ZERO ($\approx 0.00$)** and batch accuracy must reach **100%**.
- *Rule:* If a model cannot overfit 8 samples, the architecture, loss formulation, or optimizer contains severe defects. **STRICTLY PROHIBIT REMOTE CLOUD/KAGGLE TRAINING** until this test passes.

### 5. Environment Path Compatibility Test
- Assert that dynamic paths adapt seamlessly when executing in local vs Kaggle environments (`/kaggle/input/` and `/kaggle/working/`).

---

## Anti-Rationalization
- [X] *"Deep Learning models take too long to run, skip overfit testing and train directly on Kaggle"* -> **Rejected:** Training for hours on Kaggle only to discover the model never converges exhausts weekly GPU quotas. A 5-second local overfit verification is mandatory.

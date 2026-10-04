---
name: debugging-and-error-recovery
description: Diagnose and resolve specialized failure modes in Machine Learning and Deep Learning training loops. Direct alignment with dl_workflows/dl_training_and_optimization_guide.md to handle CUDA OOM, NaN/Inf loss, Kaggle filesystem permissions, and vanishing gradients.
---

# debugging-and-error-recovery (DS & AI Edition)

## Overview
This skill provides a 5-step triage protocol and diagnostic handbook for resolving classic failure modes encountered during deep learning training on local workstations and cloud platforms like Kaggle.

## Workflow References
When encountering technical anomalies, agents **MUST** consult and apply standard remedies from:
- [`dl_workflows/dl_training_and_optimization_guide.md`](../../dl_workflows/dl_training_and_optimization_guide.md): GPU VRAM management, Automatic Mixed Precision (AMP FP16), gradient clipping, and early stopping protocols.

---

## Technical Diagnostic Handbook for Deep Learning & Kaggle

### 1. GPU Memory Exhaustion (`RuntimeError: CUDA out of memory`)
- **Technical Resolutions:**
  1. Reduce `batch_size` (e.g. from 64 to 32 or 16) and compensate with **Gradient Accumulation** over $N$ steps to preserve effective batch dynamics.
  2. Enable Automatic Mixed Precision: `torch.amp.autocast('cuda', dtype=torch.float16)` -> Reduces VRAM by approximately 50%.
  3. Prevent computational graph leaks: Accumulate scalars with `total_loss += loss.item()` rather than `total_loss += loss`.
  4. Explicitly clear cached memory: `torch.cuda.empty_cache()`.

### 2. Loss Instability (Loss = NaN or Inf)
- **Technical Resolutions:**
  1. Introduce gradient clipping before optimizer updates:
     ```python
     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
     ```
  2. Validate input tensors: Execute `assert not torch.isnan(inputs).any()`.
  3. Prevent zero-division or $\log(0)$ instability via numeric clamping: `torch.clamp(x, min=1e-7)`.
  4. Reduce learning rate by an order of magnitude (e.g. from $10^{-2}$ to $10^{-3}$ or $10^{-4}$).

### 3. Kaggle-Specific Runtime Errors
- **Error 1 (Bus error - DataLoader killed abruptly):**
  - *Root Cause:* High `num_workers` exhausting the `/dev/shm` shared memory partition of Kaggle VMs.
  - *Resolution:* Restrict `num_workers = 2` or `num_workers = 0`.
- **Error 2 (Permission denied - Read-only file system):**
  - *Root Cause:* Attempting to create or write directories under `/kaggle/input/`.
  - *Resolution:* Direct all write targets, model checkpoints, and logs to `/kaggle/working/`.
- **Error 3 (Connection refused downloading Pretrained weights):**
  - *Root Cause:* Internet access disabled in Kaggle notebook settings.
  - *Resolution:* Enable `Settings -> Internet -> Internet on` in the Kaggle notebook interface.

---

## 5-Step Triage Protocol
1. **Reproduce:** Re-run on a single mini-batch to isolate the failure under minimal state.
2. **Localize:** Determine whether the root cause originates from DataLoader, Model Forward, Loss, or Backward steps.
3. **Isolate:** Print tensor shapes and assert `torch.isnan().any()` at suspected module boundaries.
4. **Fix:** Implement robust structural corrections rather than temporary workarounds.
5. **Guard:** Add automated test assertions or pytest cases to prevent regressions.

---
name: source-driven-development
description: Ground all AI and Data Science implementation decisions in official library documentation (PyTorch, Hugging Face, Scikit-learn, Torchvision). Strictly eliminate deprecated patterns and hallucinated keyword arguments.
---

# source-driven-development (DS & AI Edition)

## Overview
Deep learning and AI frameworks evolve rapidly with frequent API breaking changes. This skill mandates that AI agents verify implementations against up-to-date official documentation, eliminating deprecated functions and hallucinated arguments.

## When to Use
- Implementing pipelines with modern frameworks: `PyTorch 2.x`, `Torchvision`, `Hugging Face Transformers`, `Scikit-learn`, `Polars`.
- Configuring advanced training features: `torch.amp.autocast`, `DistributedDataParallel`, `BitsAndBytes` quantization, Learning Rate Schedulers.

## Enforcement Rules

### 1. Verify API Versions and Syntax
- Always adopt modern framework standards:
  - Use `torch.amp.autocast('cuda')` rather than legacy `torch.cuda.amp.autocast()`.
  - Use `from torchvision.transforms.v2 import ...` rather than deprecated v1 transforms.
  - Use `optimizer.zero_grad(set_to_none=True)` to optimize GPU memory rather than `optimizer.zero_grad()`.

### 2. Document Sources in Implementation Comments
- When applying non-trivial hyperparameters or complex scheduling logic, reference the official documentation in docstrings:
  ```python
  # PyTorch official reference: torch.optim.lr_scheduler.CosineAnnealingLR
  # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
  ```

### 3. Strictly Prohibit Hallucinated Arguments (No Hallucinated Kwargs)
- If uncertain whether a function accepts a specific keyword argument, inspect its signature via `inspect.signature` or verify with a short one-line script before embedding it in production modules.

## Anti-Rationalization
- [X] *"I recall this parameter worked in PyTorch back in 2021"* -> **Rejected:** Modern releases (PyTorch 2.x, Transformers 4.x) have restructured function signatures. Always use current official syntax.

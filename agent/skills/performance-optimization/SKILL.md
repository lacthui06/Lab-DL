---
name: performance-optimization
description: Optimize computational throughput, data ingestion pipelines, GPU utilization, and memory consumption across Data Science, Machine Learning, and Deep Learning workloads.
---

# performance-optimization (DS & AI Edition)

## Overview
This skill focuses on diagnosing and eliminating computation, memory, and I/O bottlenecks. In deep learning engineering, systematic optimization reduces training turnaround from days to hours and allows training larger architectures within constrained GPU budgets.

## When to Use
- Data loading bottlenecks leaving GPU compute underutilized (GPU Utilization < 50%).
- GPU VRAM constraints preventing effective batch sizing.
- High inference latency exceeding production service-level objectives (SLOs).

## The 4 Optimization Pillars in DS & AI

### 1. Data Pipeline and Host CPU Optimization
- **Vectorized Computation:** Replace Python `for` loops and Pandas `.apply()` calls with vectorized **NumPy** routines or high-performance engines like **Polars**.
- **PyTorch `DataLoader` Tuning:**
  - Configure `num_workers = 2` or `4` to enable parallel background preprocessing.
  - Set `pin_memory = True` for faster host-to-device page-locked memory transfers.
  - Adopt high-throughput columnar storage formats (`Parquet` or `Feather`) over plain `.csv`.

### 2. GPU Accelerated Training Optimization
- **Automatic Mixed Precision (AMP FP16/BF16):**
  - Combine 16-bit and 32-bit floating-point arithmetic to reduce VRAM requirements by ~50% and double throughput on Tensor Cores:
  ```python
  scaler = torch.amp.GradScaler('cuda')
  for inputs, labels in dataloader:
      optimizer.zero_grad(set_to_none=True)
      with torch.amp.autocast('cuda'):
          outputs = model(inputs)
          loss = criterion(outputs, labels)
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
  ```
- **Kernel Fusion with `torch.compile()`:** On PyTorch 2.0+, optimize execution graphs with `model = torch.compile(model)`.

### 3. Footprint Reduction (Quantization & Pruning)
- Apply post-training quantization to 8-bit or 4-bit precision (via `bitsandbytes` or PyTorch Quantization).

### 4. Inference Runtime Optimization
- **Disable Autograd Engine:** Enclose all inference passes within `with torch.inference_mode():` (more efficient than `torch.no_grad()`).
- **Export to Optimized Runtimes:** Export trained PyTorch weights to **ONNX** or **TensorRT** for serving via `onnxruntime`.

## Verification Criteria
- Measured throughput (samples/second) shows measurable improvement, and GPU compute utilization remains consistently high (>85%).

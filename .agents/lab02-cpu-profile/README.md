# Lab02 Mac Intel CPU calibration

User explicitly approved the exception to the repository CPU training rule for Lab02,
with a three-hour budget. This phase authorizes bounded profiling only; a concrete full
training configuration must still be presented for review before the full experiment.

Calibration: real CIFAR-10 train/validation subsets (640/160), seed 42, RGB 64x64,
four CPU threads, batch 64, ResNet18 and DenseNet121 pretrained ImageNet weights.
No test evaluation, no production config changes, no selected_run registry changes.

Measurements: train/validation feature extraction including frozen-prefix cache creation,
head-only sanity on eight actual feature vectors, one head epoch, one last-block fine-tune
epoch with cache and one via full image forward, validation inference throughput.
Fine-tuning policies start from identical weights. Cached/direct outputs are checked before
training; finite nonzero gradients in the unfrozen backbone are required.

The profiling script defaults to a ten-minute alarm after it starts. Any trial quality
metrics are diagnostic only and cannot replace final training/test results.

Preliminary implementation sanity: two existing real-model two-block one-batch overfit tests
passed in 20.85 seconds. Those tests use random initialization and synthetic data;
they do not establish pretrained CIFAR-10 performance or throughput.

Assets: initial sandbox networking failed; approved network access exposed a local Python
certificate issue. The installed certifi CA bundle is used via SSL_CERT_FILE, without
disabling HTTPS verification. Single-stream CIFAR download was interrupted because its
measured speed was about 45–50 KB/s; the existing official-checksum chunked downloader is used.

Profiling status: completed on official CIFAR-10 and downloaded ImageNet weights.
Dataset archive verified by official MD5; weight downloader used check_hash=True.
Both profiling jobs completed without timeout: batch 64 in 46.84 seconds, batch 128 in 52.08 seconds.

## Measured values

Each fit epoch includes 640 train and 160 validation samples; four CPU threads, RGB 64x64.
Extraction uses feature_batch_size=64 independently of the fit batch size. Fit wall time
includes validation, logging and checkpoint writes. Single observations, not statistically stable benchmarks.

| Model | Fit batch | Train+val extraction (s) | Head epoch (s) | Cached last-block epoch (s) | Full-image last-block epoch (s) |
|---|---:|---:|---:|---:|---:|
| ResNet18 | 64 | 2.954 | 0.046 | 3.509 | 7.646 |
| ResNet18 | 128 | 4.176 | 0.080 | 4.823 | 5.810 |
| DenseNet121 | 64 | 7.953 | 0.043 | 2.179 | 9.508 |
| DenseNet121 | 128 | 8.217 | 0.039 | 3.558 | 10.010 |

Head sanity on eight actual pretrained feature vectors passed for both models, both runs:
ResNet final loss 0.001062, DenseNet final loss 0.006667, accuracy 100%.
This is head-only sanity, not evidence of generalization. The earlier synthetic tests cover
real-model two-block overfit. Real-data last-block fits also checked finite nonzero backbone
gradients and cached/direct equivalence before updates. No test-set inference was performed.

Full diagnostic results (including sample-only quality metrics):
measurements/timings.json and measurements_batch128/timings.json.
These remain separate from labs/lab02/outputs/runs and do not update selected_run.json.

## Reviewable proposed control — not launched

See control_proposed.json. This draft differs explicitly from active GPU configs:
CPU / 64x64, full split 45000/5000/10000, head LR {0.001,0.01} x batch {64,128},
10 head epochs per trial, one fine-tune candidate with 10 fixed epochs, no scheduler,
same frozen-prefix cache and BatchNorm policy for both models. Not an augmentation/depth/optimizer ablation.
Ten epochs do not establish convergence. Choosing batch by validation is distinct from
claiming batch 128 is faster; cached fine-tune batch 128 was slower in these small samples.

Projection formula per epoch:

    train_pass_seconds * 45000 / 640
    + validation_seconds * 5000 / 160
    + measured per-fit non-pass overhead

Extraction scales train/validation separately. Head-grid estimate sums both measured
batch policies x two learning rates x ten epochs. Fine-tune estimate uses the slower
cached epoch projection across batch policies x ten epochs. Test cost estimates two
full test passes (head and fine) per model using measured validation inference throughput.

| Component | ResNet18 projected minutes | DenseNet121 projected minutes |
|---|---:|---:|
| Feature/prefix extraction | 4.10 | 8.55 |
| Four head trials, ten epochs each | 1.09 | 0.86 |
| One fine-tune trial, ten epochs | 38.20 | 33.78 |
| Two test passes | 1.29 | 3.21 |
| Total | 44.68 | 46.40 |

Combined point estimate: 91.08 minutes; multiplying by 1.5 gives 136.61 minutes.
This is a planning margin, not a guarantee or a confidence interval. It excludes download,
report/notebook export, possible larger-cache memory pressure, and thermal throttling.
Reserve the remainder of the three-hour budget for preparation, output and contingencies.
Do not launch the full experiment until the user approves this concrete control. At execution,
enforce remaining wall-clock budget explicitly; the draft config itself has no timeout.

Validation: script imports without starting work; Ruff F checks passed. Real calibration
completed for both batches and models. No product source or active config modified in this phase.
Hardware-profile data are scoped to this Mac/environment; do not transfer timings from the old Windows run.

# DS & AI Workflow Framework

Standardized engineering processes and technical capabilities designed specifically for AI Coding Agents developing Data Science, Machine Learning, and Deep Learning projects.

---

## DS & AI Development Lifecycle Overview

Unlike traditional software engineering, a DS & AI project requires rigorous protocols spanning data contracts, leakage auditing, GPU resource management, and model packaging. The framework comprises 14 core skills across 6 stages:

```text
 INITIALIZATION       SPEC & PLANNING            BUILD & TDD             DEBUG & DOUBT         OPTIMIZE & SERVE        REVIEW & REPORT
 ┌─────────────┐     ┌─────────────────────┐    ┌─────────────────┐    ┌────────────────────┐   ┌─────────────────┐    ┌─────────────────┐
 │ Ideation    │ ──> │ Spec & Architecture │ ─> │ Code & Data     │ ─> │ GPU / NaN Triage   │ ─>│ Vectorize / GPU │ ─> │ Clean & Refactor│
 │ Scoping     │     │ Task Decomposition  │    │ Tensor Tests    │    │ Leakage Audit      │   │ API Serving     │    │ Living Report   │
 └─────────────┘     └─────────────────────┘    └─────────────────┘    └────────────────────┘   └─────────────────┘    └─────────────────┘
  interview-me        spec-driven-dev            tdd-for-ai             debug-error-recovery     performance-opt        simplify
  idea-refine         planning-breakdown         source-driven-dev      doubt-driven-dev         api-interface-design   doc-adrs
                                                                                                                        walkthrough
```

---

## Catalog of 14 Specialized Skills for DS & AI

### 0. Orchestration Layer (Meta)
- `using-ds-ai-skills`: Master orchestrator coordinating the entire lifecycle, identifying problem types, and routing tasks to appropriate skills and playbook guides.

### 1. Initialization and Scoping (Define)
- `idea-refine`: Refines research ideas, course project proposals, and hackathon challenges; identifies public datasets and viable model architectures.
- `interview-me`: Conducts structured interviews to clarify requirements, class imbalance ratios, target evaluation metrics, and computing constraints.
- `spec-driven-development`: Defines technical specifications: baseline models, loss functions, target metrics (F1, AUC, RMSE), and data contracts before writing code. Maps directly to Section 1 of `walkthrough_lab_0X.md`.

### 2. Planning and Pipeline Architecture (Plan)
- `planning-and-task-breakdown`: Decomposes pipelines into atomic modules: Ingestion -> Preprocessing -> Feature Engineering -> Train -> Eval -> Serving.

### 3. Standards-Compliant Implementation (Build & Test)
- `test-driven-development`: Specialized TDD for AI/DL: data schema validation, tensor dimension assertions, leakage prevention tests, and local 1-batch sanity overfit checks.
- `source-driven-development`: Grounds implementation decisions directly in official documentation (PyTorch, Hugging Face, Scikit-learn), eliminating deprecated patterns and hallucinated arguments.

### 4. Diagnostics and Adversarial Auditing (Verify & Doubt)
- `debugging-and-error-recovery`: Deep learning technical troubleshooting: CUDA Out-of-Memory handling, NaN/Inf loss resolution, exploding/vanishing gradients, and Kaggle-specific execution errors.
- `doubt-driven-development`: Adversarial auditing: When a model achieves suspiciously high accuracy (>98%), rigorously inspects for target leakage, split contamination, and label leakage.

### 5. Optimization and Deployment (Optimize & Serve)
- `performance-optimization`: Maximizes throughput: NumPy vectorization, GPU DataLoader tuning (`num_workers`, `pin_memory`), and Automatic Mixed Precision (AMP FP16).
- `api-and-interface-design`: Packages trained models into production inference services using FastAPI or TorchServe with structured request/response schemas.

### 6. Code Quality, Lineage Tracking, and Submission (Review & Ship)
- `code-simplification`: Refactors exploratory notebook code into clean, modular, maintainable Python source packages in `src/`.
- `walkthrough-and-experiment-tracking`: Tracks the full-flow lineage of each experimental iteration (hyperparameters, architecture changes, quantitative deltas) directly in Section 2 of `walkthrough_lab_0X.md`.
- `documentation-and-adrs`: Compiles model cards, confusion matrix diagnostics, error analysis slices, and conclusions ready for submission directly in Section 3 of `walkthrough_lab_0X.md`.

---

## Usage

1. **Antigravity / IDE Integration:** Place `AGENTS.md` in the project root to enforce rules on all agent interactions.
2. **Context-Aware Activation:** Communicate naturally in English or Vietnamese; the agent routes the intent to the corresponding technical skill and workflow document automatically.

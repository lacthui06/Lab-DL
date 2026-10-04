---
name: idea-refine
description: Refine and scope research proposals, academic capstones, and hackathon challenges in Data Science and AI. Formulate viable problem definitions, identify accessible datasets, and recommend feasible model architectures.
---

# idea-refine (DS & AI Edition)

## Overview
Transforms broad concepts or open-ended prompts into concrete, technically viable Machine Learning or Deep Learning project proposals, identifying accessible open datasets and realistic model backbones.

## When to Use
- The user presents an unformed project topic (e.g., *"I want to build an AI project for healthcare or agriculture"*).
- Scoping hackathon entries or academic research proposals.
- Weighing trade-offs between classical ML baselines and deep learning approaches.

## The 4-Step Process

### Step 1: Divergent Brainstorming
- Propose 2 to 3 distinct formulations addressing the user's domain of interest.
- For each formulation, clarify:
  - Input modality (Image, Text, Tabular, Audio).
  - Target output (Discrete class label, continuous scalar, bounding box, generative text).

### Step 2: Dataset Sourcing
- Identify verified public data repositories (Kaggle, Hugging Face Datasets, PapersWithCode, UCI Machine Learning Repository).
- Alert the user if the problem requires proprietary or uncurated data that cannot be readily obtained.

### Step 3: Hardware and Compute Feasibility Assessment
- Evaluate compute requirements: Can this architecture be trained within free-tier GPU quotas (Kaggle/Colab Tesla T4 15GB VRAM)?
- Propose right-sized architectures: Avoid recommending massive foundation models for tasks solvable with ResNet or gradient-boosted trees.

### Step 4: Convergent Selection
- Present a refined proposal to the user containing:
  - Proposed project title.
  - Recommended public dataset.
  - Initial baseline model and advanced iteration candidate.

## Anti-Rationalization
- [X] *"Always select the most complex SOTA model to maximize grades"* -> **Rejected:** Overly complex architectures on small datasets cause severe overfitting and prolonged training cycles. Prioritize feasible, verifiable baselines first.

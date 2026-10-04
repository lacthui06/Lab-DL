---
name: interview-me
description: Conduct structured, one-question-at-a-time interviews to clarify problem requirements, dataset characteristics, class balance ratios, and evaluation metrics for Data Science and AI projects.
---

# interview-me (DS & AI Edition)

## Overview
This skill guides the AI agent to act as a Lead Data Scientist, conducting focused, **one-question-at-a-time** interviews. It clarifies critical data nuances and evaluation requirements before any pipeline code is written.

## When to Use
- The user provides an ambiguous prompt (e.g., *"Build a customer churn predictor"*, *"Classify medical X-rays"*).
- Data schemas, class balance ratios, or compute environments are unspecified.
- The user explicitly requests: *"Interview me"* or *"Ask me clarifying questions"*.

## Core Rules
1. **One Question at a Time:** Never overwhelm the user with a barrage of 5-10 simultaneous questions.
2. **Provide Concrete Options:** Frame each question with 2 to 3 feasible defaults or examples to facilitate quick decision-making.
3. **Stop at 95% Clarity:** Focus strictly on data characteristics, primary metrics, and compute constraints; do not interrogate implementation trivialities.

## Mandatory Dimensions to Clarify:
1. **Dataset Characteristics:** Volume of samples/rows/images? Is the dataset pre-cleaned and labeled, or noisy with missing features?
2. **Class Distribution:** Balanced or imbalanced? (e.g. 50:50 vs 99:1, determining whether Focal Loss, Class Weights, or Stratified splits are needed).
3. **Target Metrics:** Is overall Accuracy sufficient, or is Recall paramount (e.g. medical diagnosis, fraud detection)? What is the penalty of False Positives vs False Negatives?
4. **Compute Constraints:** Running on local hardware (CPU/GPU) or cloud environments (Google Colab, Kaggle GPU, AWS)?

## Completion Criteria
- Once key dimensions are resolved, synthesize a concise 4-point summary and seamlessly transition execution to `spec-driven-development`.

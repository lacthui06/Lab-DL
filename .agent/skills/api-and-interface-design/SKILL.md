---
name: api-and-interface-design
description: Design and implement production inference API services for AI and Deep Learning models using FastAPI, enforcing strict input/output validation schemas and safe model lifecycle management.
---

# api-and-interface-design (DS & AI Edition)

## Overview
This skill packages trained machine learning and deep learning models into industry-standard Web API services (typically using `FastAPI`). It enables external applications to query the model with low latency, robust error handling, and validated data payloads.

## When to Use
- Packaging a completed model into an inference API for demonstration, course project delivery, or system integration.
- Defining formal Request and Response contracts for model serving.

## Core Model Serving Principles

### 1. Load Model Weights Once at Startup (Lifespan Loading)
- **Golden Rule:** Model weights (`.pt`, `.onnx`, `.pkl`) must be loaded into memory (RAM/VRAM) **exactly once during application startup** using FastAPI lifespan context managers. Never reload weights inside individual request handlers.

### 2. Strict Schema Validation with Pydantic
- All input features and output predictions must enforce type safety and domain constraints:
  ```python
  from pydantic import BaseModel, Field
  from typing import List

  class PredictRequest(BaseModel):
      features: List[float] = Field(..., example=[5.1, 3.5, 1.4, 0.2], description="Input feature vector")

  class PredictResponse(BaseModel):
      prediction: int = Field(..., description="Predicted class index")
      class_name: str = Field(..., description="Predicted human-readable class name")
      confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence probability")
      latency_ms: float = Field(..., description="Inference latency in milliseconds")
  ```

### 3. Standard Endpoint Architecture
A production model service must expose at minimum:
1. `GET /health`: Healthcheck verifying server status and confirmed in-memory model availability.
2. `GET /metadata`: Model metadata (architecture version, training date, target evaluation metric).
3. `POST /predict`: Primary endpoint receiving feature payloads and returning structured inference outputs.

### 4. Resilient Exception Handling
- Catch corrupted input data (e.g. invalid image buffers, malformed tensors) and return informative HTTP 400/422 responses rather than unhandled HTTP 500 crashes.

## Verification Criteria
- Service initializes cleanly and `GET /health` returns `{"status": "healthy", "model_loaded": true}`.
- Sample payload sent to `/predict` successfully returns a typed response within acceptable latency budgets (< 200 ms).

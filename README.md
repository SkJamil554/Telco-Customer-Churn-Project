# Telco Customer Churn — Telco-Customer-Churn-Project

A reproducible MLOps pipeline and model serving application for predicting customer churn on a telecommunications dataset. The repository provides deterministic feature engineering, an XGBoost training pipeline with MLflow experiment tracking, and a FastAPI + Gradio inference service for predictions.

Table of contents
- Overview
- Stack
- Key features
- Architecture & components
- Getting started (local)
- Training
- Serving (API & UI)
- MLflow & artifacts
- Testing
- Docker
- CI/CD notes
- Contributing & contact

## Overview
This project implements a full training and serving lifecycle for a churn prediction model using the Telco customer dataset. Training and serving use the same preprocessing and feature engineering to avoid inference drift; models and preprocessing artifacts are tracked with MLflow so runs are reproducible and portable into the serving container.

## Stack
- **Language(s):** Jupyter Notebooks (experiments), Python (implementation)
- **Framework / runtime:** FastAPI (serving), Uvicorn (ASGI), MLflow (tracking & pyfunc)
- **Notable libraries:** mlflow, xgboost, fastapi, gradio, great_expectations

## Key features
- Data validation using Great Expectations
- Deterministic feature engineering (binary mappings + one-hot with drop_first)
- XGBoost classifier with tuned hyperparameters
- MLflow experiment tracking for metrics and artifacts
- FastAPI REST endpoint with an optional Gradio UI for interactive predictions
- Containerized for deployment (Docker)

## Architecture & components
- Training pipeline: `scripts/run_pipeline.py`
  - Steps: data loading → validation → preprocessing → feature engineering → XGBoost training → MLflow logging
  - Produces artifacts: `model/` (pyfunc), `feature_columns.txt`, `preprocessing.pkl`

- Serving components:
  - FastAPI app: `src/app/main.py` (GET /, POST /predict)
  - Inference utilities: `src/serving/inference.py` (loads MLflow model, applies binary mappings and get_dummies, aligns features using `feature_columns`)
  - Gradio UI mounted at `/ui` via the FastAPI app

- Utilities:
  - Data validation: `src/utils/validate_data.py` (Great Expectations checks)
  - Feature engineering: `src/features/build_features.py`

## Getting started (local)
Prerequisites
- Python 3.11 (Dockerfile uses python:3.11-slim)
- pip, git
- (Optional) Docker

Install
```bash
git clone https://github.com/SkJamil554/Telco-Customer-Churn-Project.git
cd Telco-Customer-Churn-Project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training
- End-to-end training and MLflow logging:
```bash
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn
```
- Prepare processed data only:
```bash
python scripts/prepare_processed_data.py
```

Model config highlights (defaults from config in repo)
- XGBoost: n_estimators=301, learning_rate=0.034, max_depth=7
- Class imbalance handled via dynamic scale_pos_weight
- Default classification threshold: 0.35 (configurable)

## Serving (FastAPI + Gradio)
Run locally
```bash
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000
# or alternate entry point
python -m uvicorn src.app.app:app --host 0.0.0.0 --port 8000
```
Endpoints
- GET / → health check ({"status":"ok"})
- POST /predict → accepts `CustomerData` Pydantic schema (18 features) and returns prediction + probability
- /ui → Gradio web UI for interactive prediction

Serving details
- Model is loaded via MLflow pyfunc from `/app/model` (in container) or from `src/serving/model` for local testing
- Serving applies the same BINARY_MAP used in training and uses `pd.get_dummies(..., drop_first=True)`; features are aligned to `feature_columns.txt` produced at training time

## MLflow & artifacts
- Experiment name: "Telco Churn"
- File-based MLflow store: `{project_root}/mlruns` (the repo also contains an `mlflow.db` file)
- Logged metrics: precision, recall, f1, roc_auc, train_time, pred_time, data_quality_pass
- Artifacts saved per run: `model/` (pyfunc), `feature_columns.txt`, `preprocessing.pkl`

To view MLflow UI locally:
```bash
mlflow ui --backend-store-uri file:./mlruns
```

## Testing
Provided test scripts (not a formal test suite):
```bash
python scripts/test_pipeline_phase1_data_features.py
python scripts/test_pipeline_phase2_modeling.py
python scripts/test_fastapi.py
```

## Docker
Build image
```bash
docker build -t telco-churn-app .
```
Run container (expects model artifacts copied into image or mounted at /app/model)
```bash
docker run -p 8000:8000 telco-churn-app
```
Notes: Dockerfile uses `python:3.11-slim` and sets `PYTHONPATH=/app/src` for imports. The build copies a specific MLflow run into `/app/model` for serving by default.

## CI/CD notes
- The repository includes a GitHub workflow that builds the Docker image on push to main and can push to Docker Hub (requires DOCKERHUB_USERNAME, DOCKERHUB_TOKEN secrets).
- Deployment notes reference a manual ECS Fargate update behind an ALB.

## Contributing & contact
- Raise issues for bugs or feature requests, open PRs for fixes/improvements.
- Maintain feature parity between training and serving preprocessing to avoid inference-time drift.

Maintainer: <your name> — <email@example.com>
GitHub: https://github.com/SkJamil554/Telco-Customer-Churn-Project

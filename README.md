# Telco Customer Churn — Telco-Customer-Churn-Project

A reproducible MLOps pipeline and model serving application for predicting customer churn in a telecommunications dataset. This repository includes training pipelines, data validation, feature engineering, model training with XGBoost, MLflow experiment tracking, and a FastAPI + Gradio serving layer.

Table of contents
- Overview
- Key features
- Architecture & components
- Getting started (local)
- Training pipeline
- Serving (API & UI)
- MLflow & artifacts
- Testing
- Docker
- CI/CD
- Development notes
- Contributing & license
- Contact

Overview
This project implements a full training and serving lifecycle for a churn prediction model. Training and serving use identical feature transformations and artifacts are tracked with MLflow to ensure reproducibility.

Key features
- Data validation (Great Expectations)
- Deterministic feature engineering (binary mappings, one-hot with drop_first)
- XGBoost model with tuned hyperparameters
- MLflow experiment tracking (models, feature columns, preprocessing artifacts)
- FastAPI REST endpoint + Gradio UI for inference
- Containerized for production deployment

Architecture & components
- Training pipeline: scripts/run_pipeline.py
  - Steps: Data loading → Data validation → Preprocessing → Feature engineering → XGBoost training → MLflow logging
- Serving pipeline:
  - FastAPI app: `src/app/main.py` (GET /, POST /predict)
  - Gradio UI mounted at `/ui`
  - Inference utilities: `src/serving/inference.py`
- Artifacts & storage:
  - MLflow tracking URI: file-based at `mlruns/`
  - Logged artifacts: `model/`, `feature_columns.txt` (or JSON), `preprocessing.pkl`
  - Container runtime path for model: `/app/model/`

Getting started (local)
Prerequisites
- Python 3.11 (project uses 3.11 in Dockerfile)
- pip
- git
- (Optional) Docker for containerized runs

Install
1. Clone the repo:
   git clone <repository-url>
2. Create a virtual environment and install:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

Training pipeline
- Run end-to-end training:
  python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn
- Prepare processed data only:
  python scripts/prepare_processed_data.py

Model config highlights
- XGBoost: n_estimators=301, learning_rate=0.034, max_depth=7
- scale_pos_weight is computed dynamically to handle class imbalance
- Default classification threshold: 0.35 (configurable)

Serving (FastAPI + Gradio)
Run locally:
- Start the FastAPI app:
  python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000
  # Or alternate entry point:
  python -m uvicorn src.app.app:app --host 0.0.0.0 --port 8000
- Endpoints:
  - GET / → health check ({"status":"ok"})
  - POST /predict → accepts `CustomerData` Pydantic schema with expected attributes (18 features)
  - /ui → Gradio web UI for interactive prediction

Serving details
- Model is loaded via MLflow pyfunc from `/app/model` in container environment.
- Serving uses a fixed BINARY_MAP for binary features and pd.get_dummies(..., drop_first=True) to match training.
- Features are aligned using `feature_columns.txt` produced at training time; serving enforces the same feature order.

MLflow & artifacts
- Experiment name: "Telco Churn"
- mlruns directory: `{project_root}/mlruns` (file-based tracking)
- Logged metrics: precision, recall, f1, roc_auc, train_time, pred_time, data_quality_pass
- Artifacts saved per run: `model/`, `feature_columns.txt`, `preprocessing.pkl`

Testing
- Test data processing and feature engineering:
  python scripts/test_pipeline_phase1_data_features.py
- Test model training and evaluation:
  python scripts/test_pipeline_phase2_modeling.py
- Test FastAPI endpoints:
  python scripts/test_fastapi.py

Docker
Build image:
  docker build -t telco-churn-app .
Run container:
  docker run -p 8000:8000 telco-churn-app

Notes:
- Dockerfile base image: python:3.11-slim
- The Docker build copies a specific MLflow run to `/app/model` for serving
- Ensure `PYTHONPATH=/app/src` is set for proper imports

CI/CD
- Workflow: push to main triggers CI that builds Docker image and pushes to Docker Hub (example: anasriad8/telco-fastapi:latest)
- Requires secrets: DOCKERHUB_USERNAME, DOCKERHUB_TOKEN
- Deployment: manual ECS Fargate service update behind an ALB (per project notes)

Development notes
- No formal unit test suite; rely on provided test scripts
- To view MLflow UI locally:
  mlflow ui --backend-store-uri file:./mlruns
- Keep preprocessing and feature engineering code identical between training and serving to avoid inference-time drift

Contributing
- Raise issues for bugs or feature requests.
- Follow the repository's code style and test changes using the provided test scripts.

License
- Add your chosen license here (e.g., MIT, Apache-2.0).

Contact
- Maintainer: <your name> — <email@example.com>
- GitHub: https://github.com/SkJamil554/Telco-Customer-Churn-Project

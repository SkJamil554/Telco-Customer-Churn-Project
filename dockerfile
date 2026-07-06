# 1. Use the official lightweight Python base image
FROM python:3.13-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy only dependency file first (for Docker caching)
COPY requirements.txt .

# 4. Install Python dependencies
#    (curl is included in case you use an MLflow local tracking URI / healthchecks;
#     remove the apt-get lines if you don't need it)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy the entire project into the image
COPY . .

# MLflow run ID is now a build arg instead of hardcoded, so retraining
# doesn't require editing this file by hand:
#   docker build --build-arg MLFLOW_RUN_ID=3b1a41221fc44548aed629fa42b762e0 -t myimage .
ARG MLFLOW_RUN_ID=3b1a41221fc44548aed629fa42b762e0

# Explicitly copy model (in case .dockerignore excluded mlruns)
# NOTE: destination matches inference.py's expected path
COPY src/serving/model /app/src/serving/model

# Copy MLflow run (artifacts + metadata) to the flat /app/model convenience path
COPY src/serving/model/${MLFLOW_RUN_ID}/artifacts/model /app/model
COPY src/serving/model/${MLFLOW_RUN_ID}/artifacts/feature_columns.txt /app/model/feature_columns.txt
COPY src/serving/model/${MLFLOW_RUN_ID}/artifacts/preprocessing.pkl /app/model/preprocessing.pkl

# ensures logs are shown in real-time (no buffering)
# NOTE: PYTHONPATH is intentionally NOT set to /app/src here. WORKDIR (/app)
# is already on sys.path, so the CMD below (src.app.main:app) resolves correctly.
# If you want to import modules as `from app...` instead of `from src.app...`,
# change PYTHONPATH AND the CMD's import path together - never one without the other,
# or you'll get duplicate module instances (src.app.X and app.X as separate objects).
ENV PYTHONUNBUFFERED=1

# 6. Expose FastAPI port
EXPOSE 8000

# 7. Run the FastAPI app using uvicorn
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
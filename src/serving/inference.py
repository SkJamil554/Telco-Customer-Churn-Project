"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output

CRITICAL PATTERN: Training/Serving Consistency
- Uses fixed BINARY_MAP for deterministic binary encoding
- Applies same one-hot encoding with drop_first=True
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully

Production Deployment:
- MODEL_DIR points to containerized model artifacts
- Feature schema loaded from training-time artifacts
- Optimized for single-row inference (real-time serving)
"""

import glob
import os
from pathlib import Path

import mlflow
import pandas as pd

# === MODEL LOADING CONFIGURATION ===
# IMPORTANT: This path is set during Docker container build
# In development: uses local MLflow artifacts or the local src/serving/model directory
MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")

model = None
FEATURE_COLS: list[str] = []


def _find_model_artifact() -> tuple[Path, Path]:
    """Find a local MLflow model artifact for production or development."""
    candidate = Path(MODEL_DIR)
    if candidate.exists():
        if (candidate / "MLmodel").exists():
            return candidate, candidate
        if (candidate / "artifacts" / "model" / "MLmodel").exists():
            return candidate / "artifacts" / "model", candidate
        if (candidate / "model" / "MLmodel").exists():
            return candidate / "model", candidate

    search_patterns = [
        "src/serving/model/*/artifacts/model",
        "./mlruns/*/*/artifacts/model",
    ]

    for pattern in search_patterns:
        matches = glob.glob(pattern)
        if matches:
            chosen = max(matches, key=os.path.getmtime)
            run_root = Path(chosen).parents[1]
            return Path(chosen).resolve(), run_root.resolve()

    raise FileNotFoundError(
        f"No usable MLflow model artifact found. Checked MODEL_DIR={MODEL_DIR}, "
        "src/serving/model/*/artifacts/model, and ./mlruns/*/*/artifacts/model."
    )


def _find_feature_file(run_root: Path) -> Path:
    """Locate the feature_columns.txt file for the selected model."""
    candidates = [
        run_root / "feature_columns.txt",
        run_root / "artifacts" / "feature_columns.txt",
        run_root / "artifacts" / "model" / "feature_columns.txt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    found = list(run_root.rglob("feature_columns.txt"))
    if found:
        return found[0]

    raise FileNotFoundError(
        f"Failed to locate feature_columns.txt under run root {run_root}."
    )


def _initialize_model() -> None:
    """Load model and feature metadata on first inference request."""
    global FEATURE_COLS, model

    if model is not None and FEATURE_COLS:
        return

    model_path, run_root = _find_model_artifact()
    feature_file = _find_feature_file(run_root)

    with open(feature_file, "r", encoding="utf-8") as fh:
        FEATURE_COLS = [ln.strip() for ln in fh if ln.strip()]

    model = mlflow.pyfunc.load_model(str(model_path))
    print(f"✅ Loaded model from {model_path}")
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from {feature_file}")


def _ensure_model_ready() -> None:
    if model is None or not FEATURE_COLS:
        _initialize_model()

# === FEATURE TRANSFORMATION CONSTANTS ===
# CRITICAL: These mappings must exactly match those used in training
# Any changes here will cause train/serve skew and degrade model performance

# Deterministic binary feature mappings (consistent with training)
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},           # Demographics
    "Partner": {"No": 0, "Yes": 1},               # Has partner
    "Dependents": {"No": 0, "Yes": 1},            # Has dependents  
    "PhoneService": {"No": 0, "Yes": 1},          # Phone service
    "PaperlessBilling": {"No": 0, "Yes": 1},      # Billing preference
}

# Numeric columns that need type coercion
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.
    
    This function is CRITICAL for production ML - it ensures that features are
    transformed exactly as they were during training to prevent train/serve skew.
    
    Transformation Pipeline:
    1. Clean column names and handle data types
    2. Apply deterministic binary encoding (using BINARY_MAP)
    3. One-hot encode remaining categorical features  
    4. Convert boolean columns to integers
    5. Align features with training schema and order
    
    Args:
        df: Single-row DataFrame with raw customer data
        
    Returns:
        DataFrame with features transformed and ordered for model input
        
    IMPORTANT: Any changes to this function must be reflected in training
    feature engineering to maintain consistency.
    """
    df = df.copy()
    
    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()
    
    # === STEP 1: Numeric Type Coercion ===
    # Ensure numeric columns are properly typed (handle string inputs)
    for c in NUMERIC_COLS:
        if c in df.columns:
            # Convert to numeric, replacing invalid values with NaN
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # Fill NaN with 0 (same as training preprocessing)
            df[c] = df[c].fillna(0)
    
    # === STEP 2: Binary Feature Encoding ===
    # Apply deterministic mappings for binary features
    # CRITICAL: Must use exact same mappings as training
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)                    # Convert to string
                .str.strip()                    # Remove whitespace
                .map(mapping)                   # Apply binary mapping
                .astype("Int64")                # Handle NaN values
                .fillna(0)                      # Fill unknown values with 0
                .astype(int)                    # Final integer conversion
            )
    
    # === STEP 3: One-Hot Encoding for Remaining Categorical Features ===
    # Find remaining object/categorical columns (not in BINARY_MAP)
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        # Apply one-hot encoding with drop_first=True (same as training)
        # This prevents multicollinearity by dropping the first category
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    
    # === STEP 4: Boolean to Integer Conversion ===
    # Convert any boolean columns to integers (XGBoost compatibility)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # === STEP 5: Feature Alignment with Training Schema ===
    # CRITICAL: Ensure features are in exact same order as training
    # Missing features get filled with 0, extra features are dropped
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return df

def predict(input_dict: dict) -> str:
    """
    Predict customer churn from raw customer input.
    Pipeline:
    1. Ensure the ML model and feature schema are loaded.
    2. Convert input dictionary to DataFrame.
    3. Apply the same feature engineering used during training.
    4. Generate prediction.
    5. Return a business-friendly prediction.
    """
    try:
        # Step 1: Load model and feature schema (only once)
        _ensure_model_ready()
        # Step 2: Convert request to DataFrame
        df = pd.DataFrame([input_dict])
        # Step 3: Apply training-time preprocessing
        df_enc = _serve_transform(df)
        # Step 4: Generate prediction
        prediction = model.predict(df_enc)

        # Convert numpy arrays to Python list
        if hasattr(prediction, "tolist"):
            prediction = prediction.tolist()

        # Extract scalar prediction
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]

        prediction = int(prediction)
        # Step 5: Convert prediction to readable output
        return (
            "Likely to churn"
            if prediction == 1
            else "Not likely to churn"
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Model artifacts not found: {e}")
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")
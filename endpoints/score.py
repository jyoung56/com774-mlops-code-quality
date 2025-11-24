import os
import json
from typing import Any, Dict

import joblib
import pandas as pd

# ------------------------------------------------------------------------
# FEATURE COLUMNS
# These are the exact inputs your model expects in the same order that it
# was trained on. Keeping this list explicit prevents schema drift.
# ------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "lines_of_code__scaled",
    "number_of_classes__scaled",
    "number_of_packages__scaled",
    "number_of_problematic_classes__scaled",
    "number_of_highly_problematic_classes__scaled",
    "commits_repo__scaled",
    "branches__scaled",
    "contributors__scaled",
    "stars__scaled",
    "forks__scaled",
]


# The filename AzureML automatically stores inside AZUREML_MODEL_DIR
MODEL_FILENAME = "model.joblib"


# Global model instance populated by init()
model = None  

# ------------------------------------------------------------------------
# INIT FUNCTION
# ------------------------------------------------------------------------
def init():
    """
    AzureML automatically calls this ONCE when the scoring container starts.

    Responsibilities:
    - Locate the registered model file inside the AzureML container
      using the AZUREML_MODEL_DIR environment variable.
    - Fall back to local paths for unit testing.
    - Load the model globally so that run() can use it repeatedly.

    If the model cannot be found (e.g. during local tests),
    we leave `model = None` and unit tests can inject a dummy model.

    """
    global model

    # AzureML sets this when the endpoint container starts
    model_dir = os.getenv("AZUREML_MODEL_DIR", None)

    # Paths checked in order, depending on environment
    candidates = []

    if model_dir:
        candidates.append(os.path.join(model_dir, MODEL_FILENAME))

    # Local development fallback locations
    candidates.append(os.path.join("outputs", MODEL_FILENAME))
    candidates.append(MODEL_FILENAME)

    model_path = None
    for c in candidates:
        if os.path.exists(c):
            model_path = c
            break

    if model_path is None:
        # No real model = acceptable for local tests
        print("[score] No model file found; model will need to be injected for tests.")
        model = None
        return

    print(f"[score] Loading model from: {model_path}")
    model = joblib.load(model_path)

# ------------------------------------------------------------------------
# FEATURE PREPARATION
# ------------------------------------------------------------------------
def _prepare_features(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts the incoming JSON payload into a Pandas DataFrame
    structured EXACTLY like the model's training data.

    This validates:
      - 'data' exists
      - 'data' is a list of rows
      - all expected feature columns are present

    Raises ValueError with clear messaging if the schema is invalid.

    """
    if "data" not in payload:
        raise ValueError("Request JSON must contain a 'data' field with a list of rows.")

    rows = payload["data"]
    if not isinstance(rows, list):
        raise ValueError("'data' must be a list of objects.")

    # Convert rows into a DataFrame
    df = pd.DataFrame(rows)

    # Check for schema mismatch (very important in MLOps)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required feature columns in request: {missing}")

    # Column order must match training order
    return df[FEATURE_COLUMNS]


# ------------------------------------------------------------------------
# RUN FUNCTION — MAIN SCORING ENTRYPOINT
# ------------------------------------------------------------------------
def run(raw_data: Any) -> Dict[str, Any]:
    """

    AzureML calls this function for every scoring request.

    Behaviour:
    - Accepts either a JSON string (Azure) or a Python dict (local tests)
    - Prepares features into a DataFrame
    - If a real model exists → predict labels + probabilities
    - If no real model → use dummy fallback (unit-testing behaviour)
    - Returns a JSON-serialisable dict

    This function MUST NOT crash — any exception must be caught
    and returned as {"error": "..."} so that Azure endpoint stays healthy.

    """
    try:
        # raw_data is a JSON string in production, dict in tests
        if isinstance(raw_data, str):
            payload = json.loads(raw_data)
        else:
            payload = raw_data

        # Prepare dataframe from request
        X_df = _prepare_features(payload)

        global model

        # Fallback: unit tests inject their own DummyModel, but if
        # no model exists AND no injection happened, we return zeros.
        if model is None:
            predictions = [0] * len(X_df)
            probabilities = [0.0] * len(X_df)
            return {
                "predictions": predictions,
                "probabilities": probabilities,
            }

        # -------------------------
        # REAL MODEL PREDICTION
        # -------------------------
        preds = model.predict(X_df)

         # Optional probability extraction
        if hasattr(model, "predict_proba"):
            probs_2d = model.predict_proba(X_df)

            # Binary classifier to probability of class 1
            probabilities = [row[1] for row in probs_2d]
        else:
            probabilities = [None] * len(preds)

        # Convert numpy arrays to Python lists
        if hasattr(preds, "tolist"):
            predictions = preds.tolist()
        else:
            predictions = list(preds)

        if hasattr(probabilities, "tolist"):
            probabilities = probabilities.tolist()

        return {
            "predictions": predictions,
            "probabilities": probabilities,
        }

    except Exception as e:
        # NEVER crash the endpoint — return error instead
        return {"error": str(e)}

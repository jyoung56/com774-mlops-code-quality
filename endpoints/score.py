import os
import json
from typing import Any, Dict

import joblib
import pandas as pd

# These are the feature columns your model was trained on
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

MODEL_FILENAME = "model.joblib"

model = None  # will be loaded in init()


def init():
    """
    AzureML calls this once when the container starts.
    For local tests, this will also run on first import.
    """
    global model

    # AZUREML_MODEL_DIR is set in the online endpoint container.
    # Locally, it may not exist, so we fall back to ./outputs or .
    model_dir = os.getenv("AZUREML_MODEL_DIR", None)

    candidates = []

    if model_dir:
        candidates.append(os.path.join(model_dir, MODEL_FILENAME))

    # Common local path (after training run)
    candidates.append(os.path.join("outputs", MODEL_FILENAME))
    candidates.append(MODEL_FILENAME)

    model_path = None
    for c in candidates:
        if os.path.exists(c):
            model_path = c
            break

    if model_path is None:
        # For local unit tests we’ll usually inject a DummyModel anyway,
        # so it’s fine if we don’t find a real model here.
        print("[score] No model file found; model will need to be injected for tests.")
        model = None
        return

    print(f"[score] Loading model from: {model_path}")
    model = joblib.load(model_path)


def _prepare_features(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Take the incoming JSON payload and turn it into a DataFrame
    with exactly the columns the model expects.
    """
    if "data" not in payload:
        raise ValueError("Request JSON must contain a 'data' field with a list of rows.")

    rows = payload["data"]
    if not isinstance(rows, list):
        raise ValueError("'data' must be a list of objects.")

    df = pd.DataFrame(rows)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in request: {missing}")

    return df[FEATURE_COLUMNS]


def run(raw_data: Any) -> Dict[str, Any]:
    """
    This is the entrypoint AzureML uses for scoring.
    It also works locally in tests.
    """
    try:
        # raw_data is usually a JSON string from Azure / tests
        if isinstance(raw_data, str):
            payload = json.loads(raw_data)
        else:
            # tests may pass a dict directly in future
            payload = raw_data

        X_df = _prepare_features(payload)

        # If no real model is loaded (e.g. in unit tests), we expect
        # tests to inject a dummy into the global 'model'.
        global model
        if model is None:
            # Safe fallback – but in our tests we override 'model'
            predictions = [0] * len(X_df)
            probabilities = [0.0] * len(X_df)
            return {
                "predictions": predictions,
                "probabilities": probabilities,
            }

        # Predict labels
        preds = model.predict(X_df)

        # Predict probabilities if supported
        if hasattr(model, "predict_proba"):
            probs_2d = model.predict_proba(X_df)
            # Assume binary classifier: take probability of class 1
            probabilities = [row[1] for row in probs_2d]
        else:
            probabilities = [None] * len(preds)

        # Normalise types (numpy -> Python)
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
        # Return error in a consistent JSON structure
        return {"error": str(e)}

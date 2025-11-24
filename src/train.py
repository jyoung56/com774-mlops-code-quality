import os 
import argparse
import joblib

import pandas as pd 
from azureml.core import Dataset, Run, Model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def main(args):
    run = Run.get_context()
    run.tag("run_type", f"random_forest_{args.n_estimators}_tress")
    ws = run.experiment.workspace

    # Load dataset
    dataset = Dataset.get_by_name(ws, "sonarqube_code_quality_labelled")
    df = dataset.to_pandas_dataframe()

    target_col = "high_risk"

    candidate_feature_cols = [
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

    # Add only existing columns
    feature_cols = [c for c in candidate_feature_cols if c in df.columns]

    print("Using feature columns:", feature_cols)

    if not feature_cols:
        raise ValueError("No usable feature columns found in dataframe!")

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test,y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    acc =accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    # Log Parameters
    run.log("n_estimators", args.n_estimators)
    run.log("accuracy", acc)
    run.log("f1_score", f1)
    run.log("precision", prec)
    run.log("recall", rec)
    run.log("roc_auc", roc)

    print(f"Accuracy: {acc:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"ROC_AUC: {roc:.3f}")

    # Save Model and Register
    os.makedirs("outputs", exist_ok=True)
    model_path = os.path.join("outputs", "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Upload model to run artifacts
    run.upload_file(name="outputs/model.joblib", path_or_stream=model_path)
    print("Uploaded model file to run artifacts as 'outputs/model.joblib'")

    # Register model in Azure ML Model Registry
    registered_model = run.register_model(
        model_name = "code_quality_rf_scaled",
        model_path=  "outputs/model.joblib",
        tags={
            "run-type": f"random_forest_{args.n_estimators}_trees",
            "n_estimators": str(args.n_estimators),
        },
        properties={
            "accuracy": acc,
            "f1_score": f1,
            "precision": prec,
            "recall": rec,
            "roc_auc": roc,
        },
    )

    print(f"Registered Model: {registered_model.name} v{registered_model.version}")
    run.complete()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()
    main(args)

    


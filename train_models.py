"""
train_models.py
----------------
Trains and evaluates prediction models for three diseases:
    1. Liver disease   (Indian Liver Patient Dataset)
    2. Kidney disease  (Chronic Kidney Disease Dataset)
    3. Parkinson's disease

For every disease, three algorithms are trained:
    - Logistic Regression
    - Random Forest
    - XGBoost

The best model (highest test-set F1-score) is saved along with its
StandardScaler and the ordered list of feature names, so the Streamlit
app can load them and make predictions on new, raw user input.

Run from the project root:
    python src/train_models.py
"""

import os
import json
import joblib
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from preprocess import (
    load_and_prepare_liver,
    load_and_prepare_kidney,
    load_and_prepare_parkinsons,
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def get_candidate_models():
    """Returns a fresh dict of candidate models for each training run."""
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
        )
    return models


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc_auc = float("nan")

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def train_disease_model(disease_name, X, y, feature_names):
    print(f"\n{'=' * 60}")
    print(f"Training models for: {disease_name.upper()}")
    print(f"{'=' * 60}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Class balance:\n{y.value_counts(normalize=True)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    fitted_models = {}

    for name, model in get_candidate_models().items():
        model.fit(X_train_scaled, y_train)
        metrics = evaluate_model(model, X_test_scaled, y_test)

        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, cv=5, scoring="f1"
        )
        metrics["cv_f1_mean"] = float(np.mean(cv_scores))

        results[name] = metrics
        fitted_models[name] = model

        print(f"\n-- {name} --")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1-score : {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
        print(f"  CV F1(5) : {metrics['cv_f1_mean']:.4f}")
        print(f"  Confusion Matrix: {metrics['confusion_matrix']}")

    # Pick the best model by test F1-score
    best_name = max(results, key=lambda k: results[k]["f1_score"])
    best_model = fitted_models[best_name]
    print(f"\n>>> Best model for {disease_name}: {best_name} "
          f"(F1={results[best_name]['f1_score']:.4f})")

    # Persist model + scaler + feature order + metrics
    joblib.dump(best_model, os.path.join(MODEL_DIR, f"{disease_name}_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{disease_name}_scaler.pkl"))

    with open(os.path.join(MODEL_DIR, f"{disease_name}_features.json"), "w") as f:
        json.dump(feature_names, f)

    with open(os.path.join(MODEL_DIR, f"{disease_name}_metrics.json"), "w") as f:
        json.dump(
            {"best_model": best_name, "all_results": results}, f, indent=2
        )

    return best_name, results[best_name]


def main():
    summary = {}

    # 1. Liver
    X, y, feats = load_and_prepare_liver(
        os.path.join(DATA_DIR, "indian_liver_patient.csv")
    )
    summary["liver"] = train_disease_model("liver", X, y, feats)

    # 2. Kidney
    X, y, feats = load_and_prepare_kidney(
        os.path.join(DATA_DIR, "kidney_disease.csv")
    )
    summary["kidney"] = train_disease_model("kidney", X, y, feats)

    # 3. Parkinson's
    X, y, feats = load_and_prepare_parkinsons(
        os.path.join(DATA_DIR, "parkinsons.csv")
    )
    summary["parkinsons"] = train_disease_model("parkinsons", X, y, feats)

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE — SUMMARY")
    print(f"{'=' * 60}")
    for disease, (best_name, metrics) in summary.items():
        print(
            f"{disease:12s} | best={best_name:18s} | "
            f"acc={metrics['accuracy']:.3f} | f1={metrics['f1_score']:.3f} | "
            f"roc_auc={metrics['roc_auc']:.3f}"
        )
    print(f"\nModels saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()

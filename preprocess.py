"""
preprocess.py
--------------
Data loading and preprocessing utilities for the Multiple Disease
Prediction project (Liver, Kidney, Parkinson's datasets).

Each `load_and_prepare_*` function returns:
    X (pd.DataFrame) : cleaned / encoded feature matrix
    y (pd.Series)    : target labels (1 = disease present, 0 = healthy)
    feature_names (list) : ordered list of feature columns
"""

import pandas as pd
import numpy as np


# --------------------------------------------------------------------------
# 1. LIVER DISEASE (Indian Liver Patient Dataset)
# --------------------------------------------------------------------------
def load_and_prepare_liver(path: str):
    df = pd.read_csv(path)

    # Encode Gender: Male -> 1, Female -> 0
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    # Fill missing numeric values with column median
    df["Albumin_and_Globulin_Ratio"] = df["Albumin_and_Globulin_Ratio"].fillna(
        df["Albumin_and_Globulin_Ratio"].median()
    )

    # Target: Dataset column -> 1 = liver disease, 2 = no disease
    # Convert to standard 1 = disease, 0 = healthy
    df["target"] = df["Dataset"].map({1: 1, 2: 0})

    feature_names = [c for c in df.columns if c not in ("Dataset", "target")]
    X = df[feature_names].copy()
    y = df["target"].copy()

    return X, y, feature_names


# --------------------------------------------------------------------------
# 2. KIDNEY DISEASE (Chronic Kidney Disease Dataset)
# --------------------------------------------------------------------------
def load_and_prepare_kidney(path: str):
    df = pd.read_csv(path)

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Clean up any stray whitespace / tabs that are common in this dataset
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan})

    # Columns that look numeric but were read as text (e.g. "44", "7800")
    for c in ["pcv", "wc", "rc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Binary categorical columns -> 0 / 1
    binary_maps = {
        "rbc": {"normal": 1, "abnormal": 0},
        "pc": {"normal": 1, "abnormal": 0},
        "pcc": {"present": 1, "notpresent": 0},
        "ba": {"present": 1, "notpresent": 0},
        "htn": {"yes": 1, "no": 0},
        "dm": {"yes": 1, "no": 0},
        "cad": {"yes": 1, "no": 0},
        "appet": {"good": 1, "poor": 0},
        "pe": {"yes": 1, "no": 0},
        "ane": {"yes": 1, "no": 0},
    }
    for col, mapping in binary_maps.items():
        df[col] = df[col].map(mapping)

    # Target: classification -> ckd = 1, notckd = 0
    df["target"] = df["classification"].map({"ckd": 1, "notckd": 0})
    df = df.drop(columns=["classification"])

    feature_names = [c for c in df.columns if c != "target"]

    # Impute remaining missing values with the median of each column
    for c in feature_names:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # Drop any rows where the target itself is missing
    df = df.dropna(subset=["target"])

    X = df[feature_names].copy()
    y = df["target"].astype(int).copy()

    return X, y, feature_names


# --------------------------------------------------------------------------
# 3. PARKINSON'S DISEASE
# --------------------------------------------------------------------------
def load_and_prepare_parkinsons(path: str):
    df = pd.read_csv(path)

    if "name" in df.columns:
        df = df.drop(columns=["name"])

    feature_names = [c for c in df.columns if c != "status"]
    X = df[feature_names].copy()
    y = df["status"].astype(int).copy()

    return X, y, feature_names

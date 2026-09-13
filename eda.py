"""
eda.py
------
Quick exploratory data analysis for the three disease datasets.
Generates and saves summary plots (class balance + correlation heatmap)
to the `outputs/` folder. Useful for the project report / presentation.

Run from the project root:
    python src/eda.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import (
    load_and_prepare_liver,
    load_and_prepare_kidney,
    load_and_prepare_parkinsons,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def analyze(name, X, y):
    # Class balance plot
    plt.figure(figsize=(4, 3))
    y.value_counts().plot(kind="bar", color=["#4C72B0", "#DD8452"])
    plt.title(f"{name.title()} — Class Balance")
    plt.xlabel("Class (1 = disease, 0 = healthy)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}_class_balance.png"))
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = X.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title(f"{name.title()} — Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}_correlation_heatmap.png"))
    plt.close()

    print(f"Saved EDA plots for {name} -> {OUT_DIR}")


def main():
    X, y, _ = load_and_prepare_liver(os.path.join(DATA_DIR, "indian_liver_patient.csv"))
    analyze("liver", X, y)

    X, y, _ = load_and_prepare_kidney(os.path.join(DATA_DIR, "kidney_disease.csv"))
    analyze("kidney", X, y)

    X, y, _ = load_and_prepare_parkinsons(os.path.join(DATA_DIR, "parkinsons.csv"))
    analyze("parkinsons", X, y)


if __name__ == "__main__":
    main()

"""
app.py
------
Streamlit frontend for the Multiple Disease Prediction System.

Supports three diseases:
    - Liver disease
    - Kidney disease
    - Parkinson's disease

Run with:
    streamlit run app.py

Make sure you have already run `python src/train_models.py` once so that
the trained models exist inside the `models/` folder.
"""

import os
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(
    page_title="Multiple Disease Prediction System",
    page_icon="🩺",
    layout="wide",
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts(disease: str):
    """Load model, scaler and feature list for a disease. Cached so it
    only runs once per session."""
    model_path = os.path.join(MODEL_DIR, f"{disease}_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"{disease}_scaler.pkl")
    features_path = os.path.join(MODEL_DIR, f"{disease}_features.json")
    metrics_path = os.path.join(MODEL_DIR, f"{disease}_metrics.json")

    if not os.path.exists(model_path):
        return None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(features_path) as f:
        features = json.load(f)
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    return {"model": model, "scaler": scaler, "features": features, "metrics": metrics}


def risk_level(probability: float) -> str:
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Moderate"
    else:
        return "High"


def show_prediction(probability: float, prediction: int, positive_label: str, negative_label: str):
    level = risk_level(probability)
    color = {"Low": "green", "Moderate": "orange", "High": "red"}[level]

    col1, col2 = st.columns([1, 1])
    with col1:
        if prediction == 1:
            st.error(f"⚠️ Prediction: **{positive_label}**")
        else:
            st.success(f"✅ Prediction: **{negative_label}**")
        st.markdown(
            f"**Probability of disease:** {probability * 100:.1f}%  \n"
            f"**Risk level:** :{color}[{level}]"
        )

    with col2:
        fig, ax = plt.subplots(figsize=(4, 0.6))
        ax.barh([0], [probability], color=color)
        ax.barh([0], [1], color="lightgray", zorder=0)
        ax.barh([0], [probability], color=color, zorder=1)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Predicted probability of disease")
        st.pyplot(fig)


def show_model_info(metrics: dict):
    if not metrics:
        return
    best = metrics["best_model"]
    m = metrics["all_results"][best]
    with st.expander("Model performance details"):
        st.write(f"**Best model:** {best}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{m['accuracy']:.2%}")
        c2.metric("Precision", f"{m['precision']:.2%}")
        c3.metric("Recall", f"{m['recall']:.2%}")
        c4.metric("F1-score", f"{m['f1_score']:.2%}")
        c5.metric("ROC-AUC", f"{m['roc_auc']:.2f}")
        st.write("Confusion matrix (test set):")
        st.write(np.array(m["confusion_matrix"]))


# --------------------------------------------------------------------------
# Page: Liver Disease
# --------------------------------------------------------------------------
def liver_page():
    st.header("🫀 Liver Disease Prediction")
    st.write("Enter the patient's lab test results to estimate the risk of liver disease.")

    artifacts = load_artifacts("liver")
    if artifacts is None:
        st.warning("Model not found. Please run `python src/train_models.py` first.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, value=1.0, step=0.1)
        direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, value=0.3, step=0.1)
    with col2:
        alk_phos = st.number_input("Alkaline Phosphotase", min_value=0, value=200)
        alt = st.number_input("Alamine Aminotransferase (ALT)", min_value=0, value=30)
        ast = st.number_input("Aspartate Aminotransferase (AST)", min_value=0, value=35)
    with col3:
        total_protein = st.number_input("Total Proteins", min_value=0.0, value=6.5, step=0.1)
        albumin = st.number_input("Albumin", min_value=0.0, value=3.2, step=0.1)
        ag_ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.0, value=1.0, step=0.1)

    if st.button("Predict Liver Disease", type="primary"):
        input_dict = {
            "Age": age,
            "Gender": 1 if gender == "Male" else 0,
            "Total_Bilirubin": total_bilirubin,
            "Direct_Bilirubin": direct_bilirubin,
            "Alkaline_Phosphotase": alk_phos,
            "Alamine_Aminotransferase": alt,
            "Aspartate_Aminotransferase": ast,
            "Total_Protiens": total_protein,
            "Albumin": albumin,
            "Albumin_and_Globulin_Ratio": ag_ratio,
        }
        X = pd.DataFrame([input_dict])[artifacts["features"]]
        X_scaled = artifacts["scaler"].transform(X)
        proba = artifacts["model"].predict_proba(X_scaled)[0][1]
        pred = int(proba >= 0.5)
        show_prediction(proba, pred, "Liver disease likely", "No liver disease detected")

    show_model_info(artifacts["metrics"])


# --------------------------------------------------------------------------
# Page: Kidney Disease
# --------------------------------------------------------------------------
def kidney_page():
    st.header("🧫 Kidney Disease Prediction")
    st.write("Enter the patient's clinical details to estimate the risk of chronic kidney disease.")

    artifacts = load_artifacts("kidney")
    if artifacts is None:
        st.warning("Model not found. Please run `python src/train_models.py` first.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=50, key="k_age")
        bp = st.number_input("Blood Pressure", min_value=0, value=80, key="k_bp")
        sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=2)
        al = st.slider("Albumin (0-5)", 0, 5, 0)
        su = st.slider("Sugar (0-5)", 0, 5, 0)
        rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
        pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
        pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
        ba = st.selectbox("Bacteria", ["notpresent", "present"])
    with col2:
        bgr = st.number_input("Blood Glucose Random", min_value=0, value=120)
        bu = st.number_input("Blood Urea", min_value=0, value=40)
        sc = st.number_input("Serum Creatinine", min_value=0.0, value=1.2, step=0.1)
        sod = st.number_input("Sodium", min_value=0, value=135)
        pot = st.number_input("Potassium", min_value=0.0, value=4.5, step=0.1)
        hemo = st.number_input("Hemoglobin", min_value=0.0, value=13.0, step=0.1)
        pcv = st.number_input("Packed Cell Volume", min_value=0, value=40)
    with col3:
        wc = st.number_input("White Blood Cell Count", min_value=0, value=8000)
        rc = st.number_input("Red Blood Cell Count", min_value=0.0, value=4.5, step=0.1)
        htn = st.selectbox("Hypertension", ["no", "yes"])
        dm = st.selectbox("Diabetes Mellitus", ["no", "yes"])
        cad = st.selectbox("Coronary Artery Disease", ["no", "yes"])
        appet = st.selectbox("Appetite", ["good", "poor"])
        pe = st.selectbox("Pedal Edema", ["no", "yes"])
        ane = st.selectbox("Anemia", ["no", "yes"])

    if st.button("Predict Kidney Disease", type="primary"):
        binary_map_yn = {"yes": 1, "no": 0}
        input_dict = {
            "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
            "rbc": 1 if rbc == "normal" else 0,
            "pc": 1 if pc == "normal" else 0,
            "pcc": 1 if pcc == "present" else 0,
            "ba": 1 if ba == "present" else 0,
            "bgr": bgr, "bu": bu, "sc": sc, "sod": sod, "pot": pot, "hemo": hemo,
            "pcv": pcv, "wc": wc, "rc": rc,
            "htn": binary_map_yn[htn], "dm": binary_map_yn[dm], "cad": binary_map_yn[cad],
            "appet": 1 if appet == "good" else 0,
            "pe": binary_map_yn[pe], "ane": binary_map_yn[ane],
        }
        X = pd.DataFrame([input_dict])[artifacts["features"]]
        X_scaled = artifacts["scaler"].transform(X)
        proba = artifacts["model"].predict_proba(X_scaled)[0][1]
        pred = int(proba >= 0.5)
        show_prediction(proba, pred, "Chronic kidney disease likely", "No kidney disease detected")

    show_model_info(artifacts["metrics"])


# --------------------------------------------------------------------------
# Page: Parkinson's Disease
# --------------------------------------------------------------------------
def parkinsons_page():
    st.header("🧠 Parkinson's Disease Prediction")
    st.write("Enter the patient's voice measurement features to estimate the risk of Parkinson's disease.")

    artifacts = load_artifacts("parkinsons")
    if artifacts is None:
        st.warning("Model not found. Please run `python src/train_models.py` first.")
        return

    st.info(
        "These values come from acoustic analysis of a sustained vowel recording "
        "(e.g. using tools like Praat). If you don't have exact values, the defaults "
        "represent a typical healthy voice sample."
    )

    defaults = {
        "MDVP:Fo(Hz)": 150.0, "MDVP:Fhi(Hz)": 180.0, "MDVP:Flo(Hz)": 110.0,
        "MDVP:Jitter(%)": 0.005, "MDVP:Jitter(Abs)": 0.00003, "MDVP:RAP": 0.003,
        "MDVP:PPQ": 0.003, "Jitter:DDP": 0.009, "MDVP:Shimmer": 0.03,
        "MDVP:Shimmer(dB)": 0.3, "Shimmer:APQ3": 0.015, "Shimmer:APQ5": 0.018,
        "MDVP:APQ": 0.024, "Shimmer:DDA": 0.045, "NHR": 0.02, "HNR": 22.0,
        "RPDE": 0.5, "DFA": 0.7, "spread1": -5.5, "spread2": 0.2, "D2": 2.3, "PPE": 0.2,
    }

    cols = st.columns(3)
    input_dict = {}
    for i, (feat, default_val) in enumerate(defaults.items()):
        with cols[i % 3]:
            input_dict[feat] = st.number_input(
                feat, value=float(default_val), format="%.5f", key=f"p_{feat}"
            )

    if st.button("Predict Parkinson's Disease", type="primary"):
        X = pd.DataFrame([input_dict])[artifacts["features"]]
        X_scaled = artifacts["scaler"].transform(X)
        proba = artifacts["model"].predict_proba(X_scaled)[0][1]
        pred = int(proba >= 0.5)
        show_prediction(proba, pred, "Parkinson's disease likely", "No Parkinson's disease detected")

    show_model_info(artifacts["metrics"])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    st.sidebar.title("🩺 Multiple Disease Prediction")
    st.sidebar.write(
        "Select a disease below to get a prediction based on patient data. "
        "This tool is for educational purposes and is **not** a substitute "
        "for professional medical advice."
    )
    choice = st.sidebar.radio(
        "Choose a disease to predict:",
        ["Liver Disease", "Kidney Disease", "Parkinson's Disease"],
    )

    st.title("Multiple Disease Prediction System")

    if choice == "Liver Disease":
        liver_page()
    elif choice == "Kidney Disease":
        kidney_page()
    else:
        parkinsons_page()

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Built with Scikit-learn, XGBoost and Streamlit as part of the "
        "Multiple Disease Prediction capstone project."
    )


if __name__ == "__main__":
    main()

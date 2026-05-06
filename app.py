import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

MODEL_PATH = ARTIFACTS / "heart_mlp.keras"
SCALER_PATH = ARTIFACTS / "heart_scaler.joblib"
META_PATH = ARTIFACTS / "heart_meta.json"
IMPORTANCE_PATH = ARTIFACTS / "input_importance.csv"


def load_assets():
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    imp_df = pd.read_csv(IMPORTANCE_PATH)
    return model, scaler, meta, imp_df


def prepare_model_input(raw_row, scaler, meta):
    df = pd.DataFrame([raw_row])
    encoded = pd.get_dummies(df, columns=meta["categorical_cols"], drop_first=False)

    for c in meta["model_columns"]:
        if c not in encoded.columns:
            encoded[c] = 0
    encoded = encoded[meta["model_columns"]].copy()

    cont_cols = meta["continuous_cols"]
    encoded[cont_cols] = scaler.transform(encoded[cont_cols])
    return encoded


def top3_local_drivers(model_input, imp_df):
    imp_map = dict(zip(imp_df["feature"], imp_df["importance"]))
    row = model_input.iloc[0]
    scores = []
    for f in model_input.columns:
        local_score = abs(float(row[f])) * float(imp_map.get(f, 0.0))
        scores.append((f, local_score))
    top3 = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
    return top3


def clean_name(feat):
    return feat.replace("_", " ").title()


def feature_bar(top3):
    names = [clean_name(x[0]) for x in top3][::-1]
    vals = [x[1] for x in top3][::-1]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh(names, vals, color=["#4e79a7", "#f28e2b", "#e15759"])
    ax.set_xlabel("Contribution Score")
    ax.set_title("Top 3 Features Driving This Prediction")
    plt.tight_layout()
    return fig


st.set_page_config(page_title="Heart Disease Risk Dashboard", layout="wide")
st.title("Part E: Local Heart Disease Prediction Dashboard")
st.caption("Interactive local dashboard for Assignment 4 (localhost run).")

if not (MODEL_PATH.exists() and SCALER_PATH.exists() and META_PATH.exists() and IMPORTANCE_PATH.exists()):
    st.error(
        "Missing model artifacts. Run the Part E artifact-export cell in the notebook first, then rerun this app."
    )
    st.stop()

model, scaler, meta, imp_df = load_assets()

st.subheader("E1: Input Form")

default_vals = meta["default_patient"]

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age (20-80)", min_value=20, max_value=80, value=int(default_vals["age"]), step=1)
    sex = st.selectbox("Sex (0=Female, 1=Male)", options=[0, 1], index=int(default_vals["sex"]))
    cp = st.selectbox("Chest Pain Type cp (1-4)", options=[1, 2, 3, 4], index=int(default_vals["cp"]) - 1)
    trestbps = st.number_input("Resting BP trestbps (80-220)", min_value=80, max_value=220, value=int(default_vals["trestbps"]), step=1)
    chol = st.number_input("Serum Cholesterol chol (100-600)", min_value=100, max_value=600, value=int(default_vals["chol"]), step=1)
with col2:
    fbs = st.selectbox("Fasting Blood Sugar fbs (0/1)", options=[0, 1], index=int(default_vals["fbs"]))
    restecg = st.selectbox("Resting ECG restecg (0-2)", options=[0, 1, 2], index=int(default_vals["restecg"]))
    thalach = st.number_input("Max Heart Rate thalach (70-210)", min_value=70, max_value=210, value=int(default_vals["thalach"]), step=1)
    exang = st.selectbox("Exercise Angina exang (0/1)", options=[0, 1], index=int(default_vals["exang"]))
with col3:
    oldpeak = st.number_input("ST Depression oldpeak (0.0-6.5)", min_value=0.0, max_value=6.5, value=float(default_vals["oldpeak"]), step=0.1, format="%.1f")
    slope = st.selectbox("Slope (1-3)", options=[1, 2, 3], index=int(default_vals["slope"]) - 1)
    ca = st.number_input("Major Vessels ca (0-3)", min_value=0, max_value=3, value=int(default_vals["ca"]), step=1)
    thal = st.selectbox("Thal (3, 6, 7)", options=[3, 6, 7], index=[3, 6, 7].index(int(default_vals["thal"])))

predict = st.button("Predict", type="primary")

if predict:
    raw_input = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    x_model = prepare_model_input(raw_input, scaler, meta)
    proba = float(model.predict(x_model, verbose=0)[0][0])
    pred = int(proba >= 0.5)
    conf = proba if pred == 1 else 1.0 - proba

    st.subheader("E2: Results Panel")
    label = "Disease Present" if pred == 1 else "No Disease"
    if pred == 1:
        st.error(f"Predicted Class: {label}")
    else:
        st.success(f"Predicted Class: {label}")

    st.write(f"Confidence: **{conf * 100:.2f}%**")

    top3 = top3_local_drivers(x_model, imp_df)
    st.pyplot(feature_bar(top3))

    names = [clean_name(x[0]) for x in top3]
    if pred == 1:
        st.write(
            f"This patient is predicted as higher cardiac risk with {conf*100:.1f}% confidence. "
            f"The strongest contributing signals are {names[0]}, {names[1]}, and {names[2]}. "
            "Please correlate with ECG findings, symptoms, and physician assessment for triage."
        )
    else:
        st.write(
            f"This patient is predicted as lower cardiac risk with {conf*100:.1f}% confidence. "
            f"The most influential factors were {names[0]}, {names[1]}, and {names[2]}. "
            "Continue routine monitoring and clinical follow-up based on overall presentation."
        )

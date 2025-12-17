import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Framingham Heart Disease Risk Assessment")

st.warning(
    "⚠️ Educational purposes only. Not a medical diagnosis."
)

# --------------------------------------------------
# Load model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.joblib")

model = load_model()

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 80, 50)
    male = st.radio("Gender", ["Male", "Female"])
    education = st.selectbox("Education Level", [1, 2, 3, 4])

    currentSmoker = st.checkbox("Current Smoker")
    cigsPerDay = st.slider("Cigarettes per Day", 0, 40, 0)

    BPMeds = st.checkbox("On BP Medication")
    prevalentStroke = st.checkbox("Previous Stroke")
    prevalentHyp = st.checkbox("Hypertension")
    diabetes = st.checkbox("Diabetes")

with col2:
    totChol = st.slider("Total Cholesterol", 100, 400, 200)
    sysBP = st.slider("Systolic BP", 90, 200, 120)
    diaBP = st.slider("Diastolic BP", 60, 120, 80)
    BMI = st.slider("BMI", 15.0, 45.0, 25.0)
    heartRate = st.slider("Heart Rate", 40, 120, 72)
    glucose = st.slider("Glucose", 60, 200, 90)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Calculate Risk", type="primary"):

    input_df = pd.DataFrame([{
        'male': 1 if male == "Male" else 0,
        'age': age,
        'education': education,
        'currentSmoker': int(currentSmoker),
        'cigsPerDay': cigsPerDay,
        'BPMeds': int(BPMeds),
        'prevalentStroke': int(prevalentStroke),
        'prevalentHyp': int(prevalentHyp),
        'diabetes': int(diabetes),
        'totChol': totChol,
        'sysBP': sysBP,
        'diaBP': diaBP,
        'BMI': BMI,
        'heartRate': heartRate,
        'glucose': glucose
    }])

    # Predict probability
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Result")

    st.metric("10-year CHD Risk", f"{prob:.2%}")

    if prob >= 0.5:
        st.error("⚠️ HIGH RISK")
    else:
        st.success("✅ LOW RISK")

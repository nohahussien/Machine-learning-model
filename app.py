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
st.warning("⚠️ This is NOT a medical diagnosis. Consult a healthcare professional.")

# --------------------------------------------------
# Load model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.joblib")

model = load_model()

# --------------------------------------------------
# User Inputs - ALL MANUAL INPUTS
# --------------------------------------------------
st.header("Patient Information")
st.write("Please enter all patient information below:")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics & Lifestyle")
    
    age = st.number_input(
        "Age (years)",
        min_value=20,
        max_value=100,
        value=None,
        placeholder="Enter age...",
        help="Patient's age in years"
    )
    
    male = st.radio(
        "Gender",
        ["Male", "Female"],
        index=None,
        help="Patient's biological sex"
    )
    
    education = st.selectbox(
        "Education Level",
        [None, 1, 2, 3, 4],
        format_func=lambda x: {
            None: "Select education level...",
            1: "1: Some High School",
            2: "2: High School/GED",
            3: "3: Some College/Vocational School",
            4: "4: College Graduate"
        }.get(x, str(x)),
        help="Highest education level achieved"
    )
    
    st.markdown("---")
    st.subheader("Smoking Status")
    
    currentSmoker = st.radio(
        "Current Smoker?",
        ["Yes", "No"],
        index=None,
        help="Does the patient currently smoke cigarettes?"
    )
    
    cigsPerDay = st.number_input(
        "Cigarettes per Day",
        min_value=0,
        max_value=100,
        value=None,
        placeholder="Enter number...",
        help="If not a smoker, enter 0"
    )
    
    st.markdown("---")
    st.subheader("Medical History")
    
    BPMeds = st.radio(
        "On Blood Pressure Medication?",
        ["Yes", "No"],
        index=None,
        help="Is the patient currently taking blood pressure medication?"
    )
    
    prevalentStroke = st.radio(
        "Previous Stroke?",
        ["Yes", "No"],
        index=None,
        help="Has the patient had a stroke before?"
    )
    
    prevalentHyp = st.radio(
        "Hypertension Diagnosis?",
        ["Yes", "No"],
        index=None,
        help="Has the patient been diagnosed with hypertension?"
    )
    
    diabetes = st.radio(
        "Diabetes Diagnosis?",
        ["Yes", "No"],
        index=None,
        help="Has the patient been diagnosed with diabetes?"
    )

with col2:
    st.subheader("Clinical Measurements")
    
    totChol = st.number_input(
        "Total Cholesterol (mg/dL)",
        min_value=100,
        max_value=500,
        value=None,
        placeholder="Enter value...",
        help="Normal range: 125-200 mg/dL"
    )
    
    sysBP = st.number_input(
        "Systolic Blood Pressure (mmHg)",
        min_value=80,
        max_value=250,
        value=None,
        placeholder="Enter value...",
        help="Normal: <120 mmHg"
    )
    
    diaBP = st.number_input(
        "Diastolic Blood Pressure (mmHg)",
        min_value=50,
        max_value=150,
        value=None,
        placeholder="Enter value...",
        help="Normal: <80 mmHg"
    )
    
    BMI = st.number_input(
        "Body Mass Index (kg/m²)",
        min_value=15.0,
        max_value=60.0,
        value=None,
        placeholder="Enter value...",
        step=0.1,
        help="Underweight: <18.5, Normal: 18.5-24.9, Overweight: 25-29.9, Obese: ≥30"
    )
    
    heartRate = st.number_input(
        "Resting Heart Rate (bpm)",
        min_value=40,
        max_value=150,
        value=None,
        placeholder="Enter value...",
        help="Normal resting: 60-100 bpm"
    )
    
    glucose = st.number_input(
        "Glucose Level (mg/dL)",
        min_value=60,
        max_value=300,
        value=None,
        placeholder="Enter value...",
        help="Normal fasting: 70-100 mg/dL"
    )

# --------------------------------------------------
# Validation and Prediction
# --------------------------------------------------

# Check if all fields are filled
def validate_inputs():
    required_fields = [
        age is not None,
        male is not None,
        education is not None,
        currentSmoker is not None,
        cigsPerDay is not None,
        BPMeds is not None,
        prevalentStroke is not None,
        prevalentHyp is not None,
        diabetes is not None,
        totChol is not None,
        sysBP is not None,
        diaBP is not None,
        BMI is not None,
        heartRate is not None,
        glucose is not None
    ]
    
    if all(required_fields):
        return True
    else:
        missing = []
        if age is None: missing.append("Age")
        if male is None: missing.append("Gender")
        if education is None: missing.append("Education Level")
        if currentSmoker is None: missing.append("Current Smoker")
        if cigsPerDay is None: missing.append("Cigarettes per Day")
        if BPMeds is None: missing.append("Blood Pressure Medication")
        if prevalentStroke is None: missing.append("Previous Stroke")
        if prevalentHyp is None: missing.append("Hypertension")
        if diabetes is None: missing.append("Diabetes")
        if totChol is None: missing.append("Total Cholesterol")
        if sysBP is None: missing.append("Systolic Blood Pressure")
        if diaBP is None: missing.append("Diastolic Blood Pressure")
        if BMI is None: missing.append("BMI")
        if heartRate is None: missing.append("Heart Rate")
        if glucose is None: missing.append("Glucose")
        
        return False, missing

# Prediction button
if st.button("Calculate Risk", type="primary", use_container_width=True):
    
    # Validate inputs
    validation_result = validate_inputs()
    
    if validation_result is True:
        # All fields filled, proceed with prediction
        
        # Convert inputs to model format
        input_df = pd.DataFrame([{
            'male': 1 if male == "Male" else 0,
            'age': age,
            'education': education,
            'currentSmoker': 1 if currentSmoker == "Yes" else 0,
            'cigsPerDay': cigsPerDay,
            'BPMeds': 1 if BPMeds == "Yes" else 0,
            'prevalentStroke': 1 if prevalentStroke == "Yes" else 0,
            'prevalentHyp': 1 if prevalentHyp == "Yes" else 0,
            'diabetes': 1 if diabetes == "Yes" else 0,
            'totChol': totChol,
            'sysBP': sysBP,
            'diaBP': diaBP,
            'BMI': BMI,
            'heartRate': heartRate,
            'glucose': glucose
        }])
        
        # Show entered values for confirmation
        with st.expander("📋 Review Entered Values", expanded=True):
            review_col1, review_col2 = st.columns(2)
            
            with review_col1:
                st.write("**Demographics:**")
                st.write(f"- Age: {age} years")
                st.write(f"- Gender: {male}")
                st.write(f"- Education Level: {education}")
                st.write("")
                st.write("**Lifestyle:**")
                st.write(f"- Current Smoker: {currentSmoker}")
                if currentSmoker == "Yes":
                    st.write(f"- Cigarettes per Day: {cigsPerDay}")
            
            with review_col2:
                st.write("**Medical History:**")
                st.write(f"- BP Medication: {BPMeds}")
                st.write(f"- Previous Stroke: {prevalentStroke}")
                st.write(f"- Hypertension: {prevalentHyp}")
                st.write(f"- Diabetes: {diabetes}")
                st.write("")
                st.write("**Measurements:**")
                st.write(f"- Cholesterol: {totChol} mg/dL")
                st.write(f"- BP: {sysBP}/{diaBP} mmHg")
                st.write(f"- BMI: {BMI} kg/m²")
                st.write(f"- Heart Rate: {heartRate} bpm")
                st.write(f"- Glucose: {glucose} mg/dL")
        
        try:
            # Predict probability
            prob = model.predict_proba(input_df)[0][1]
            
            st.subheader("🎯 Risk Assessment Result")
            
            # Display risk with color coding
            risk_col1, risk_col2, risk_col3 = st.columns([1, 2, 1])
            
            with risk_col2:
                st.metric(
                    "10-year Coronary Heart Disease Risk",
                    f"{prob:.1%}",
                    delta=None
                )
            
            # Risk interpretation
            st.markdown("---")
            
            if prob >= 0.5:
                st.error("""
                ### ⚠️ HIGH RISK
                
                **Clinical Implications:**
                - ≥50% probability of developing coronary heart disease in the next 10 years
                - Immediate medical consultation recommended
                - Comprehensive risk factor management needed
                
                **Next Steps:**
                - Consult a cardiologist
                - Lifestyle modification (diet, exercise, smoking cessation)
                - Consider medication as advised by physician
                """)
            elif prob >= 0.2:
                st.warning("""
                ### ⚠️ MODERATE RISK
                
                **Clinical Implications:**
                - 20-49% probability of developing coronary heart disease in the next 10 years
                - Preventive measures strongly recommended
                
                **Next Steps:**
                - Regular monitoring by primary care physician
                - Focus on modifiable risk factors
                - Consider preventive medications if indicated
                """)
            else:
                st.success("""
                ### ✅ LOW RISK
                
                **Clinical Implications:**
                - <20% probability of developing coronary heart disease in the next 10 years
                - Continue healthy lifestyle practices
                
                **Recommendations:**
                - Maintain healthy lifestyle
                - Regular health check-ups
                - Monitor risk factors annually
                """)
            
            # Disclaimer
            st.info("""
            **Disclaimer:** This prediction is based on the Framingham Heart Study risk algorithm. 
            Individual risk may vary based on factors not included in this model. 
            Always consult with a healthcare professional for personalized medical advice.
            """)
            
        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            st.info("""
            **Troubleshooting:**
            1. Ensure all inputs are within valid ranges
            2. The model might be expecting specific data format
            3. Check if model file is properly loaded
            """)
            
    else:
        # Show missing fields
        st.error("❌ Please fill in all required fields before calculating risk.")
        
        missing_fields = validation_result[1]
        
        st.warning(f"**Missing Information ({len(missing_fields)} fields):**")
        
        missing_col1, missing_col2 = st.columns(2)
        
        with missing_col1:
            for i, field in enumerate(missing_fields[:len(missing_fields)//2]):
                st.write(f"• {field}")
        
        with missing_col2:
            for i, field in enumerate(missing_fields[len(missing_fields)//2:]):
                st.write(f"• {field}")
        
        st.info("Please scroll up and complete all the fields marked with red asterisks.")

# --------------------------------------------------
# Information Section
# --------------------------------------------------
st.markdown("---")
st.subheader("📊 About This Tool")

with st.expander("Learn more about the Framingham Risk Score"):
    st.write("""
    The Framingham Risk Score is used to estimate the 10-year cardiovascular risk of an individual. 
    It was developed based on the Framingham Heart Study, a long-term epidemiological study of cardiovascular disease.
    
    **Variables included in this assessment:**
    - **Demographics:** Age, Gender, Education
    - **Lifestyle:** Smoking status
    - **Medical History:** Hypertension, Diabetes, Stroke, BP medication
    - **Clinical Measurements:** Cholesterol, Blood Pressure, BMI, Heart Rate, Glucose
    
    **Risk Categories:**
    - **Low Risk:** <20% (Green)
    - **Moderate Risk:** 20-49% (Yellow/Orange)
    - **High Risk:** ≥50% (Red)
    
    **Limitations:**
    - Does not include family history of heart disease
    - Does not account for physical activity or diet
    - May not be accurate for all ethnic groups
    - Should be used as a screening tool, not a diagnostic tool
    """)

# --------------------------------------------------
# CSS for better styling
# --------------------------------------------------
st.markdown("""
<style>
    .stNumberInput input, .stSelectbox, .stRadio {
        background-color: #f0f2f6;
    }
    .required-field::after {
        content: " *";
        color: red;
    }
</style>
""", unsafe_allow_html=True)
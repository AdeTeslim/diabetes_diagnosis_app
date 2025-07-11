import streamlit as st
import joblib
import pandas as pd
import numpy as np

# App Header
st.set_page_config(page_title="Diabetes Diagnosis", layout="centered")
st.title("🩺 Medical Diagnosis on Diabetes")
st.subheader("Developed by Yabatech Digital Academy Team")
st.markdown("---")
st.write("Enter your medical details below to check for signs of diabetes risk")

# Load Model + Scaler
svm = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define feature order from training
feature_order = ['No_Pation', 'Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']

# Input Form
with st.form("diabetes_form"):
    st.write("### 🧪 Input Medical Values")

    # Realistic Ranges and Abbreviations
    col1, col2 = st.columns(2)

    with col1:
        no_patient = st.slider("No_Pation (No. of Pregnancies)", 0, 20, 2)
        gender = st.slider("Gender (0 = Female, 1 = Male)", 0, 1, 0)
        age = st.slider("AGE (in years)", 10, 100, 35)
        urea = st.slider("Urea (mg/dL)", 0.0, 50.0, 4.0)
        cr = st.slider("Creatinine (mg/dL)", 0.0, 300.0, 60.0)
        hba1c = st.slider("HbA1c (%)", 0.0, 20.0, 6.0)
    
    with col2:
        chol = st.slider("Cholesterol (mg/dL)", 0.0, 300.0, 4.0)
        tg = st.slider("Triglycerides (mg/dL)", 0.0, 400.0, 1.5)
        hdl = st.slider("HDL (mg/dL)", 0.0, 100.0, 1.0)
        ldl = st.slider("LDL (mg/dL)", 0.0, 200.0, 2.0)
        vldl = st.slider("VLDL (mg/dL)", 0.0, 100.0, 0.5)
        bmi = st.slider("BMI (kg/m²)", 10.0, 50.0, 25.0)

    submit_btn = st.form_submit_button("🔍 Diagnose")

# Process + Predict
if submit_btn:
    try:
        # Create input DataFrame with correct feature order
        input_data = pd.DataFrame(
            [[no_patient, gender, age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi]],
            columns=feature_order
        )
        
        # Debug: Display input data
        st.write("**Input Data:**")
        st.write(input_data)

        # Scale input data
        input_scaled = scaler.transform(input_data)
        
        # Debug: Display scaled data
        st.write("**Scaled Input Data:**")
        st.write(input_scaled)

        # Predict
        prediction = svm.predict(input_scaled)[0]

        # Map prediction to result
        result_map = {
            0: "🟢 No Diabetes Detected.",
            1: "🟡 Possible Risk of Diabetes. Monitor regularly.",
            2: "🔴 Diabetes Confirmed. Please consult a medical professional."
        }

        st.success(f"**Prediction Result:** {result_map.get(prediction, 'Unknown prediction.')}")
        
        # Debug: Display prediction probability
        prob = svm.predict_proba(input_scaled)[0]
        st.write("**Prediction Probabilities:**")
        st.write(f"No Diabetes: {prob[0]:.2f}, Possible Risk: {prob[1]:.2f}, Diabetes: {prob[2]:.2f}")

    except Exception as e:
        st.error("⚠️ An error occurred while making the prediction. Please check your inputs and try again.")
        st.exception(e)

# Footnote
st.markdown("---")
st.caption("This application is for educational purposes only and not a replacement for professional medical diagnosis.")


# TO RUN
# streamlit run streamlitapp.py


# ABBREVIATON MEANINGS
# No_Pation: Number of Pregnancies
# Gender: 0 = Female, 1 = Male
# AGE: Age in years
# Urea: Urea level in blood (mg/dL)
# Cr: Creatinine (Kidney function)
# HbA1c: Glycated hemoglobin(%)
# Chol: Cholestrol (mg/dL)
# TG: Triglycerides (mg/dL)
# HDL: High-Density Lipoprotein (mg/dL)
# LDL: Low-Density Lipoprotein (mg/dL) 
# VLDL: Very-Low-Density Lipoprotein(mg/dL)
# BMI: Body Mass Index (kg/m²)




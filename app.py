import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the saved model and scaler
model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Heart Disease Prediction App", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction System")
st.write("Enter your medical details below to check your heart health status using a Machine Learning Random Forest model.")

# Create input form for features (adjust these fields based on the exact features your dataset used)
with st.form("prediction_form"):
    st.subheader("Patient Medical Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
        chol = st.number_input("Serum Cholesterol in mg/dl (chol)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[( "True", 1), ("False", 0)], format_func=lambda x: x[0])[1]
        
    with col2:
        restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", options=[0, 1, 2])
        thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina (exang)", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
        oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", options=[0, 1, 2])
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", options=[0, 1, 2, 3, 4])
        thal = st.selectbox("Thal (thal)", options=[1, 2, 3])

    submit_button = st.form_submit_button(label="Predict Health Status")

# When the user clicks the submit button
if submit_button:
    # Gather inputs into an array
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # Scale input using your saved scaler
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)
    
    st.subheader("Results:")
    if prediction[0] == 0:
        st.success(f"🟢 The person does not have heart disease. (Confidence: {prediction_proba[0][0]*100:.2f}%)")
    else:
        st.error(f"🔴 The person has heart disease. (Confidence: {prediction_proba[0][1]*100:.2f}%)")
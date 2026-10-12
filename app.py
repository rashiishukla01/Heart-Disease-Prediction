import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Cardiovascular Risk Assessment",
    page_icon="🫀",
    layout="wide"
)

# Reference population medians and thresholds (UCI Heart Disease benchmark)
POPULATION_BENCHMARKS = {
    "trestbps": {"median": 130.0, "high": 140.0, "unit": "mm Hg", "name": "Resting Blood Pressure"},
    "chol": {"median": 240.0, "high": 240.0, "unit": "mg/dL", "name": "Total Cholesterol"},
    "thalach": {"median": 153.0, "low": 120.0, "unit": "bpm", "name": "Peak Heart Rate"},
    "oldpeak": {"median": 0.8, "high": 2.0, "unit": "mm", "name": "ST Depression"}
}


@st.cache_resource
def load_artifacts():
    model = joblib.load("heart_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Please ensure 'heart_model.pkl' and 'scaler.pkl' exist.")
    st.stop()

st.title("Coronary Heart Disease Risk Classifier")
st.caption("Inference engine trained on the Cleveland Clinic Foundation dataset.")

# Clinical feature encodings
CP_OPTIONS = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-Anginal Discomfort": 2,
    "Asymptomatic": 3
}

ECG_OPTIONS = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

SLOPE_OPTIONS = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

THAL_OPTIONS = {
    "Normal Blood Flow": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

with st.form("clinical_intake_form"):
    st.subheader("1. Patient Profile & Resting Vitals")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=95, value=52)
        sex = st.selectbox("Sex", options=["Male", "Female"])
        sex_encoded = 1 if sex == "Male" else 0
    with col2:
        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80, max_value=220, value=128,
            help="Systemic arterial pressure on admission"
        )
        chol = st.number_input(
            "Serum Cholesterol (mg/dL)",
            min_value=100, max_value=580, value=204,
            help="Total fasting serum cholesterol"
        )
    with col3:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=["False", "True"])
        fbs_encoded = 1 if fbs == "True" else 0
        restecg = st.selectbox("Resting Electrocardiogram", options=list(ECG_OPTIONS.keys()))
        restecg_encoded = ECG_OPTIONS[restecg]

    st.subheader("2. Exercise Tolerance & Stress Testing")
    col4, col5 = st.columns(2)
    with col4:
        cp = st.selectbox("Chest Pain Presentation", options=list(CP_OPTIONS.keys()))
        cp_encoded = CP_OPTIONS[cp]
        thalach = st.number_input(
            "Maximum Achieved Heart Rate (bpm)",
            min_value=60, max_value=220, value=156,
            help="Max heart rate reached during treadmill exertion"
        )
        exang = st.selectbox("Exercise-Induced Angina", options=["No", "Yes"])
        exang_encoded = 1 if exang == "Yes" else 0
    with col5:
        oldpeak = st.number_input(
            "ST Depression (oldpeak)",
            min_value=0.0, max_value=7.0, value=1.0, step=0.1,
            help="ST depression induced by exercise relative to rest"
        )
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=list(SLOPE_OPTIONS.keys()))
        slope_encoded = SLOPE_OPTIONS[slope]

    st.subheader("3. Diagnostic Imaging Markers")
    col6, col7 = st.columns(2)
    with col6:
        ca = st.selectbox(
            "Major Fluoroscopy Vessels Colored (0–3)",
            options=[0, 1, 2, 3],
            help="Number of major coronary vessels visible under fluoroscopy"
        )
    with col7:
        thal = st.selectbox("Thallium Scintigraphy", options=list(THAL_OPTIONS.keys()))
        thal_encoded = THAL_OPTIONS[thal]

    submitted = st.form_submit_button("Compute Prediction", use_container_width=True)

if submitted:
    # Maintain strict tabular structure for model reproducibility
    raw_inputs = pd.DataFrame([{
        "age": age,
        "sex": sex_encoded,
        "cp": cp_encoded,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs_encoded,
        "restecg": restecg_encoded,
        "thalach": thalach,
        "exang": exang_encoded,
        "oldpeak": oldpeak,
        "slope": slope_encoded,
        "ca": ca,
        "thal": thal_encoded
    }])

    scaled_inputs = scaler.transform(raw_inputs)
    prediction = model.predict(scaled_inputs)[0]
    probabilities = model.predict_proba(scaled_inputs)[0]
    disease_prob = probabilities[1]

    st.divider()

    # Results summary
    res_col1, res_col2, res_col3 = st.columns([1, 1, 2])
    with res_col1:
        st.metric(
            label="Diagnostic Output",
            value="Positive (Disease Risk)" if prediction == 1 else "Negative (No Disease)",
            delta="- High Risk" if prediction == 1 else "+ Low Risk",
            delta_color="inverse"
        )
    with res_col2:
        st.metric(
            label="Model Probability",
            value=f"{disease_prob * 100:.1f}%",
            help="Calibrated posterior probability of the positive class"
        )
    with res_col3:
        st.write("**Risk Spectrum**")
        st.progress(float(disease_prob))
        st.caption("Threshold boundary: 0.50 (binary cut-off)")

    # Data-grounded patient metrics vs population baseline
    st.subheader("Clinical Value Deviations")
    st.caption("Comparison of patient markers against standard cohort median thresholds:")

    deviation_data = []
    for feature_key, spec in POPULATION_BENCHMARKS.items():
        val = raw_inputs.iloc[0][feature_key]
        delta_from_median = val - spec["median"]
        deviation_data.append({
            "Metric": spec["name"],
            "Patient Value": f"{val} {spec['unit']}",
            "Cohort Median": f"{spec['median']} {spec['unit']}",
            "Delta": f"{delta_from_median:+.1f} {spec['unit']}"
        })

    st.dataframe(pd.DataFrame(deviation_data), use_container_width=True, hide_index=True)

    st.info(
        "**Clinical Disclaimer:** This tool implements a statistical classifier for screening exploration. "
        "It is not validated for standalone diagnostic or therapeutic decisions."
    )

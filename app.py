import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Heart Risk Assessment Tool",
    page_icon="❤️",
    layout="wide"
)

# Optional theme styling
theme = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)

if theme == "Dark":
    st.markdown("""
        <style>
        .stApp { background-color: #0d1117; color: #c9d1d9; }
        div[data-testid="stMetricValue"] { color: #f0f6fc; }
        .tip-card {
            background-color: #161b22;
            border-left: 4px solid #58a6ff;
            padding: 12px 16px;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp { background-color: #ffffff; color: #24292f; }
        .tip-card {
            background-color: #f6f8fa;
            border-left: 4px solid #0969da;
            padding: 12px 16px;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("heart_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
except Exception:
    st.error("Missing model files. Make sure 'heart_model.pkl' and 'scaler.pkl' are in the project root.")
    st.stop()

st.title("Cardiovascular Risk Predictor")
st.write("A machine learning tool to screen for potential heart disease risk based on clinical test metrics.")

# Form options mapped to model integers
CP_MAP = {
    "Typical angina (chest tightness / pressure)": 0,
    "Atypical angina (unusual chest discomfort)": 1,
    "Non-anginal pain (e.g., sharp, muscle-related, acid reflux)": 2,
    "Asymptomatic (no chest pain)": 3
}

ECG_MAP = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Probable or definite left ventricular hypertrophy": 2
}

SLOPE_MAP = {
    "Upsloping (normal response to exercise)": 0,
    "Flat (moderate stress)": 1,
    "Downsloping (unhealthy cardiac strain)": 2
}

THAL_MAP = {
    "Normal blood flow": 1,
    "Fixed defect (permanent tissue damage / old scar)": 2,
    "Reversible defect (blood flow blocked during stress)": 3
}

with st.form("patient_data_form"):
    st.markdown("### Patient Profile & Vitals")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=48)
        sex = st.selectbox("Biological Sex", ["Male", "Female"])
        sex_val = 1 if sex == "Male" else 0
    with c2:
        bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=125,
                             help="Measured sitting down. Optimal is generally under 120/80.")
        chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=550, value=210,
                               help="Total cholesterol from a lipid test. Target is typically < 200.")
    with c3:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", ["No (Normal)", "Yes (Elevated)"])
        fbs_val = 1 if "Yes" in fbs else 0
        rest_ecg = st.selectbox("Resting ECG Result", list(ECG_MAP.keys()))
        rest_ecg_val = ECG_MAP[rest_ecg]

    st.markdown("### Exercise Stress Test Results")
    c4, c5 = st.columns(2)
    with c4:
        cp = st.selectbox("Chest Pain Sensation", list(CP_MAP.keys()))
        cp_val = CP_MAP[cp]
        max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=152,
                                help="Maximum heart rate reached during physical exertion.")
        exang = st.selectbox("Exercise-Induced Chest Pain?", ["No", "Yes"])
        exang_val = 1 if exang == "Yes" else 0
    with c5:
        oldpeak = st.number_input("ST Depression from Exercise", min_value=0.0, max_value=8.0, value=1.0, step=0.1,
                                 help="ECG value tracking cardiac strain under physical stress.")
        slope = st.selectbox("Peak Exercise ST Slope", list(SLOPE_MAP.keys()))
        slope_val = SLOPE_MAP[slope]

    st.markdown("### Imaging & Lab Markers")
    c6, c7 = st.columns(2)
    with c6:
        ca = st.selectbox("Major Vessels Colored by Fluoroscopy (0-4)", [0, 1, 2, 3, 4],
                          help="Number of coronary arteries seen clearly under contrast dye.")
    with c7:
        thal = st.selectbox("Thallium Stress Test Result", list(THAL_MAP.keys()))
        thal_val = THAL_MAP[thal]

    st.write("")
    submitted = st.form_submit_button("Run Risk Prediction", use_container_width=True)


if submitted:
    # Feature order must match training data: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    features = np.array([[
        age, sex_val, cp_val, bp, chol, fbs_val,
        rest_ecg_val, max_hr, exang_val, oldpeak, slope_val, ca, thal_val
    ]])

    scaled_features = scaler.transform(features)
    pred = model.predict(scaled_features)[0]
    prob = model.predict_proba(scaled_features)[0]

    risk_pct = prob[1] * 100
    healthy_pct = prob[0] * 100

    st.divider()
    st.subheader("Results")

    col_res1, col_res2, col_res3 = st.columns([1, 1, 2])
    with col_res1:
        st.metric("Model Verdict", "Low Risk" if pred == 0 else "Elevated Risk")
    with col_res2:
        st.metric("Risk Score", f"{risk_pct:.1f}%")
    with col_res3:
        st.write("**Risk Meter**")
        st.progress(float(risk_pct / 100.0))
        st.caption(f"Estimated probability of cardiovascular risk: {risk_pct:.1f}%")

    if pred == 0:
        st.success(f"**Low Risk Detected** (Confidence: {healthy_pct:.1f}%)\n\nThe supplied health markers do not suggest significant indicators of coronary heart disease based on the training dataset.")
    else:
        st.error(f"**Elevated Risk Detected** (Risk score: {risk_pct:.1f}%)\n\nSeveral clinical values correspond to patterns associated with cardiovascular risk.")

    # Check for specific high-risk indicators to provide practical pointers
    st.markdown("### Observations & Follow-ups")
    feedback = []

    if bp >= 140:
        feedback.append(f"**High Resting Blood Pressure ({bp} mm Hg):** This falls into hypertensive ranges. Lowering sodium intake and speaking to a doctor about blood pressure management is advised.")
    elif bp >= 130:
        feedback.append(f"**Slightly Elevated Blood Pressure ({bp} mm Hg):** Keep an eye on regular readings and maintain cardio activity.")

    if chol >= 240:
        feedback.append(f"**High Total Cholesterol ({chol} mg/dL):** Elevated cholesterol increases arterial plaque risk. A full lipid panel (LDL/HDL/Triglycerides) is recommended.")
    elif chol >= 200:
        feedback.append(f"**Borderline Cholesterol ({chol} mg/dL):** Moderating dietary saturated fats can help bring this below 200.")

    if fbs_val == 1:
        feedback.append("**Elevated Fasting Glucose:** Blood sugar above 120 mg/dL is an independent risk multiplier for cardiovascular strain.")

    if exang_val == 1 or oldpeak > 1.5:
        feedback.append("**Exercise Strain Marker:** Chest discomfort or higher ST depression under physical exertion warrants a formal treadmill stress test with a cardiologist.")

    if not feedback:
        feedback.append("All primary vitals (blood pressure, fasting sugar, cholesterol) sit within standard ranges.")

    for item in feedback:
        st.markdown(f"<div class='tip-card'>{item}</div>", unsafe_allow_html=True)

    st.caption("Note: This app is a portfolio demonstration built with scikit-learn and is not meant to replace clinical medical advice.")

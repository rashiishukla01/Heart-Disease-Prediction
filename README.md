# Heart-Disease-Prediction
# ❤️ Heart Disease Risk Predictor

A lightweight machine learning web application that predicts the likelihood of heart disease using clinical diagnostic indicators.

This project started as an exploratory machine learning experiment in Google Colab and was turned into an interactive, deployable web tool so anyone can test custom inputs against the trained model in real time.

---

## 📌 Overview

Early detection of cardiovascular risk factors is critical. This project takes common health metrics—such as blood pressure, resting ECG, cholesterol levels, and exercise-induced markers—and feeds them through a trained classifier to provide an immediate risk assessment along with prediction confidence.

### Performance Highlights

* **Model:** Random Forest Classifier (`n_estimators=100`).


* **Test Accuracy:** ~96.8% evaluated on unseen test data.


* **Training Accuracy:** 100%.


* **Dataset:** Cleaned and merged cardiovascular diagnostic records (`cleaned_merged_heart_dataset.csv`).



---

## 🛠️ Tech Stack

* **Language:** Python
* **Machine Learning:** Scikit-learn, NumPy, Pandas


* **Serialization:** Joblib


* **User Interface:** Streamlit


* **Environment:** Google Colab (training) → Local / Streamlit Cloud (deployment)



---

## 🗂️ Project Structure

```text
heart-disease-predictor/
├── app.py                 # Streamlit web interface and inference logic
├── heart_model.pkl        # Serialized Random Forest model
├── scaler.pkl             # Pre-fitted feature scaler for preprocessing
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

```

---

## 🩺 Clinical Parameters Considered

The prediction form accepts 13 key diagnostic metrics:

| Parameter | Description |
| --- | --- |
| **Age** | Patient age in years

 |
| **Sex** | Biological sex (Male / Female)

 |
| **Chest Pain Type (cp)** | Angina classification (0 to 3)

 |
| **Resting BP (trestbps)** | Resting blood pressure in mm Hg

 |
| **Cholesterol (chol)** | Serum cholesterol in mg/dl

 |
| **Fasting Blood Sugar (fbs)** | Whether fasting blood sugar exceeds 120 mg/dl

 |
| **Resting ECG (restecg)** | Resting electrocardiographic results (0, 1, 2)

 |
| **Max Heart Rate (thalach)** | Maximum heart rate achieved during testing

 |
| **Exercise Angina (exang)** | Angina induced by physical exercise (Yes / No)

 |
| **ST Depression (oldpeak)** | ST depression induced by exercise relative to rest

 |
| **Slope** | Slope of the peak exercise ST segment

 |
| **Fluoroscopy (ca)** | Number of major vessels colored by fluoroscopy (0–4)

 |
| **Thalassemia (thal)** | Thalassemia status (Normal, Fixed Defect, Reversible Defect)

 |

---

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-predictor.git
cd heart-disease-predictor

```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt

```

### 3. Launch the Streamlit application

```bash
python -m streamlit run app.py

```

Open `http://localhost:8501` in your browser to interact with the interface.

---

## ⚠️ Disclaimer

*This application is built for educational and portfolio demonstration purposes only. It should not be used as a substitute for professional clinical diagnosis or medical consultation.*

---

Would you like me to generate a matching `.gitignore` file or help you set up an interactive GitHub badge showing the live demo link?

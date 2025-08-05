import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sqlite3
import os
from sklearn.preprocessing import StandardScaler

# Load the dataset and model
df = pd.read_csv("heart.csv")
model = joblib.load("heart_disease_model.pkl")

# Fit scaler with training data
X = df.drop("target", axis=1)
scaler = StandardScaler()
scaler.fit(X)

# Streamlit UI
st.title("🫀 Cardiovascular Disease Prediction")
st.subheader("Enter Patient Details")

# Patient Name
patient_name = st.text_input("Patient Name")

# Input fields
age = st.number_input("Age", 20, 100, step=1)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, step=1)
chol = st.number_input("Cholesterol (mg/dL)", 100, 600, step=1)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, step=1)
exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, step=0.1)
slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# Encode values
sex_encoded = 1 if sex == "Male" else 0
cp_encoded = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
fbs_encoded = 1 if fbs == "Yes" else 0
restecg_encoded = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
exang_encoded = 1 if exang == "Yes" else 0
slope_encoded = ["Upsloping", "Flat", "Downsloping"].index(slope)
thal_encoded = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)

# Create input array
input_data = np.array([[age, sex_encoded, cp_encoded, trestbps, chol, fbs_encoded,
                        restecg_encoded, thalach, exang_encoded, oldpeak,
                        slope_encoded, ca, thal_encoded]])

input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    if not patient_name:
        st.warning("⚠️ Please enter a patient name.")
    else:
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][prediction] * 100

        # Show result
        if prediction == 0:
            st.error(f"🔴 {patient_name} is likely to have heart disease ({prob:.2f}% confidence).")
        else:
            st.success(f"✅ {patient_name} is not likely to have heart disease ({prob:.2f}% confidence).")

        # Save to CSV
        result_dict = {
            "Patient Name": patient_name,
            "Age": age,
            "Sex": sex,
            "Chest Pain Type": cp,
            "Resting BP": trestbps,
            "Cholesterol": chol,
            "Fasting Sugar >120": fbs,
            "Resting ECG": restecg,
            "Max HR": thalach,
            "Exercise Angina": exang,
            "Oldpeak": oldpeak,
            "Slope": slope,
            "CA": ca,
            "Thal": thal,
            "Prediction": "Heart Disease" if prediction == 0 else "No Heart Disease",
            "Confidence": f"{prob:.2f}%"
        }

        df_result = pd.DataFrame([result_dict])

        if os.path.exists("prediction_history.csv"):
            df_result.to_csv("prediction_history.csv", mode='a', header=False, index=False)
        else:
            df_result.to_csv("prediction_history.csv", index=False)

        st.info("📄 Saved to prediction_history.csv")

        # Save to SQLite
        conn = sqlite3.connect("predictions.db")
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                name TEXT,
                age INTEGER,
                sex TEXT,
                cp TEXT,
                trestbps INTEGER,
                chol INTEGER,
                fbs TEXT,
                restecg TEXT,
                thalach INTEGER,
                exang TEXT,
                oldpeak REAL,
                slope TEXT,
                ca INTEGER,
                thal TEXT,
                prediction TEXT,
                confidence TEXT
            )
        ''')
        c.execute('''
            INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_name, age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak, slope, ca, thal,
            "Heart Disease" if prediction == 0 else "No Heart Disease",
            f"{prob:.2f}%"
        ))
        conn.commit()
        conn.close()
        st.info("✅ Saved to local database (predictions.db)")

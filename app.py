# app.py (Streamlit)
import streamlit as st
import joblib
import pandas as pd

# Load pre-trained model (replace with your actual model path)
model = joblib.load("diabetes_model.pkl")

# Title
st.title("🛡️ MediShield Health Predictor")
st.subheader("Check your diabetes risk")

# Input form
with st.form("health_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.slider("Pregnancies", 0, 10, 1)
        glucose = st.slider("Glucose (mg/dL)", 50, 200, 100)
        blood_pressure = st.slider("Blood Pressure (mmHg)", 40, 140, 70)
    
    with col2:
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 50, 20)
        insulin = st.slider("Insulin (IU/mL)", 0, 200, 80)
        bmi = st.slider("BMI", 15.0, 50.0, 25.0)
        age = st.slider("Age", 10, 100, 30)

    submitted = st.form_submit_button("Calculate Risk")

# Prediction
if submitted:
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, age]],
                             columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age"])
    
    risk = model.predict_proba(input_data)[0][1] * 100  # Probability of diabetes
    
    # Show result
    st.subheader("Results")
    st.metric(label="Diabetes Risk", value=f"{risk:.1f}%")
    
    # Risk interpretation
    if risk < 30:
        st.success("✅ Low risk - Maintain healthy habits!")
    elif risk < 60:
        st.warning("⚠️ Moderate risk - Consider lifestyle changes.")
    else:
        st.error("🚨 High risk - Consult a doctor soon.")

# Run with: streamlit run app.py
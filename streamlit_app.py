import streamlit as st
import numpy as np
import joblib

# -------------------------------
# Load the trained model
# -------------------------------
MODEL_PATH = "drug-recommender.pkl"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file not found! Please make sure '{MODEL_PATH}' is in the same folder as this app.")
    st.stop()

# -------------------------------
# App title
# -------------------------------
st.set_page_config(page_title="💊 Drug Recommender App", layout="centered")
st.title("💊 Drug Recommender using Machine Learning")
st.write("This app predicts the most suitable **drug** for a patient based on health parameters.")

# -------------------------------
# Collect user input
# -------------------------------
st.header("🧍‍♂️ Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cholesterol = st.selectbox("Cholesterol", ["High", "Normal"])

with col2:
    bp = st.selectbox("Blood Pressure (BP)", ["Low", "Normal", "High"])
    na_to_k = st.number_input("Na_to_K Ratio", min_value=0.0, max_value=50.0, value=15.0, step=0.1)

# -------------------------------
# Encode categorical variables
# -------------------------------
sex_encoded = 1 if sex == "Male" else 0
chol_encoded = 1 if cholesterol == "High" else 0

# BP dummy variables (Normal dropped)
bp_low = 1 if bp == "Low" else 0
bp_high = 1 if bp == "High" else 0

# -------------------------------
# Prepare feature array
# -------------------------------
features = np.array([[age, sex_encoded, na_to_k, chol_encoded, bp_low, bp_high]])

# -------------------------------
# Predict
# -------------------------------
if st.button("🔍 Predict Drug"):
    prediction = model.predict(features)
    st.success(f"💡 Recommended Drug: **{prediction[0]}**")

    st.write("### Input Summary")
    st.json({
        "Age": age,
        "Sex": sex,
        "Blood Pressure": bp,
        "Cholesterol": cholesterol,
        "Na_to_K Ratio": na_to_k
    })

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and Scikit-learn")

import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("drug-recommender.pkl")

# Streamlit UI setup
st.set_page_config(page_title="Drug Recommender App")
st.title("💊 Drug Recommender App")

# Input fields
age = st.number_input("Enter your Age", min_value=0, step=1)
gender = st.selectbox("Select Gender", ["male", "female"])
cholesterol = st.selectbox("Cholesterol Level", ["cholesterolHigh", "cholesterolLow"])
bp = st.selectbox("Blood Pressure Level", ["bpHigh", "bpLow", "bpNormal"])
na_to_k = st.number_input("Enter Na_to_K Value", min_value=0.0, step=0.001, format="%.3f")

# Predict button
if st.button("Predict Drug"):
    try:
        # Encoding input values
        Male = 1 if gender == "male" else 0
        Cholesterol_Status = 1 if cholesterol == "cholesterolHigh" else 0

        if bp == "bpHigh":
            BP_HIGH, BP_LOW = 1, 0
        elif bp == "bpLow":
            BP_HIGH, BP_LOW = 0, 1
        else:  # bpNormal
            BP_HIGH, BP_LOW = 0, 0

        # Prepare feature vector
        features = np.array([[age, Male, Cholesterol_Status, na_to_k, BP_HIGH, BP_LOW]])

        # Make prediction
        prediction = model.predict(features)[0]

        st.success(f"🏥 Recommended Drug: **{prediction}**")

    except Exception as e:
        st.error(f"An error occurred: {e}")

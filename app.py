import streamlit as st
import pickle
import pandas as pd

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("Insurance Charges Prediction")

# User inputs
age = st.number_input("Age", min_value=18, max_value=100)
bmi = st.number_input("BMI")
children = st.number_input("Number of Children", min_value=0, max_value=10)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

# BUTTON (THIS IS THE IMPORTANT PART)
if st.button("Predict Charges"):
    input_data = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex": [sex],
        "smoker": [smoker],
        "region": [region]
    })

    prediction = model.predict(input_data)

    st.success(f"Estimated Insurance Charges: {prediction[0]:.2f}")

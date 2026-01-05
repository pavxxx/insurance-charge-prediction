import streamlit as st
import pandas as pd

st.title("Insurance Charges Prediction")

age = st.number_input("Age", 18, 100)
bmi = st.number_input("BMI")
children = st.number_input("Children", 0, 10)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

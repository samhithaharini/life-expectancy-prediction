import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Life Expectancy Prediction", layout="centered")
st.title("🌍 Life Expectancy Prediction App")

# -----------------------------
# Load your pickle files
# -----------------------------
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)


# -----------------------------
# User Inputs (Simple UI)
# -----------------------------
st.subheader("Enter Input Values")

def float_input(label):
    return st.number_input(label, value=0.0)

def int_input(label):
    return st.number_input(label, value=0)

country = st.text_input("Country", "India")
year = int_input("Year")
status = st.selectbox("Status", ["Developing", "Developed"])
adult_mortality = float_input("Adult Mortality")
infant_deaths = int_input("Infant Deaths")
alcohol = float_input("Alcohol")
percentage_expenditure = float_input("Percentage Expenditure")
hepatitis_b = float_input("Hepatitis B (%)")
measles = int_input("Measles Cases")
bmi = float_input("BMI")
under_five = int_input("Under Five Deaths")
polio = float_input("Polio (%)")
total_exp = float_input("Total Expenditure")
diphtheria = float_input("Diphtheria (%)")
hiv = float_input("HIV/AIDS (%)")
gdp = float_input("GDP")
population = float_input("Population")
thin_10_19 = float_input("Thinness 10–19 years")
thin_5_9 = float_input("Thinness 5–9 years")
income = float_input("Income Composition")
schooling = float_input("Schooling")


# -----------------------------
# Create DF for prediction
# -----------------------------
input_df = pd.DataFrame({
    "Country": [country],
    "Year": [year],
    "Status": [status],
    "Adult Mortality": [adult_mortality],
    "infant deaths": [infant_deaths],
    "Alcohol": [alcohol],
    "percentage expenditure": [percentage_expenditure],
    "Hepatitis B": [hepatitis_b],
    "Measles ": [measles],
    " BMI ": [bmi],
    "under-five deaths ": [under_five],
    "Polio": [polio],
    "Total expenditure": [total_exp],
    "Diphtheria ": [diphtheria],
    " HIV/AIDS": [hiv],
    "GDP": [gdp],
    "Population": [population],
    " thinness  1-19 years": [thin_10_19],
    " thinness 5-9 years": [thin_5_9],
    "Income composition of resources": [income],
    "Schooling": [schooling]
})

# -----------------------------
# Run Prediction
# -----------------------------
if st.button("Predict Life Expectancy"):
    try:
        # Preprocess input
        X_processed = preprocessor.transform(input_df)

        # Select RFE Features
        X_final = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())
        X_final = X_final[selected_features]

        # Predict
        prediction = model.predict(X_final)[0]

        st.success(f"Predicted Life Expectancy: **{prediction:.2f} years**")

    except Exception as e:
        st.error("Error occurred during prediction.")
        st.write(e)

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('heart_model.pkl')

model = load_model()

# App Title and Description
st.title("❤️ Heart Disease Prediction System")
st.markdown("""
This application uses a Machine Learning model (Logistic Regression) to predict the likelihood of heart disease based on 4 key medical attributes.
*Please enter the patient's details below and click 'Predict'.*
""")

# Create a form for user inputs
with st.form("prediction_form"):
    st.subheader("Patient Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=50, step=1)
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        
    with col2:
        cp = st.selectbox(
            "Chest Pain Type (cp)", 
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "1: Typical Angina",
                2: "2: Atypical Angina",
                3: "3: Non-anginal Pain",
                4: "4: Asymptomatic"
            }[x]
        )
        thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150, step=1)

    # Submit button
    submit_button = st.form_submit_button(label="Predict Heart Disease Risk")

# Logic when button is clicked
if submit_button:
    # Prepare input for the model
    input_data = pd.DataFrame([[age, sex, cp, thalach]], columns=['age', 'sex', 'cp', 'thalach'])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    st.markdown("---")
    st.subheader("Prediction Result:")
    
    if prediction[0] == 1:
        st.error("⚠️ **High Risk:** The model predicts a presence of heart disease.")
        st.info("Please consult a cardiologist for a thorough medical evaluation.")
    else:
        st.success("✅ **Low Risk:** The model predicts no heart disease.")
        st.info("Maintain a healthy lifestyle and regular checkups.")

st.markdown("---")
st.caption("Machine Learning Assignment 2 | Deployed using Streamlit")
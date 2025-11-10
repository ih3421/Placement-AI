import streamlit as st
import numpy as np
import joblib

model1 = joblib.load("logistic_model.pkl")
model2 = joblib.load("randomForest_model.pkl")

scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Placement Predictor", page_icon="", layout="centered")
st.title(" Student Placement Probability Predictor")
st.markdown("Enter student details below to estimate placement chances.")

st.subheader(" Student Information")

IQ= st.slider("IQ Score", min_value=70, max_value=160, value=100)
prev_sem_results = st.number_input("Previous Semester Marks", min_value=0, max_value=10, value=7)
cgpa = st.slider("CGPA", 0.0, 10.0, 7.5, 0.1)
Academic_performance= st.number_input("Academic Performance", 0, 10, 6)
internship_experience = st.selectbox("Do you have Internship Experience?", ["Yes", "No"])
if internship_experience == "Yes":
    internship_experience = 1
elif internship_experience == "No":
    internship_experience = 0
extra_curricular = st.slider("What is your rating in Extra-Curricular Activities",0, 10,5 )
comm_skill = st.slider("Communication Skills (1–10)", 1, 10, 7)
projects = st.number_input("Number of Projects", min_value=0, max_value=10, value=2)

# --- Prepare input data ---
features = np.array([[ IQ, prev_sem_results, cgpa, Academic_performance, internship_experience, extra_curricular, comm_skill, projects ]])
scaled_features = scaler.transform(features)

# --- Make prediction ---
placement_prob1 = model1.predict_proba(scaled_features)[0][1]
placement_label1 = " Likely to be Placed" if placement_prob1 >= 0.5 else " Less Likely to be Placed"

placement_prob2 = model2.predict_proba(scaled_features)[0][1]
placement_label2 = " Likely to be Placed" if placement_prob2 >= 0.5 else " Less Likely to be Placed"

# --- Display results ---
model=st.selectbox("Select Model for Prediction", ["Logistic Regression", "Random Forest"])
if model == "Logistic Regression":
    st.subheader(" Prediction Result")
    st.metric(label="Placement Probability", value=f"{placement_prob1*100:.2f}%")
    st.markdown(f"**Prediction:** {placement_label1}")
elif model == "Random Forest":
    st.subheader(" Prediction Result")
    st.metric(label="Placement Probability", value=f"{placement_prob2*100:.2f}%")
    st.markdown(f"**Prediction:** {placement_label2}")

# --- Optional: Add footer ---
st.markdown("---")
st.caption("Built by Ismail Hussain (23261A04E7)")
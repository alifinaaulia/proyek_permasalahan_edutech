import streamlit as st
import numpy as np
import pickle

# Load model, scaler, dan fitur dari folder 'model'
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model/model_features.pkl', 'rb') as f:
    model_features = pickle.load(f)


# Mapping course codes to names
course_options = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening attendance)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening attendance)"
}

st.title("Student Dropout Risk Prediction")

st.markdown("Enter the following information to predict if a student is at risk of dropping out:")

with st.form("student_form"):
    Course = st.selectbox("Course (Program)", options=list(course_options.keys()), format_func=lambda x: course_options[x])
    Age_at_enrollment = st.number_input("Age at Enrollment", min_value=17, max_value=70, value=20)
    Previous_qualification_grade = st.number_input("Previous Qualification Grade", min_value=95.0, max_value=190.0, value=130.0)
    Admission_grade = st.number_input("Admission Grade", min_value=95.0, max_value=190.0, value=130.0)
    Tuition_fees_up_to_date = st.selectbox("Are tuition fees up to date?", ("Yes", "No"))
    Curricular_units_1st_sem_approved = st.number_input("Number of 1st Semester Courses Passed", min_value=0, max_value=26, value=5)
    Curricular_units_1st_sem_grade = st.number_input("Average Grade 1st Semester", min_value=0.0, max_value=20.0, value=10.0)
    Curricular_units_2nd_sem_approved = st.number_input("Number of 2nd Semester Courses Passed", min_value=0, max_value=20, value=5)
    Curricular_units_2nd_sem_grade = st.number_input("Average Grade 2nd Semester", min_value=0.0, max_value=20.0, value=10.0)
    Curricular_units_2nd_sem_evaluations = st.number_input("Number of Evaluations Taken in the 2nd Semester", min_value=0, max_value=33, value=8)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Encode binary for tuition fees
    Tuition_fees_up_to_date = 1 if Tuition_fees_up_to_date == "Yes" else 0

    input_data = np.array([[
        Curricular_units_2nd_sem_approved,
        Curricular_units_2nd_sem_grade,
        Curricular_units_1st_sem_approved,
        Tuition_fees_up_to_date,
        Curricular_units_1st_sem_grade,
        Age_at_enrollment,
        Admission_grade,
        Previous_qualification_grade,
        Curricular_units_2nd_sem_evaluations,
        Course
    ]])

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100  # in percent

    if prediction == 1:
        st.error(f"⚠️ The student is at **risk of dropout**. Probability: {probability:.2f}%")
    else:
        st.success(f"✅ The student is **not at risk of dropout**. Probability: {100 - probability:.2f}%")

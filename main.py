import streamlit as st
from prediction_helper import predict

st.title("Health Insurance Predictor")

categorical_col={
    "insurance_plan":['Bronze','Silver','Gold'],
    "gender":['Male','Female'],
    "marital_status": ['Unmarried','Married'],
    "bmi_category" : ['Normal','Obesity','Overweight','Underweight'],
    "smoking_status" : ['No Smoking','Regular','Occasional'],
    "employment_status" : ['Salaried','Self-Employed','Freelancer'],
    "region" : ['Northwest','Southeast','Northeast','Southwest'],
    "medical_history" : ['Diabetes','High blood pressure','No Disease',
 'Diabetes & High blood pressure','Thyroid','Heart disease',
'High blood pressure & Heart disease','Diabetes & Thyroid',
 'Diabetes & Heart disease']
}

row_1 = st.columns(3)
row_2 = st.columns(3)
row_3 = st.columns(3)
row_4 = st.columns(3)
with row_1[0]:
    age = st.number_input("Age",min_value=18,max_value=100,value=18,step=1)
with row_1[1]:
    number_of_dependants = st.number_input("Number of dependants",min_value=0,max_value=6,step=1)
with row_1[2]:
    income_lakhs = st.number_input("Income in Lakhs",min_value=1,max_value=100,step=1)

with row_2[0]:
    genetical_risk=st.number_input("Genetical Risk",min_value=0,max_value=5,step=1)
with row_2[1]:
    insurance_plan=st.selectbox("Insurance Plan",categorical_col["insurance_plan"])
with row_2[2]:
    bmi_category = st.selectbox("BMI Category",categorical_col["bmi_category"])
with row_3[0]:
    gender = st.selectbox("Gender",categorical_col["gender"])
with row_3[1]:
    smoking_status = st.selectbox("Smoking Status",categorical_col["smoking_status"])
with row_3[2]:
    region = st.selectbox("Region",categorical_col["region"])

with row_4[0]:
    marital_status = st.selectbox("Marital Status",categorical_col["marital_status"])
with row_4[1]:
    employment_status=st.selectbox("Employment Status",categorical_col["employment_status"])
with row_4[2]:
    medical_history=st.selectbox("Medical History",categorical_col["medical_history"])


if st.button("Predict"):
    gather_info={
        "age":age,
        "number_of_dependants":number_of_dependants,
        "income_lakhs":income_lakhs,
        "genetical_risk":genetical_risk,
        "insurance_plan":insurance_plan,
        "bmi_category":bmi_category,
        "gender":gender,
        "smoking_status":smoking_status,
        "employment_status":employment_status,
        "medical_history":medical_history,
        "region":region,
        "marital_status":marital_status,
    }
    p=predict(gather_info)
    st.success(f"Predicted Premium: ₹{p:.2f}")

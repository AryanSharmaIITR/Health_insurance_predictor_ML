import pandas as pd
from joblib import load
import logging

# Setup logging
logging.basicConfig(
    filename='activity.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load models and scalers
model_adult = load("artifacts/model_xgb_adult.joblib")
model_young = load("artifacts/model_linear_young.joblib")
scaler_adult = load("artifacts/scaler_rest.joblib")
scaler_young = load("artifacts/scaler_linear_young.joblib")

# Lookup tables
plans = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
risk = {
    "Diabetes": 6, "High blood pressure": 6, "Thyroid": 6,
    "Heart disease": 8, "No Disease": 1
}
bmi_category_score = {"Normal": 2, "Underweight": 1, "Overweight": 3, "Obesity": 4}
smoking_map = {"No Smoking": 1, "Occasional": 2, "Regular": 3}

def mh(medical_history):
    """Calculate total disease risk from text"""
    return sum(risk[k] for k in risk if k.lower() in medical_history.lower())

def final_Score(medical_history, smoking_status):
    rs = mh(medical_history)
    return rs * smoking_map.get(smoking_status, 1)

def validate_input(data):
    """Ensure safe defaults and valid values"""
    data['number_of_dependants'] = max(0, data.get('number_of_dependants', 0))
    data['insurance_plan'] = data.get('insurance_plan', 'Bronze')
    data['smoking_status'] = data.get('smoking_status', 'No Smoking')
    data['bmi_category'] = data.get('bmi_category', 'Normal')
    data['gender'] = data.get('gender', 'Male')
    data['marital_status'] = data.get('marital_status', 'Unmarried')
    data['region'] = data.get('region', 'Northwest')
    data['employment_status'] = data.get('employment_status', 'Salaried')
    return data

def encode_region(df, region):
    for r in ['Northwest', 'Southeast', 'Southwest']:
        df[f'region_{r}'] = int(region == r)

def encode_employment(df, status):
    df['employment_status_Salaried'] = int(status == 'Salaried')
    df['employment_status_Self-Employed'] = int(status == 'Self Employed')

def young(input_data):
    input_data = validate_input(input_data)
    columns = [
        'age', 'number_of_dependants', 'smoking_status', 'income_lakhs',
        'medical_history', 'insurance_plan', 'genetical_risk', 'gender_Male',
        'marital_status_Unmarried', 'region_Northwest', 'region_Southeast',
        'region_Southwest', 'bmi_category_Obesity', 'bmi_category_Overweight',
        'bmi_category_Underweight', 'employment_status_Salaried',
        'employment_status_Self-Employed'
    ]

    df = pd.DataFrame(0, columns=columns, index=[0])
    df.at[0, 'age'] = input_data['age']
    df.at[0, 'number_of_dependants'] = input_data['number_of_dependants']
    df.at[0, 'income_lakhs'] = input_data['income_lakhs']
    df.at[0, 'insurance_plan'] = plans.get(input_data['insurance_plan'], 1)
    df.at[0, 'smoking_status'] = {'No Smoking': 0, 'Occasional': 1, 'Regular': 2}.get(input_data['smoking_status'], 0)
    df.at[0, 'medical_history'] = mh(input_data['medical_history'])
    df.at[0, 'genetical_risk'] = input_data['genetical_risk']
    df.at[0, 'gender_Male'] = int(input_data['gender'] == 'Male')
    df.at[0, 'marital_status_Unmarried'] = int(input_data['marital_status'] == 'Unmarried')

    # Region & employment
    encode_region(df, input_data['region'])
    encode_employment(df, input_data['employment_status'])

    # BMI categories
    for cat in ['Obesity', 'Overweight', 'Underweight']:
        df[f'bmi_category_{cat}'] = int(input_data.get('bmi_category', 'Normal') == cat)

    scaler = scaler_young['scaler']
    cols = scaler_young['cols_to_scale']
    df[cols] = scaler.transform(df[cols])
    return df

def adult(input_data):
    input_data = validate_input(input_data)
    columns = [
        'age', 'number_of_dependants', 'bmi_category', 'income_lakhs',
        'insurance_plan', 'score', 'gender_Male', 'region_Northwest',
        'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    df = pd.DataFrame(0, columns=columns, index=[0])
    df.at[0, 'age'] = input_data['age']
    df.at[0, 'number_of_dependants'] = input_data['number_of_dependants']
    df.at[0, 'bmi_category'] = bmi_category_score.get(input_data['bmi_category'], 2)
    df.at[0, 'income_lakhs'] = input_data['income_lakhs']
    df.at[0, 'insurance_plan'] = plans.get(input_data['insurance_plan'], 1)
    df.at[0, 'score'] = final_Score(input_data['medical_history'], input_data['smoking_status'])
    df.at[0, 'gender_Male'] = int(input_data['gender'] == 'Male')
    df.at[0, 'marital_status_Unmarried'] = int(input_data['marital_status'] == 'Unmarried')

    # Region & employment
    encode_region(df, input_data['region'])
    encode_employment(df, input_data['employment_status'])

    scaler = scaler_adult['scaler']
    cols = scaler_adult['cols_to_scale']
    df[cols] = scaler.transform(df[cols])
    return df

def predict(input_data):
    input_data = validate_input(input_data)

    if input_data['age'] <= 25:
        input_df = young(input_data)
        model = model_young
    else:
        input_df = adult(input_data)
        model = model_adult

    prediction = model.predict(input_df)
    logging.info("Input Data:\n%s", input_df.to_string())

    logging.info("Prediction: %.2f", prediction)

    return float(prediction)

import pandas as pd
import joblib

MODEL_YOUNG_PATH = "artifacts/model_young.joblib"
MODEL_REST_PATH = "artifacts/model_rest.joblib"
SCALER_YOUNG_PATH = "artifacts/scaler_young.joblib"
SCALER_REST_PATH = "artifacts/scaler_rest.joblib"


def load_artifacts():
    model_young = joblib.load(MODEL_YOUNG_PATH)
    model_rest = joblib.load(MODEL_REST_PATH)
    scaler_young = joblib.load(SCALER_YOUNG_PATH)
    scaler_rest = joblib.load(SCALER_REST_PATH)
    return model_young, model_rest, scaler_young, scaler_rest


model_young, model_rest, scaler_young, scaler_rest = load_artifacts()


def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }

    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)
    normalized_risk_score = total_risk_score / 14
    return normalized_risk_score


def preprocess_input(input_dict):
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight',
        'smoking_status_Occasional', 'smoking_status_Regular',
        'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    df['age'] = input_dict.get('Age', 0)
    df['number_of_dependants'] = input_dict.get('Number of Dependants', 0)
    df['income_lakhs'] = input_dict.get('Income in Lakhs', 0)
    df['genetical_risk'] = input_dict.get('Genetical Risk', 1)
    df['insurance_plan'] = insurance_plan_encoding.get(input_dict.get('Insurance Plan', 'Bronze'), 1)

    if input_dict.get('Gender') == 'Male':
        df['gender_Male'] = 1

    if input_dict.get('Region') == 'Northwest':
        df['region_Northwest'] = 1
    elif input_dict.get('Region') == 'Southeast':
        df['region_Southeast'] = 1
    elif input_dict.get('Region') == 'Southwest':
        df['region_Southwest'] = 1

    if input_dict.get('Marital Status') == 'Unmarried':
        df['marital_status_Unmarried'] = 1

    if input_dict.get('BMI Category') == 'Obesity':
        df['bmi_category_Obesity'] = 1
    elif input_dict.get('BMI Category') == 'Overweight':
        df['bmi_category_Overweight'] = 1
    elif input_dict.get('BMI Category') == 'Underweight':
        df['bmi_category_Underweight'] = 1

    if input_dict.get('Smoking Status') == 'Occasional':
        df['smoking_status_Occasional'] = 1
    elif input_dict.get('Smoking Status') == 'Regular':
        df['smoking_status_Regular'] = 1

    if input_dict.get('Employment Status') == 'Salaried':
        df['employment_status_Salaried'] = 1
    elif input_dict.get('Employment Status') == 'Self-Employed':
        df['employment_status_Self-Employed'] = 1

    df['normalized_risk_score'] = calculate_normalized_risk(input_dict.get('Medical History', 'none'))

    df = handle_scaling(input_dict.get('Age', 0), df)
    return df


def handle_scaling(age, df):
    scaler_object = scaler_young if age <= 25 else scaler_rest
    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    df['income_level'] = 0
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df.drop('income_level', axis=1, inplace=True)

    return df


def predict(input_dict):
    input_df = preprocess_input(input_dict)
    prediction = (
        model_young.predict(input_df)
        if input_dict.get('Age', 0) <= 25
        else model_rest.predict(input_df)
    )
    return int(prediction[0])

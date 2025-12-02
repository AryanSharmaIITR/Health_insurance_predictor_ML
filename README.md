# Health_insurance_predictor_ML
Machine Learning Project For Insurance amount prediction

A **Streamlit-based web application** that predicts health insurance premiums based on user input. The prediction leverages machine learning models trained on relevant health and demographic data.

---

## 🚀 Features

- 📝 User-friendly form for inputting personal, health, and lifestyle details  
- ⚡ Predicts insurance premium instantly  
- 🔄 Handles categorical and numerical features  
- 🤖 Uses pre-trained ML models for adults and young users

---

## 💡 How It Works

1. Users enter their information (age, income, BMI category, medical history, etc.) on the web interface  
2. The app encodes and scales the input data  
3. Depending on the user's age, the appropriate model is used for prediction  
4. The predicted premium is displayed on the screen

---

## 📁 File Structure

main.py # Streamlit application UI and logic
prediction_helper.py # Data preprocessing, encoding, and prediction logic
artifacts/
├── model_xgb_adult.joblib # Model for adults
├── model_linear_young.joblib # Model for young users
├── scaler_rest.joblib # Scaler for general preprocessing
└── scaler_linear_young.joblib # Scaler for young users




## ⚙️ Getting Started

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_folder>
```
### 2.Install Dependencies
pip install -r requirements.txt

### 3. Run the Application
streamlit run main.py

### 4. Usage
Open the provided local URL in your browser
Fill in the required fields
Click Predict to see your estimated premium
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== LOAD MODEL & ARTIFACTS ====================
@st.cache_resource
def load_model_artifacts():
    """Load trained model, scaler, and feature names"""
    models_dir = Path(__file__).parent / 'saved_models'
    
    model_path = models_dir / 'diabetes_model.pkl'
    scaler_path = models_dir / 'scaler.pkl'
    feature_names_path = models_dir / 'feature_names.pkl'
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_names_path)
    
    return model, scaler, feature_names

# Load artifacts
dt_model, scaler, feature_names = load_model_artifacts()

# ==================== UTILITY FUNCTIONS ====================
def make_prediction(input_data):
    """Make prediction with the trained model"""
    # Convert to DataFrame
    df = pd.DataFrame([input_data], columns=feature_names)
    
    # Scale the features
    df_scaled = scaler.transform(df)
    
    # Get prediction and probability
    prediction = dt_model.predict(df_scaled)[0]
    probability = dt_model.predict_proba(df_scaled)[0]
    
    # Calculate risk percentage
    diabetes_risk = probability[1] * 100
    
    return prediction, diabetes_risk, probability

def get_risk_level(risk_percentage):
    """Categorize risk level based on percentage"""
    if risk_percentage < 30:
        return "LOW"
    elif risk_percentage < 60:
        return "MODERATE"
    else:
        return "HIGH"

# ==================== HEADER ====================
st.title("Diabetes Risk Prediction System")
st.markdown("""
**Powered by Machine Learning | Random Forest (GridSearchCV Optimized)**

This application uses an optimized Random Forest ensemble model trained on SMOTE-balanced data 
to predict diabetes risk with 77.4% accuracy and 80% disease detection rate.
""")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("Input Method")
    input_method = st.radio("Select how to input data:", ["Manual Input", "Use Example Profile"])
    
    st.divider()
    st.header("About This Model")
    st.info("""
    **Model Details:**
    - **Algorithm:** Random Forest (100 trees)
    - **Optimization:** GridSearchCV with 5-Fold CV
    - **Data Handling:** SMOTE-balanced (class imbalance)
    - **Accuracy:** 77.40%
    - **Precision:** 63.49%
    - **Recall (Disease Detection):** 80.00%
    - **F1-Score:** 0.708
    - **Training Samples:** 582 (SMOTE-balanced to 764)
    - **Test Samples:** 146
    - **Features:** 8 health metrics
    
    **Top Predictive Features:**
    1. Glucose (29.5%)
    2. BMI (17.3%)
    3. Age (16.3%)
    4. Diabetes Pedigree Function (9.9%)
    5. Pregnancies (8.7%)
    """)

# ==================== MAIN CONTENT ====================
if input_method == "Manual Input":
    st.header("Enter Patient Health Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.slider(
            "Pregnancies",
            min_value=0,
            max_value=17,
            value=1,
            help="Number of pregnancies "
        )
        
        glucose = st.slider(
            "Glucose (mg/dL)",
            min_value=0,
            max_value=200,
            value=120,
            help="Fasting blood glucose level"
        )
        
        blood_pressure = st.slider(
            "Blood Pressure (mm Hg)",
            min_value=0,
            max_value=122,
            value=70,
            help="Diastolic blood pressure"
        )
        
        skin_thickness = st.slider(
            "Skin Thickness (mm)",
            min_value=0,
            max_value=99,
            value=20,
            help="Triceps skin fold thickness"
        )
    
    with col2:
        insulin = st.slider(
            "Insulin (mu U/ml)",
            min_value=0,
            max_value=846,
            value=80,
            help="2-Hour serum insulin level"
        )
        
        bmi = st.slider(
            "BMI (kg/m²)",
            min_value=0.0,
            max_value=70.0,
            value=25.0,
            step=0.1,
            help="Body Mass Index"
        )
        
        diabetes_pedigree = st.slider(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=2.5,
            value=0.5,
            step=0.05,
            help="Genetic predisposition score"
        )
        
        age = st.slider(
            "Age (years)",
            min_value=21,
            max_value=81,
            value=35,
            help="Patient age"
        )
    
    # Prepare input data
    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, 
                  insulin, bmi, diabetes_pedigree, age]

else:  # Use Example Profile
    st.header("Select Example Patient Profile")
    
    examples = {
        "Healthy Profile": {
            "Pregnancies": 1,
            "Glucose": 100,
            "BloodPressure": 70,
            "SkinThickness": 20,
            "Insulin": 50,
            "BMI": 24.0,
            "DiabetesPedigreeFunction": 0.3,
            "Age": 30
        },
        "Moderate Risk": {
            "Pregnancies": 3,
            "Glucose": 140,
            "BloodPressure": 82,
            "SkinThickness": 28,
            "Insulin": 150,
            "BMI": 28.5,
            "DiabetesPedigreeFunction": 0.7,
            "Age": 45
        },
        "High Risk": {
            "Pregnancies": 6,
            "Glucose": 180,
            "BloodPressure": 95,
            "SkinThickness": 35,
            "Insulin": 400,
            "BMI": 35.5,
            "DiabetesPedigreeFunction": 1.5,
            "Age": 55
        }
    }
    
    selected_profile = st.selectbox("Choose a profile:", list(examples.keys()))
    profile_data = examples[selected_profile]
    
    # Display profile metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pregnancies", profile_data["Pregnancies"])
        st.metric("Glucose", f"{profile_data['Glucose']} mg/dL")
    with col2:
        st.metric("Blood Pressure", f"{profile_data['BloodPressure']} mm Hg")
        st.metric("Skin Thickness", f"{profile_data['SkinThickness']} mm")
    with col3:
        st.metric("Insulin", f"{profile_data['Insulin']} mu U/ml")
        st.metric("BMI", f"{profile_data['BMI']} kg/m²")
    with col4:
        st.metric("Diabetes Pedigree", profile_data["DiabetesPedigreeFunction"])
        st.metric("Age", f"{profile_data['Age']} years")
    
    input_data = [
        profile_data["Pregnancies"],
        profile_data["Glucose"],
        profile_data["BloodPressure"],
        profile_data["SkinThickness"],
        profile_data["Insulin"],
        profile_data["BMI"],
        profile_data["DiabetesPedigreeFunction"],
        profile_data["Age"]
    ]

# ==================== PREDICTION ====================
st.divider()

# Make prediction
prediction, diabetes_risk, probability = make_prediction(input_data)
risk_level = get_risk_level(diabetes_risk)

# Display results
col1, col2, col3 = st.columns(3)

with col1:
    prediction_text = "DIABETES DETECTED" if prediction == 1 else "NO DIABETES"
    st.metric("Prediction", prediction_text)

with col2:
    st.metric("Risk Level", risk_level)

with col3:
    st.metric("Diabetes Risk", f"{diabetes_risk:.2f}%")

# Detailed result card
st.divider()
st.subheader("Detailed Analysis")

result_col1, result_col2 = st.columns(2)

with result_col1:
    st.write("**Probability Distribution:**")
    prob_df = pd.DataFrame({
        "Outcome": ["No Diabetes", "Diabetes"],
        "Probability": [probability[0] * 100, probability[1] * 100]
    })
    st.bar_chart(prob_df.set_index("Outcome"))

with result_col2:
    st.write("**Input Metrics Summary:**")
    metrics_df = pd.DataFrame({
        "Metric": feature_names,
        "Value": input_data
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# ==================== RECOMMENDATIONS ====================
st.divider()
st.subheader("Health Recommendations")

if diabetes_risk < 30:
    st.success("""
    **LOW RISK PROFILE**
    
    Based on the analysis, this patient profile shows a **low diabetes risk**. Continue with:
    - Regular health checkups
    - Balanced diet and exercise
    - Monitor glucose levels periodically
    - Maintain healthy BMI
    """)
elif diabetes_risk < 60:
    st.warning("""
    **MODERATE RISK PROFILE**
    
    This patient profile indicates **moderate diabetes risk**. Recommendations:
    - Schedule regular medical consultations
    - Increase physical activity (150+ min/week)
    - Improve diet (reduce sugar and processed foods)
    - Monitor blood glucose levels regularly
    - Check weight and BMI trends
    """)
else:
    st.error("""
    **HIGH RISK PROFILE**
    
    This patient profile shows **high diabetes risk**. Immediate actions:
    - Consult healthcare provider urgently
    - Get comprehensive metabolic panel testing
    - Consider diabetes screening tests 
    - Implement lifestyle modifications
    - Follow professional medical guidance
    """)

# ==================== FOOTER ====================
st.divider()
# st.info("""
# **Model Information:**
# - **Algorithm:** Random Forest Classifier (100 trees)
# - **Optimization:** GridSearchCV with 5-Fold Cross-Validation
# - **Data Handling:** SMOTE-balanced (handles class imbalance)
# - **Test Accuracy:** 77.40%
# - **Test Precision:** 63.49% (correct positive predictions)
# - **Test Recall:** 80.00% (catches diabetes cases)
# - **F1-Score:** 0.708 (balanced precision-recall metric)
# - **Features:** 8 health metrics (Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age)
# - **Top Predictors:** 
#   1. Glucose (29.5% importance)
#   2. BMI (17.3% importance)
#   3. Age (16.3% importance)

# **Important Notes:**
# 1. This is a **prototype for educational purposes only**
# 2. Not validated for clinical use
# 3. Should not replace professional medical diagnosis
# 4. Based on Pima Indians Diabetes Dataset (728 samples after cleaning)
# 5. Always consult healthcare professionals for medical decisions
# 6. 80% recall prioritizes disease detection (minimizes missed cases)
# """)

st.caption("Diabetes Risk Prediction System | Built with Streamlit & Machine Learning")

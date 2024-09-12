# # Streamlit Application
import numpy as np
import streamlit as st
import joblib
import pandas as pd
from model import CustomPipeline

# Load the model
pipeline = joblib.load('model/pipeline.pkl')

# Chargement des données
data = pd.read_csv('datasets/enhanced_feature_engineering_data.csv')

# Imputation des valeurs manquantes
cols_to_impute_with_mean = ['prepayment', 'monthly_income', 'DTI_fraction']
for col in cols_to_impute_with_mean:
    data[col] = pd.to_numeric(data[col])
    data[col].fillna(data[col].mean())

# Séparation des caractéristiques et des cibles
X = data.drop(['EverDelinquent', 'prepayment'], axis=1)
y_class = data['EverDelinquent']
y_reg = data['prepayment']

# Sélection des caractéristiques
X_selected = X[['DTI', 'OrigUPB', 'OrigInterestRate', 'monthly_rate', 'monthly_payment',
                'total_payment', 'interest_amount', 'cur_principal', 'DTI_fraction',
                'monthly_income']]

# Streamlit page configuration
st.set_page_config(page_title="Mortgage Prediction Application", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .stButton > button {
            background-color: #4CAF50; /* Green */
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 5px;
        }
        .stButton > button:hover {
            background-color: #45a049; /* Darker green */
        }
        .stNumberInput > div {
            margin-bottom: 10px;
        }
        .stSelectbox > div {
            margin-bottom: 10px;
        }
        .stTitle {
            font-size: 36px;
            font-weight: bold;
            color: #333;
        }
        .stHeader {
            font-size: 28px;
            font-weight: bold;
            color: #555;
            margin-top: 20px;
        }
        .stWrite {
            font-size: 16px;
            color: #666;
        }
    </style>
""", unsafe_allow_html=True)

st.title(":blue[Mortgage Prediction Application]")

# Collect user inputs
st.header(":green[Enter Details]")

# Input fields for the simplified dataframe
col1, col2 = st.columns(2)

with col1:
    DTI = st.sidebar.slider('DTI', float(X_selected['DTI'].min()), float(
        X_selected['DTI'].max()), float(X_selected['DTI'].mean()))
    OrigUPB = st.sidebar.slider('OrigUPB', float(X_selected['OrigUPB'].min()), float(
        X_selected['OrigUPB'].max()), float(X_selected['OrigUPB'].mean()))
    OrigInterestRate = st.sidebar.slider('OrigInterestRate', float(X_selected['OrigInterestRate'].min(
    )), float(X_selected['OrigInterestRate'].max()), float(X_selected['OrigInterestRate'].mean()))
    monthly_rate = st.sidebar.slider('monthly_rate', float(X_selected['monthly_rate'].min(
    )), float(X_selected['monthly_rate'].max()), float(X_selected['monthly_rate'].mean()))
    monthly_payment = st.sidebar.slider('monthly_payment', float(X_selected['monthly_payment'].min(
    )), float(X_selected['monthly_payment'].max()), float(X_selected['monthly_payment'].mean()))

with col2:

    total_payment = st.sidebar.slider('total_payment', float(X_selected['total_payment'].min(
    )), float(X_selected['total_payment'].max()), float(X_selected['total_payment'].mean()))
    interest_amount = st.sidebar.slider('interest_amount', float(X_selected['interest_amount'].min(
    )), float(X_selected['interest_amount'].max()), float(X_selected['interest_amount'].mean()))
    cur_principal = st.sidebar.slider('cur_principal', float(X_selected['cur_principal'].min(
    )), float(X_selected['cur_principal'].max()), float(X_selected['cur_principal'].mean()))
    DTI_fraction = st.sidebar.slider('DTI_fraction', float(X_selected['DTI_fraction'].min(
    )), float(X_selected['DTI_fraction'].max()), float(X_selected['DTI_fraction'].mean()))
    monthly_income = st.sidebar.slider('monthly_income', float(X_selected['monthly_income'].min(
    )), float(X_selected['monthly_income'].max()), float(X_selected['monthly_income'].mean()))

# Create a DataFrame from user inputs
user_input_df = pd.DataFrame([{
    'DTI': DTI,
    'OrigUPB': OrigUPB,
    'OrigInterestRate': OrigInterestRate,
    'monthly_rate': monthly_rate,
    'monthly_payment': monthly_payment,
    'total_payment': total_payment,
    'interest_amount': interest_amount,
    'cur_principal': cur_principal,
    'DTI_fraction': DTI_fraction,
    'monthly_income': monthly_income
}])

# Display the DataFrame to the user
st.write("### :orange[User Input DataFrame:]")
st.write(user_input_df)

# Predict and display results
if st.button('Predict Classification and Regression'):
    try:
        # Get classification and regression predictions
        y_class_pred, y_reg_pred = pipeline.predict(user_input_df)

        # Display classification result
        st.subheader("Prediction Results")
        st.write(
            f"**Classification Prediction (Ever Delinquent):** {y_class_pred[0]}")

        # Check if regression data exists
        if y_reg_pred.size > 0:
            st.write(
                f"**Regression Prediction (Prepayment):** {abs(y_reg_pred[0])}")
        else:
            st.write(
                "No data available for regression prediction as classification result is negative.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

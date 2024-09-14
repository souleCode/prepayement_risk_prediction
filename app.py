import streamlit as st
import requests
import joblib
import pandas as pd
import numpy as np
from model import CustomPipelineWithFeatureSelection


# load model
with open('model/pipeline.pkl', 'rb') as f:
    model_pipeline = joblib.load(f)

st.set_page_config(
    page_title="msb-app", page_icon=":star:")

# st.title('MSB-Mortgage-Prediction App')

st.markdown(
    """
    <style>
    .title {
        background: linear-gradient(to right, #ff7e5f, #feb47b);  /* linear gradient */
        -webkit-background-clip: text;
        color: transparent;
        font-size: 50px;
        font-weight: bold;
        font-family: 'Courier New', Courier, monospace;  /* change font */
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True
)
st.markdown('<h1 class="title">MSB-Mortgage Prepayment Prediction App</h1>',
            unsafe_allow_html=True)
st.header('Enter your own details here for predictions')

# Formulaire pour entrer les données avec des clés uniques
input_data = {
    'CreditScore': st.text_input('Enter the Credit Score', key='credit_score'),
    'Units': st.selectbox('Choose the Unit', [0, 1], key='unit_selector'),
    'PropertyType': st.text_input('Enter the PropertyType', key='property_type'),
    'OrigLoanTerm': st.text_input('Enter the OrigLoanTerm', key='orig_loanTerm'),
    'NumBorrowers': st.text_input('Enter the NumBorrowers', key='num_borrowers'),
    'MonthsDelinquent': st.selectbox('Choose the MonthsDelinquent', [0, 1], key='months_delinquent'),
    'MonthsInRepayment': st.text_input('Enter the MonthsInRepayment', key='months_in_repayment'),
    'Occupancy_O': st.selectbox('Choose the Occupancy', [0, 'juste 0'], key='occupancy_o'),
    'MonthlyIncome': st.text_input('Enter the MonthlyIncome', key='monthly_income'),
    'InterestAmount': st.text_input('Enter the InterestAmount', key='interest_amount'),
    'Totalpayment': st.text_input('Enter the Totalpayment', key='total_payment'),
    'MonthlyInstallment': st.text_input('Enter the MonthlyInstallment', key='monthly_installment'),
    'OrigUPB': st.text_input('Enter the OrigUPB', key='orig_upb'),
    'CurrentUPB': st.text_input('Enter the CurrentUPB', key='current_upb'),
    'DTI': st.text_input('Enter the DTI', key='dti'),
    'LoanSeqNum': st.text_input('Enter the LoanSeqNum', key='loan_seq_num'),
    'FirstTimeHomebuyer': st.selectbox('Choose the FirstTimeHomebuyer', [0, 1], key='first_time_homebuyer'),
}

# Bouton pour soumettre les données avec une clé unique
if st.button('Faire une prédiction', key='predict_button'):
    df = pd.DataFrame([input_data])

    y_class_pred, y_reg_pred = model_pipeline.predict(df)
    st.subheader('Results of MSB-Mortgage  Prepayment Prediction App')
    if y_class_pred == 1:
        st.write('classification : EverDeliquent')
        # rounded_prediction = round(y_reg_pred, 2)
        st.write('Prediction of prepayment risk:', round(y_reg_pred[0], 2))
    else:
        st.write(
            'Prediction of classification :Not EverDeliquent, Prepayement risk prediction not needed')

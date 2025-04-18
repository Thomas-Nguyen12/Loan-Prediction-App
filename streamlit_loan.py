import streamlit as st 
import pandas as pd 
import numpy as np 
import scipy.stats as stats 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px 
import joblib
from sklearn.preprocessing import LabelEncoder 
from category_encoders import WOEEncoder
from streamlit_shap import st_shap
import shap

xgboost_model = joblib.load("model/optimised_xgboost_model.pkl")
random_forest_model = joblib.load("model/optimised_random_forest_model.pkl")
logistic_regression_model = joblib.load("model/optimised_logistic_model.pkl")
df2 = pd.read_csv("data/cleaned_loan_dataset.csv")
df = pd.read_csv("data/loan_dataset.csv") 
le = joblib.load("label_encoder.pkl")
property_area_woe = joblib.load('property_area_woe.pkl')
shap_values = joblib.load("random_forest_shap_values.pkl")
X_test = joblib.load("X_test.pkl")
st.title("Loan Approval Prediction Website") 


with st.container(): 
    st.dataframe(df)

st.divider() 

predictor_tab, shap_explanation = st.tabs(['predictors', 'explanation'])
with predictor_tab:
    left_column, right_column = st.columns(2) 

    with left_column: 
        # Predictors will go here
        # in this website, I am including the predictors: 

        # 1. Credit history
        # 2. Property area
        # 3. education 
        # 4. loan amount 
        # 5. loan amount term
        # 6. applicant income 
        # 7. coapplicant income


        st.header("Model Predictors")
        credit_history = st.selectbox('Does your credit history meet guidelines?', ('Yes', 'No'))
        property_area = st.selectbox("What kind of are do you live in?", ("Urban", "Rural", 'Semiurban'))
        education = st.selectbox('What is your education level?', ("Graduate", 'Not Graduate'))
        loan_amount = st.slider("Loan Amount (£)", 0, 100000000, 100)
        loan_amount_term = st.slider("Loan Amount Term (days)", 0, 500, 10)

        applicant_income = st.slider('Applicant Income (£)', 0, 100000000, 100)
        coapplicant_income = st.slider('Co-applicant Income (£)', 0, 100000000, 100)

        


        ## Preprocessing will happen here

        # 
        
        credit_history_encoded_values = {
            "Yes": 1,
            "No": 0,
        }
        credit_history_encoded = credit_history_encoded_values[credit_history]

        
        

        # making a dataframe of variables

        property_area_encoded_values = {
            'Rural': -0.353492,
            'Urban': -0.175678,
            'Semiurban': 0.44892948,
        }



        property_area_encoded = property_area_encoded_values[property_area]

        education_encoded_values = {
            'Graduate': 0,
            'Not Graduate': 1
        }
        education_encoded = education_encoded_values[education]
        

    with right_column: 
        # Output will go here
        st.header("Model Prediction")
        st.write(f"""Currently selected values:
        {credit_history},
        {property_area}, 
        {education}, 
        {loan_amount}, 
        {loan_amount_term},
        {applicant_income}, 
        {coapplicant_income}
        """)
        
        predictors = np.array([credit_history_encoded, property_area_encoded, education_encoded, loan_amount, loan_amount_term,
        applicant_income, coapplicant_income])
        predictors2 = np.reshape(predictors, (1, -1))
        print (f"prediction: {random_forest_model.predict(predictors2)}")

        # I need to convert the prediction to a text 
        # Y = 0
        # N = 1
        st.write("Loan Approved = 0") 
        st.write("Loan Rejected = 1")
        st.write ("==== Prediction ====")

        st.write(f"prediction {random_forest_model.predict(predictors2)}")
        





with shap_explanation: 
    st_shap(shap.plots.beeswarm(shap_values[:, :, 0]))

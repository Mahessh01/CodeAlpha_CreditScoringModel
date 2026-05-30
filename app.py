import streamlit as st
import numpy as np
import joblib


model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Credit Scoring System")
st.write("Enter customer financial details to predict credit risk")

RevolvingUtilizationOfUnsecuredLines = st.number_input("Revolving Utilization")
age = st.number_input("Age")
NumberOfTime30_59DaysPastDueNotWorse = st.number_input("30-59 Days Past Due")
DebtRatio = st.number_input("Debt Ratio")
MonthlyIncome = st.number_input("Monthly Income")
NumberOfOpenCreditLinesAndLoans = st.number_input("Open Credit Lines")
NumberOfTimes90DaysLate = st.number_input("90 Days Late")
NumberRealEstateLoansOrLines = st.number_input("Real Estate Loans")
NumberOfTime60_89DaysPastDueNotWorse = st.number_input("60-89 Days Past Due")
NumberOfDependents = st.number_input("Dependents")

# Predict button
if st.button("Predict Credit Risk"):
    
    input_data = np.array([[
        RevolvingUtilizationOfUnsecuredLines,
        age,
        NumberOfTime30_59DaysPastDueNotWorse,
        DebtRatio,
        MonthlyIncome,
        NumberOfOpenCreditLinesAndLoans,
        NumberOfTimes90DaysLate,
        NumberRealEstateLoansOrLines,
        NumberOfTime60_89DaysPastDueNotWorse,
        NumberOfDependents
    ]])

    # scale input
    input_scaled = scaler.transform(input_data)

    # prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.error("High Credit Risk")
    else:
        st.success("Low Credit Risk")

    st.write("Probability of Risk:", probability[0][1])
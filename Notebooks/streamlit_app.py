import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import joblib  # Import joblib to load the saved model
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# Load the saved model
model = joblib.load('best_bagging_model.pkl')

# Page title
st.set_page_config(page_title='Credit Score Classification', page_icon='📊')
st.title('📊 Credit Score Classification')

with st.expander('About this app'):
    st.markdown('**What can this app do?**')
    st.info('This app allows you to learn more about your credit! After entering some basic information, this app will classify your credit as either Good, Standard, or Poor.')
    st.markdown('**How to use the app?**')
    st.warning('To engage with the app, 1. Click on each of the categories to learn more about what they mean, and how to calculate them. 2. Enter appropriate values for each of the categories. 3. Click SUBMIT to reveal your credit score classification!')

with st.expander('Learn More About Our Categories'):
    st.markdown('**Age**') 
    st.info('Your current age.')
    st.markdown('**Monthly In Hand Salary**') 
    st.info('The monthly base salary you receive, after taxes have been taken out.')
    st.markdown('**Interest Rate**') 
    st.info('The average interest rate on your credit cards. This can be calulated by adding up the interest rate values for each card, and dividing by the total number of cards in possession.')
    st.markdown('**Delay from Due Date**') 
    st.info('The average number of days you are delayed on making payments to debtholders. If you always pay on time, enter 0.')
    st.markdown('**Number of Delayed Payments**') 
    st.info('The average number of delayed payments you make in a given year.')
    st.markdown('**Changed Credit Limit**') 
    st.info('The percentage change in your credit card limit in a given year. To calculate this, subtract your initial limit from your most recent limit, and multiply by 100. Do this for all credit cards, and take the average of these percentages.')
    st.markdown('**Credit Mix**') 
    st.info('Your best estimate of the classification mix of your credit. Use the dropdown to specify either Good, Standard, or Poor. For instance, if you have mortgages, credit cards, auto loans, personal loans, etc., select Good. If you only have one type of debt, select Poor. In other cases, select Standard.')
    st.markdown('**Outstanding Debt**') 
    st.info('The remaining debt you have to pay in USD.')
    st.markdown('**Credit Utilization Ratio**') 
    st.info('The average utilization ratio of your credit cards. To calculate this, divide the total amount of credit you are currently using by the total amount of credit available to you.')
    st.markdown('**Credit History Age**') 
    st.info('How long you have been establishing credit (typically how long you have owned a credit card). Enter months as a decimal. For instance, if you have had credit for 2 years and 3 months, enter 2.25.')
    st.markdown('**Total EMI Per Month**') 
    st.info('Your monthly EMI payments in USD. To calculate this, divide your loan amount by the number of months you will be repaying. Then add the monthly interest rate amount to it. Do this for each of your debts and add them up, to get the total EMI per month.')
    st.markdown('**Amount Invested Monthly**') 
    st.info('The average monthly amount invested towards paying off your debts each month.')

st.subheader('Enter appropriate values for each of the following:')

# Genres selection
inputs_list = [
    'Age', 'Monthly_Inhand_Salary', 'Interest_Rate', 'Delay_from_due_date',
    'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Credit_Mix',
    'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age',
    'Total_EMI_per_month', 'Amount_invested_monthly'
]

# Dictionary to store user inputs
user_inputs = {}

# Display input fields for each category
for input in inputs_list:
    if input == 'Credit_Mix':
        user_input = st.selectbox(f'Select value for {input}', ['Good', 'Standard', 'Poor'])
    else:
        user_input = st.text_input(f'Enter value for {input}')
    user_inputs[input] = user_input

# Define a boolean flag to track if prediction has been generated
prediction_generated = False

# Submit button to initiate prediction
if st.button('SUBMIT'):
    # Check if all necessary values are entered
    if any(value == '' for value in user_inputs.values()):
        st.write('Enter all necessary values before pressing SUBMIT')
    else:
        # Convert user inputs to the appropriate data types
        prediction_generated = True
        user_inputs['Age'] = int(user_inputs['Age'])
        user_inputs['Monthly_Inhand_Salary'] = float(user_inputs['Monthly_Inhand_Salary'])
        user_inputs['Interest_Rate'] = int(user_inputs['Interest_Rate'])
        user_inputs['Delay_from_due_date'] = int(user_inputs['Delay_from_due_date'])
        user_inputs['Num_of_Delayed_Payment'] = int(user_inputs['Num_of_Delayed_Payment'])
        user_inputs['Changed_Credit_Limit'] = float(user_inputs['Changed_Credit_Limit'])
        user_inputs['Outstanding_Debt'] = float(user_inputs['Outstanding_Debt'])
        user_inputs['Credit_Utilization_Ratio'] = float(user_inputs['Credit_Utilization_Ratio'])
        user_inputs['Credit_History_Age'] = float(user_inputs['Credit_History_Age'])
        user_inputs['Total_EMI_per_month'] = float(user_inputs['Total_EMI_per_month'])
        user_inputs['Amount_invested_monthly'] = float(user_inputs['Amount_invested_monthly'])

        # Dictionary to map categories to numerical values for the Credit_Mix column
        category_mapping = {'Good': 2, 'Standard': 1, 'Poor': 0}

        # Convert 'Credit_Mix' input to numerical value
        user_inputs['Credit_Mix'] = category_mapping.get(user_inputs['Credit_Mix'])

        # Perform scaling
        scalers = {
            'Age': MinMaxScaler(),
            'Monthly_Inhand_Salary': RobustScaler(),
            'Interest_Rate': MinMaxScaler(),
            'Delay_from_due_date': RobustScaler(),
            'Num_of_Delayed_Payment': RobustScaler(),
            'Changed_Credit_Limit': RobustScaler(),
            'Outstanding_Debt': RobustScaler(),
            'Credit_Utilization_Ratio': MinMaxScaler(),
            'Credit_History_Age': MinMaxScaler(),
            'Total_EMI_per_month': MinMaxScaler(),
            'Amount_invested_monthly': MinMaxScaler(),
        }

        # Fit the scalers
        for key in user_inputs:
            if key in scalers:
                scalers[key].fit([[user_inputs[key]]])

        # Transform the inputs
        for key in user_inputs:
            if key in scalers:
                user_inputs[key] = scalers[key].transform([[user_inputs[key]]])[0][0]

        # Convert data types if necessary
        input_data = pd.DataFrame([user_inputs])

        # Make prediction
        prediction = model.predict(input_data)
        st.session_state.prediction = prediction  # Store prediction in session state

        # Display prediction
        st.subheader('Prediction:')
        st.write(f'The predicted credit score classification is: {prediction}')

# Button for more information about credit score classification
if st.button('Good'):
    st.write('This classification means you have a good credit score, and there is no need to improve any area of your credit report!')
if st.button('Standard'):
    st.write('You score is average, but there are a few changes you can make to improve upon your score. In order to find out what changes will be most beneficial to you, check out the feature importances section.')
if st.button('Poor'):
    st.write('Your score is below average, and there are several improvements you should make to increase your score. In order to find out what changes will be most beneficial to you, check out the feature importances section.')


    # Get feature importances (NOT WORKING)
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]
        top_features = sorted_indices[:3]

        st.subheader('Top 3 Feature Importance:')
        for i, feature_index in enumerate(top_features, start=1):
            st.write(f'{i}. {inputs_list[feature_index]}: {feature_importances[feature_index]}')


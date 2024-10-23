import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle


# Load the trained model
model = tf.keras.models.load_model('my_model.h5')

processor1 = pickle.load(open('pro.pkl','rb'))


# Streamlit app

st.title('Customer Churn Prediction')

# CreditScore	Geography	Gender	Age	Tenure	Balance	NumOfProducts	HasCrCard	IsActiveMember	EstimatedSalary	

CreditScore = st.number_input('Credit Score')
Geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
Gender = st.selectbox('Gender', ['Male', 'Female'])
Age =  st.number_input('Age')
Tenure = st.number_input('Tenure')
Balance = st.number_input('Balance')
NumOfProducts = st.number_input('Number of Products')
HasCrCard = st.selectbox('Has Credit Card', [1,0])
IsActiveMember = st.selectbox('Is Active Member', [1,0])
EstimatedSalary = st.number_input('Estimated Salary')

data = pd.DataFrame({
    'CreditScore': [CreditScore],
    'Geography': [Geography],
    'Gender': [Gender],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
})     

# Transform data

transformed_data = processor1.transform(data)

trans = pd.DataFrame(data=transformed_data, columns=processor1.get_feature_names_out())

# Drop columns from data 
pred1 = data.drop(columns=['CreditScore','Geography','Gender','Age','Balance','EstimatedSalary'],axis=1)

# Concatenate trans and pred1
final_df = pd.concat([trans, pred1],axis=1)

# predict churn
prediction = model.predict(final_df)
prediction_proba = prediction[0][0]

st.write('prediction_proba:-',prediction_proba)

if prediction_proba > 0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer is not likely to churn')

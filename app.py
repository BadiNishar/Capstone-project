import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model
with open('fraud_detection_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load label encoder for 'type'
with open('type_encoder.pkl', 'rb') as file:
    type_encoder = pickle.load(file)

# Load standard scaler
with open('std_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title('Fraud Detection App')

# Input fields for transaction details
step = st.number_input('Step', min_value=1)
type_ = st.selectbox('Type', ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])
amount = st.number_input('Amount', min_value=0.0)
oldbalanceOrg = st.number_input('Old Balance Origin', min_value=0.0)
newbalanceOrig = st.number_input('New Balance Origin', min_value=0.0)
oldbalanceDest = st.number_input('Old Balance Destination', min_value=0.0)
newbalanceDest = st.number_input('New Balance Destination', min_value=0.0)


# Preprocess the input
if st.button('Predict'):
    # 1. Label encode 'type'
    encoded_type = type_encoder.transform([type_])[0]  # Encode and get the scalar value

    # 2. Create a DataFrame for the input features
    input_data = pd.DataFrame({
        'step': [step],
        'type': [encoded_type],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest]
    })

    # 3. Scale the numerical features
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)
    if prediction[0] == 1:
        st.error('Fraudulent Transaction Detected!')
    else:
        st.success('Legitimate Transaction')

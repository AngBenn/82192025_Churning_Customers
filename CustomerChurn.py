import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Function to create the MLP model
def create_mlp_model(input_shape):
    inputs = Input(shape=(input_shape,))
    hidden1 = Dense(64, activation='relu')(inputs)
    hidden2 = Dense(32, activation='relu')(hidden1)
    output = Dense(1, activation='sigmoid')(hidden2)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Load the pre-trained model
model_filename = 'CustomerChurn_model.pkl'
with open(model_filename, 'rb') as file:
    best_model = pickle.load(file)

# Function to preprocess user input
def preprocess_input(data):
    # Create a copy of the DataFrame to avoid modifying the original
    processed_data = data.copy()

    # Handle missing values before converting data types
    processed_data = processed_data.fillna(0)

    # Identify categorical columns
    categorical_columns = ["Contract", "OnlineSecurity", "PaymentMethod", "TechSupport", "OnlineBackup", "Gender"]

    # One-hot encode categorical variables
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_data = pd.DataFrame(encoder.fit_transform(processed_data[categorical_columns]))
    encoded_data.columns = encoder.get_feature_names_out(categorical_columns)

    # Concatenate the encoded data with the original data
    processed_data = pd.concat([processed_data, encoded_data], axis=1)

    

    # Select numerical columns for scaling
    numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit and transform the specified numerical columns
    processed_data[numerical_columns] = scaler.fit_transform(processed_data[numerical_columns])

    return processed_data

# Create a Streamlit app
st.title("Customer Churn Prediction")

# Get user input for prediction
tenure = st.number_input("Enter Tenure:")
monthly_charges = st.number_input("Enter Monthly Charges:")
total_charges = st.number_input("Enter Total Charges:")
gender = st.selectbox("Select Gender:", ["Male", "Female"])
contract = st.selectbox("Select Contract Type:", ["Month-to-month", "One year", "Two year"])
online_security = st.selectbox("Select Online Security:", ["Yes", "No", "No internet service"])
payment_method = st.selectbox("Select Payment Method:", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
tech_support = st.selectbox("Select Tech Support:", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Select Online Backup:", ["Yes", "No", "No internet service"])

# Button to trigger prediction
predict_button = st.button("Predict")

# Display prediction result
if predict_button:
    # Create a DataFrame with user input
    user_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "Contract": [contract],
        "OnlineSecurity": [online_security],
        "PaymentMethod": [payment_method],
        "TechSupport": [tech_support],
        "OnlineBackup": [online_backup],
        "Gender": [gender]
    })
    # One-hot encode the 'contract' column
    contract_encoding = pd.get_dummies(user_data['Contract'], drop_first=True)
    user_data = pd.concat([user_data, contract_encoding], axis=1)

# Drop the original 'Contract' column
    user_data = user_data.drop('Contract', axis=1)
    
    # Preprocess user input
    processed_user_data = preprocess_input(user_data)
    # Debug print to check the processed features
    st.write("Processed User Data:")
    st.write(processed_user_data.columns)
    
    # Make prediction using the pre-trained model
    try:
        y_pred = best_model.predict(processed_user_data)
        st.write(f"Churn Prediction: {y_pred[0]}")
    except Exception as e:
        st.write(f"Error during prediction: {e}")
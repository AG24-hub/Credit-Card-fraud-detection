import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from geopy.distance import geodesic
import lightgbm as lgb
import joblib
import streamlit as st

# Load model and encoder
model = joblib.load("/content/drive/MyDrive/Online_fraud_detection/Fraud_detection_part_2.jb")
encoder = joblib.load("/content/drive/MyDrive/Online_fraud_detection/Label_encoder.jb")

# Function to calculate distance
def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# UI
st.title('Fraud Detection System')
st.write("Enter the transaction details below:")

# Input fields
merchant = st.text_input("Merchant Name")
category = st.text_input("Category")
amt = st.number_input("Transaction amount", min_value=0.0, format="%.2f", key="amt_input")
lat = st.number_input("Latitude", format="%.6f", key="lat_input")
lon = st.number_input("Longitude", format="%.6f", key="lon_input")
merch_lat = st.number_input("Merchant Latitude", format="%.6f", key="merch_lat_input")
merch_lon = st.number_input("Merchant Longitude", format="%.6f", key="merch_lon_input")
hour = st.slider("Transaction hour", 0, 23, 12)
day = st.slider("Transaction day", 1, 31, 15)
month = st.slider("Transaction month", 1, 12, 6)
gender = st.selectbox("Gender", ["Male", "Female"])
cc_num = st.text_input("Credit Card number")

# Calculate distance
distance = calculate_distance(lat, lon, merch_lat, merch_lon)

if st.button("Check for fraud"):
    if merchant and category and cc_num:
        input_data = pd.DataFrame([[merchant, category, amt, cc_num, hour, day, month, gender, distance]],
                                  columns=['merchant', 'category', 'amt', 'cc_num', 'hour', 'day', 'month', 'gender', 'distance'])

        # Encode categorical variables
        for col in ['merchant', 'category', 'gender']:
            try:
                input_data[col] = encoder.transform(input_data[col])
            except ValueError:
                input_data[col] = -1  # unknown label

        # Hash cc number
        input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))

        # Predict
        prediction = model.predict(input_data)[0]
        result = "🚨 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
        st.success(result)
    else:
        st.warning("Please fill all required fields.")




















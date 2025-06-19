import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from geopy.distance import geodesic
import lightgbm as lgb
import joblib
import streamlit as st

# Load model and encoder
try:
    model = joblib.load("Fraud_detection_part_2.jb")
    encoder = joblib.load("Label_encoder.jb")
except Exception as e:
    st.error(f"Failed to load model or encoder: {e}")
    st.stop()

# Function to calculate distance
def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# UI
st.title('💳 Fraud Detection System')
st.write("Enter the transaction details below:")

# Input fields
merchant = st.text_input("Merchant Name")
category = st.text_input("Category")
amt = st.number_input("Transaction amount", min_value=0.0, format="%.2f")
lat = st.number_input("Latitude", format="%.6f")
lon = st.number_input("Longitude", format="%.6f")
merch_lat = st.number_input("Merchant Latitude", format="%.6f")
merch_lon = st.number_input("Merchant Longitude", format="%.6f")
hour = st.slider("Transaction hour", 0, 23, 12)
day = st.slider("Transaction day", 1, 31, 15)
month = st.slider("Transaction month", 1, 12, 6)
gender = st.selectbox("Gender", ["Male", "Female"])
cc_num = st.text_input("Credit Card number")

# Calculate distance
distance = calculate_distance(lat, lon, merch_lat, merch_lon)

# Prediction
if st.button("Check for fraud"):
    if merchant and category and cc_num:
        input_data = pd.DataFrame([[merchant, category, amt, cc_num, hour, day, month, gender, distance]],
                                  columns=['merchant', 'category', 'amt', 'cc_num', 'hour', 'day', 'month', 'gender', 'distance'])
        for col in ['merchant', 'category', 'gender']:
            try:
                input_data[col] = encoder.transform(input_data[col])
            except ValueError:
                input_data[col] = -1

        input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))
        prediction = model.predict(input_data)[0]
        result = "🚨 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
        st.success(result)
    else:
        st.warning("Please fill all required fields.")

# Footer
st.markdown("---")
st.markdown("💡 Developed with ❤️ by **Ankita Ghosh**")

import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("xgb_best_model.pkl")
scaler = joblib.load("scaler.pkl")  # تحميل الـ scaler

st.set_page_config(page_title="✈️ Flight Price Predictor", page_icon="💸")
st.title("✈️ Flight Price Prediction App")
st.markdown("Enter your flight details to estimate the ticket price:")

# Define all fixed options
airlines = ['Air India', 'IndiGo', 'Jet Airways', 'Jet Airways Business', 'Multiple carriers', 'SpiceJet']
sources = ['Delhi', 'Kolkata', 'Mumbai']
destinations = ['Cochin', 'Delhi', 'Hyderabad', 'New Delhi']
flight_length_categories = ['Medium', 'Short']
ticket_types = ['Economy', 'Business']

# User inputs
selected_airline = st.selectbox("Airline", airlines)
source = st.selectbox("Source", sources)
destination = st.selectbox("Destination", destinations)
flight_length = st.selectbox("Flight Length", flight_length_categories)
month = st.selectbox("Month of Journey", list(range(1, 13)))
day = st.slider("Day of Month", 1, 31, 15)
day_number = st.slider("Day of Week (1=Mon, 7=Sun)", 1, 7, 3)
stops = st.selectbox("Total Stops", [0, 1, 2, 3])
degree = st.selectbox("Ticket Type", ['Economy', 'Business'])
duration = st.number_input("Duration (in minutes)", min_value=30, max_value=2000, value=180)
dep_hour = st.slider("Departure Hour", 0, 23, 10)
dep_min = st.slider("Departure Minute", 0, 59, 0)
arr_hour = st.slider("Arrival Hour", 0, 23, 13)
arr_min = st.slider("Arrival Minute", 0, 59, 0)

# Prepare input dict
input_data = {}

# One-hot encode airline
for name in airlines:
    input_data[name] = 1 if name == selected_airline else 0

# One-hot encode source
for s in sources:
    input_data[s] = 1 if s == source else 0

# One-hot encode destination
for d in destinations:
    input_data[f'Dest_{d}'] = 1 if d == destination else 0

# One-hot encode flight length
for f in flight_length_categories:
    input_data[f] = 1 if f == flight_length else 0

# Encode Degree
input_data["Dgree"] = 0 if degree == "Economy" else 1

# Add numerical features
input_data["month"] = month
input_data["day "] = day  # لاحظ المسافة بعد day لأنها في الداتا الأصلية كده
input_data["Total_Stops"] = stops
input_data["Duration(Minutes)"] = duration
input_data["day_number"] = day_number
input_data["Dep_Time_hour"] = dep_hour
input_data["Dep_Time_min"] = dep_min
input_data["Arrival_Time_hour"] = arr_hour
input_data["Arrival_Time_min"] = arr_min

# Final feature order
ordered_features = ['Air India', 'IndiGo', 'Jet Airways', 'Jet Airways Business',
       'Multiple carriers', 'SpiceJet', 'Delhi', 'Kolkata', 'Mumbai',
       'Dest_Cochin', 'Dest_Delhi', 'Dest_Hyderabad', 'Dest_New Delhi',
       'Medium', 'Short', 'month', 'day ', 'Total_Stops', 'Dgree',
       'Duration(Minutes)', 'day_number', 'Dep_Time_hour', 'Dep_Time_min',
       'Arrival_Time_hour', 'Arrival_Time_min']

# Convert to numpy array
features = np.array([input_data[feat] for feat in ordered_features]).reshape(1, -1)

# Apply the scaler to the appropriate columns (e.g., Duration and Total_Stops)
features[0, ordered_features.index("Duration(Minutes)")] = scaler.transform(features[0, ordered_features.index("Duration(Minutes)")].reshape(-1, 1))
features[0, ordered_features.index("Total_Stops")] = scaler.transform(features[0, ordered_features.index("Total_Stops")].reshape(-1, 1))

# Prediction
if st.button("Predict Ticket Price 💰"):
    price = model.predict(features)[0]
    st.success(f"Estimated Ticket Price: **${round(price, 2)}**")

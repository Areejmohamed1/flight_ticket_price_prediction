# ✈️ Flight Ticket Price Prediction

A machine learning model that predicts the price of flight tickets based on airline, source, destination, date, duration, stops, and more.

## 🔍 Problem Statement
Flight ticket prices can be very dynamic and change based on multiple features. This project aims to build a predictive model that estimates the price of a flight ticket using historical flight data.

## 📊 Features Used
- Airline (One-hot encoded)
- Source & Destination (One-hot encoded)
- Flight Duration
- Number of Stops
- Time of Departure / Arrival
- Date (Day, Month, Weekend)
- Encoded Degree (e.g., Economy / Business)
- Flight Length (Short / Medium)
- Flight Category (Morning, Evening, etc.)

## ⚙️ Tech Stack
- Python
- Scikit-Learn
- XGBoost
- Pandas / NumPy
- Streamlit (for UI)

## 🎯 Model Performance
| Metric | Score |
|--------|-------|
| R² Score | 0.902 |
| MAE     | 11.11 |
| RMSE    | 16.71 |

## 🧠 Final Model
- XGBoost Regressor (tuned using RandomizedSearchCV)
- Features: [See `model_features.txt`](model_features.txt)
- Scaler: `MinMaxScaler`

## 🛠 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
2. Predict using script
python
Copy
Edit
from predict import predict_price

# Example input
input = [0, 1, 0, ..., 15]  # ← Replace with 25 feature values
price = predict_price(input)
print("Predicted Price:", price)
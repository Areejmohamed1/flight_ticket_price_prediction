import joblib
import numpy as np

model = joblib.load("xgb_best_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_price(input_features):
    # input_features: list of values in correct order (length 25)
    input_array = np.array(input_features).reshape(1, -1)
    return model.predict(input_array)[0]
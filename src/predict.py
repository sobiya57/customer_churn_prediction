"""
Prediction script for Customer Churn Prediction Project.
This script loads the trained Random Forest pipeline model
and predicts churn for new customer data.
"""

import pandas as pd
import joblib
import json
import os


# --------------------------
# Load Model + Feature Names
# --------------------------

MODEL_PATH = "./models/churn_rf_model.pkl"
FEATURES_PATH = "./models/feature_names.json"


def load_model():
    """Load trained model pipeline."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("❌ Model file not found!")

    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
    return model


def load_feature_names():
    """Load feature names (after preprocessing)."""
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError("❌ feature_names.json not found!")

    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)
    
    return features


# --------------------------
# Prediction Function
# --------------------------

def predict_churn(input_data):
    """
    Predict churn for a new customer.
    
    input_data: dict
        Example:
        {
            "tenure_months": 12,
            "monthly_charge": 29.99,
            "auto_pay": 1,
            "has_discount": 0,
            "num_logins_30d": 20,
            "total_usage_30d_min": 500,
            "avg_session_length_min": 15,
            "days_since_last_login": 2,
            "recent_activity_flag": 1,
            "support_tickets_90d": 0,
            "failed_payments_90d": 0,
            "contract_type": "Monthly",
            "plan_type": "Basic"
        }
    """

    # Convert dict → DataFrame
    df = pd.DataFrame([input_data])

    # Load trained model
    model = load_model()

    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability)
    }


# --------------------------
# Test Example (Optional)
# --------------------------

if __name__ == "__main__":
    sample = {
        "tenure_months": 12,
        "monthly_charge": 29.99,
        "auto_pay": 1,
        "has_discount": 0,
        "num_logins_30d": 20,
        "total_usage_30d_min": 500,
        "avg_session_length_min": 15,
        "days_since_last_login": 2,
        "recent_activity_flag": 1,
        "support_tickets_90d": 0,
        "failed_payments_90d": 0,
        "contract_type": "Monthly",
        "plan_type": "Basic"
    }

    output = predict_churn(sample)
    print("\nPrediction Result:", output)

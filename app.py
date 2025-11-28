import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

# ------------------------------------------------------------
# Load Model & Feature Info
# ------------------------------------------------------------
model = joblib.load("models/churn_pipeline_random_forest.joblib")

with open("models/feature_names.json", "r") as f:
    feature_info = json.load(f)

feature_cols = feature_info["feature_cols"]
numeric_features = feature_info["numeric_features"]
categorical_features = feature_info["categorical_features"]

# ------------------------------------------------------------
# Helper function: Convert date to UNIX timestamp
# ------------------------------------------------------------
def convert_date_to_timestamp(date_val):
    try:
        dt = pd.to_datetime(date_val, errors="coerce")
        return int(dt.value // 10**9)
    except:
        return 0

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.title("📊 Customer Churn Prediction App")
st.write("Fill the details below to predict customer churn.")

# ------------------------------------------------------------
# Input Fields (MATCHES MODEL EXACTLY)
# ------------------------------------------------------------

signup_date = st.date_input("Signup Date")
reference_date = st.date_input("Reference Date")

tenure_months = st.number_input("Tenure (months)", min_value=0, value=0)

contract_type = st.selectbox("Contract Type", ["Monthly", "Quarterly", "Yearly"])
plan_type = st.selectbox("Plan Type", ["Basic", "Standard", "Premium"])

monthly_charge = st.number_input("Monthly Charge (₹)", min_value=0.0, value=0.0)

auto_pay = st.selectbox("Auto Pay Enabled?", ["Yes", "No"])
has_discount = st.selectbox("Has Discount?", ["Yes", "No"])

num_logins_30d = st.number_input("Logins (last 30 days)", min_value=0, value=0)
total_usage_30d_min = st.number_input("Usage (minutes, last 30 days)", min_value=0, value=0)
avg_session_length_min = st.number_input("Avg Session Duration (min)", min_value=0.0, value=0.0)

days_since_last_login = st.number_input("Days Since Last Login", min_value=0, value=0)

recent_activity_flag = st.selectbox("Recent Activity Level", ["Low", "Medium", "High"])

support_tickets_90d = st.number_input("Support Tickets (last 90 days)", min_value=0, value=0)
failed_payments_90d = st.number_input("Failed Payments (last 90 days)", min_value=0, value=0)

# ------------------------------------------------------------
# Predict Button
# ------------------------------------------------------------
if st.button("Predict Churn"):

    # Create input DataFrame (NO customer_id here)
    input_data = pd.DataFrame([{
        "signup_date": convert_date_to_timestamp(signup_date),
        "reference_date": convert_date_to_timestamp(reference_date),
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "plan_type": plan_type,
        "monthly_charge": monthly_charge,
        "auto_pay": auto_pay,
        "has_discount": has_discount,
        "num_logins_30d": num_logins_30d,
        "total_usage_30d_min": total_usage_30d_min,
        "avg_session_length_min": avg_session_length_min,
        "days_since_last_login": days_since_last_login,
        "recent_activity_flag": recent_activity_flag,
        "support_tickets_90d": support_tickets_90d,
        "failed_payments_90d": failed_payments_90d
    }])
    
    # ------------------------------------------------------------
    # FIX: Ensure numeric columns are numeric
    # ------------------------------------------------------------
    for col in numeric_features:
        input_data[col] = pd.to_numeric(input_data[col], errors="coerce").fillna(0)

    # Categorical columns must remain strings
    for col in categorical_features:
        input_data[col] = input_data[col].astype(str)

    # Ensure correct column order
    input_data = input_data[feature_cols]

    # ------------------------------------------------------------
    # Make Prediction
    # ------------------------------------------------------------
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------
    st.subheader("🔍 Prediction Result")

    if prediction == 1:
        st.error(f"⚠ Customer is LIKELY to churn ({probability*100:.2f}% probability)")
    else:
        st.success(f"✓ Customer is NOT likely to churn ({probability*100:.2f}% probability)")






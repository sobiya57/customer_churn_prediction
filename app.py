import streamlit as st
import pandas as pd
import joblib
import json

# ------------------------------------------------
# MUST BE THE FIRST STREAMLIT COMMAND
# ------------------------------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# ------------------------------------------------
# Load Model
# ------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("./models/churn_rf_model.pkl")
    return model

model = load_model()

# ------------------------------------------------
# Streamlit UI
# ------------------------------------------------
st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details below to predict churn probability.")

# ------------------------------------------------
# Input Form
# ------------------------------------------------
st.subheader("🔧 Customer Information")

col1, col2 = st.columns(2)

with col1:
    tenure_months = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
    monthly_charge = st.number_input("Monthly Charge ($)", min_value=0.0, max_value=200.0, value=29.99)
    auto_pay = st.selectbox("Auto Pay Enabled?", [0, 1])
    has_discount = st.selectbox("Has Discount?", [0, 1])
    num_logins_30d = st.number_input("Number of Logins (30 days)", min_value=0, max_value=200, value=20)
    total_usage_30d_min = st.number_input("Total Usage (Minutes, 30 days)", min_value=0, max_value=5000, value=500)

with col2:
    avg_session_length_min = st.number_input("Avg Session Length (min)", min_value=0, max_value=60, value=15)
    days_since_last_login = st.number_input("Days Since Last Login", min_value=0, max_value=60, value=2)
    recent_activity_flag = st.selectbox("Recent Activity Flag", [0, 1])
    support_tickets_90d = st.number_input("Support Tickets (90 days)", min_value=0, max_value=50, value=0)
    failed_payments_90d = st.number_input("Failed Payments (90 days)", min_value=0, max_value=20, value=0)

contract_type = st.selectbox("Contract Type", ["Monthly", "Quarterly", "Annual"])
plan_type = st.selectbox("Plan Type", ["Basic", "Standard", "Premium"])

# ------------------------------------------------
# Predict Button
# ------------------------------------------------
if st.button("🔮 Predict Churn"):

    input_data = {
        "tenure_months": tenure_months,
        "monthly_charge": monthly_charge,
        "auto_pay": auto_pay,
        "has_discount": has_discount,
        "num_logins_30d": num_logins_30d,
        "total_usage_30d_min": total_usage_30d_min,
        "avg_session_length_min": avg_session_length_min,
        "days_since_last_login": days_since_last_login,
        "recent_activity_flag": recent_activity_flag,
        "support_tickets_90d": support_tickets_90d,
        "failed_payments_90d": failed_payments_90d,
        "contract_type": contract_type,
        "plan_type": plan_type
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    # Output
    st.subheader("📈 Prediction Result")
    st.write(f"**Churn Probability:** `{round(probability, 3)}`")

    if prediction == 1:
        st.error("⚠️ The customer is **LIKELY to churn**.")
    else:
        st.success("✅ The customer is **NOT likely to churn**.")

st.write("----")
st.write("Built with ❤️ using Streamlit & Machine Learning")






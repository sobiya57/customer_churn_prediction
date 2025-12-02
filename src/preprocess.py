"""
Preprocessing module for Customer Churn Prediction Project.
This file builds and returns the preprocessing pipeline used
for model training and prediction.
"""

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def get_preprocessor():
    """
    Creates and returns the preprocessing pipeline:
    - Scales numeric features using StandardScaler
    - Encodes categorical features using OneHotEncoder
    
    Returns:
        ColumnTransformer: preprocessing pipeline
    """

    # Categorical features (strings)
    categorical_features = ['contract_type', 'plan_type']

    # Numeric features
    numeric_features = [
        'tenure_months',
        'monthly_charge',
        'auto_pay',
        'has_discount',
        'num_logins_30d',
        'total_usage_30d_min',
        'avg_session_length_min',
        'days_since_last_login',
        'recent_activity_flag',
        'support_tickets_90d',
        'failed_payments_90d'
    ]

    # Define transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor

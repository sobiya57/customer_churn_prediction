"""
Training script for Customer Churn Prediction Project.
This script:
- Loads cleaned dataset
- Preprocesses data using pipeline from preprocess.py
- Trains Logistic Regression & Random Forest models
- Evaluates performance
- Saves trained models to the /models directory
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

# Import preprocessing pipeline
from preprocess import get_preprocessor


def load_data():
    """Load churn dataset from the data directory."""
    df = pd.read_csv("./data/churn_dataset_10000.csv")
    print("Dataset loaded successfully!")
    return df


def clean_data(df):
    """Drop unnecessary/leakage columns and separate features and target."""
    drop_cols = ['customer_id', 'signup_date', 'reference_date',
                 'churn_prob', 'churn_reason']
    
    df = df.drop(columns=drop_cols)
    
    X = df.drop(columns=['churn'])
    y = df['churn']
    
    return X, y


def build_logistic_model(preprocessor):
    """Create Logistic Regression pipeline."""
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])


def build_random_forest_model(preprocessor):
    """Create Random Forest pipeline."""
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])


def evaluate_model(model, X_test, y_test, model_name):
    """Print classification report + AUC score."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n==========================")
    print(f"📌 {model_name} Evaluation Results")
    print(f"==========================\n")
    print(classification_report(y_test, y_pred))
    print("AUC Score:", roc_auc_score(y_test, y_proba))


def save_model(model, filename):
    """Save model inside the /models folder."""
    os.makedirs("models", exist_ok=True)
    filepath = f"./models/{filename}"
    joblib.dump(model, filepath)
    print(f"Model saved: {filepath}")


def main():
    # Load data
    df = load_data()

    # Clean & prepare
    X, y = clean_data(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Load preprocessing pipeline
    preprocessor = get_preprocessor()

    # 🟦 Train Logistic Regression
    log_model = build_logistic_model(preprocessor)
    log_model.fit(X_train, y_train)
    evaluate_model(log_model, X_test, y_test, "Logistic Regression")
    save_model(log_model, "churn_log_reg_model.pkl")

    # 🟩 Train Random Forest
    rf_model = build_random_forest_model(preprocessor)
    rf_model.fit(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, "Random Forest")
    save_model(rf_model, "churn_rf_model.pkl")

    print("\n🎉 Training completed successfully!")


# Run main
if __name__ == "__main__":
    main()

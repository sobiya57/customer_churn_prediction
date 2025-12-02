📌 README.md — Professional Version
📊 Customer Churn Prediction (End-to-End Machine Learning Project)

This project predicts customer churn (whether a customer is likely to leave a service) using machine learning.
It includes EDA, model training, preprocessing pipelines, prediction scripts, and a full Streamlit web app for interactive churn prediction.

🚀 Project Overview

Customer churn is a key metric for subscription-based businesses.
This project provides an end-to-end ML solution:

📥 Import & preprocess customer data

🔍 Perform detailed Exploratory Data Analysis (EDA)

🤖 Train ML models (Logistic Regression & Random Forest)

🧠 Save trained pipelines

📝 Generate feature names for deployment

🌐 Deploy a Streamlit app for real-time prediction

📦 Production-grade src/ code for model training & inference

📁 Project Structure
customer_churn_project/
│
├── data/
│   └── churn_dataset_10000.csv
│
├── models/
│   ├── churn_rf_model.pkl
│   ├── churn_log_reg_model.pkl
│   └── feature_names.json
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── screenshots/
│   ├── app_home.png
│   ├── input_form.png
│   └── prediction_output.png
│
├── 01_EDA.ipynb
├── 02_Model_Training.ipynb
├── app.py
├── requirements.txt
└── README.md

🧪 Dataset Description

The dataset contains 10,000 customer records with:

Customer Behavior

Tenure

Monthly charges

Usage (minutes, sessions)

Login activity

Failed payments

Support history

Customer Metadata

Plan type

Contract type

Discounts

Auto-pay

Target Variable

churn (0 = not churned, 1 = churned)

📊 Exploratory Data Analysis (EDA)

The EDA notebook (01_EDA.ipynb) includes:

Missing value analysis

Target distribution

Categorical distribution

Numerical histograms

Churn vs numerical variables

Churn vs categorical variables

Correlation heatmap

Business insights

🔍 Key Insights:

Customers with low tenure churn more.

Customers with failed payments have higher churn risk.

Monthly contract users churn more than annual plan users.

Customers with auto-pay enabled churn significantly less.

🤖 Model Training

The training notebook (02_Model_Training.ipynb) performs:

✔ Data cleaning
✔ Train-test split
✔ Preprocessing pipeline using ColumnTransformer

StandardScaler (numeric features)

OneHotEncoder (categorical features)

✔ Models used:

Logistic Regression

Random Forest

✔ Evaluation metrics:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Both models are trained and saved in /models/.

🧠 Production Code (src/)
📌 preprocess.py

Contains preprocessing pipeline used for both training and prediction.

📌 train.py

Trains both models:

Saves trained pipelines

Generates feature_names.json

Evaluates models

📌 predict.py

Loads saved model and predicts churn for new customers.

🌐 Streamlit Web App

The app (app.py) provides a clean interface to input customer details and get predictions:

Features:

Numeric & categorical inputs

Displays churn probability

Highlights churn risk

Easy to use on desktop/mobile

Run locally:
streamlit run app.py

📦 Installation
pip install -r requirements.txt

🧪 Train Model
python src/train.py

🔮 Run Prediction Script
python src/predict.py

📸 Application Screenshots

🏠 Home Page

📝 Input Form

📈 Prediction Output

	

🛠️ Tech Stack

Python

Pandas

NumPy

Scikit-Learn

Matplotlib / Seaborn

Streamlit

Joblib

👩‍💻 Author

Sobiya Begum
Machine Learning & Data Science Enthusiast
📧 Email: sobiyabegumbegum@gmail.com
🔗 LinkedIn: www.linkedin.com/in/

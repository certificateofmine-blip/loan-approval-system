import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Page Config
st.set_page_config(
    page_title="Loan Intelligence Engine", page_icon="💳", layout="centered"
)

# 2. Hardcoded clean dictionary to bypass CSV parsing issues entirely
@st.cache_resource
def get_trained_pipeline():
    # Using a clean dictionary format eliminates any raw string/line-break bugs
    mock_data = {
        "Gender": ["Male", "Male", "Male", "Female", "Male", "Female"],
        "Married": ["No", "Yes", "Yes", "No", "Yes", "Yes"],
        "Dependents": [0.0, 1.0, 0.0, 0.0, 2.0, 3.0],
        "Education": [
            "Graduate",
            "Graduate",
            "Not Graduate",
            "Graduate",
            "Graduate",
            "Graduate",
        ],
        "Self_Employed": ["No", "No", "Yes", "No", "No", "Yes"],
        "Loan_Amount_Term": [360.0, 360.0, 360.0, 360.0, 360.0, 360.0],
        "Credit_History": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        "Property_Area": [
            "Urban",
            "Rural",
            "Urban",
            "Urban",
            "Urban",
            "Semiurban",
        ],
        "Total_Income": [5849.0, 6091.0, 3000.0, 3510.0, 9613.0, 12000.0],
        "Log_Total_Income": [8.67, 8.71, 8.01, 8.16, 9.17, 9.39],
        "Log_LoanAmount": [4.94, 4.86, 4.20, 4.34, 5.59, 6.21],
        "Loan_Status": [1, 0, 1, 0, 1, 1],
    }

    df = pd.DataFrame(mock_data)
    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]

    num_features = [
        "Dependents",
        "Loan_Amount_Term",
        "Total_Income",
        "Log_Total_Income",
        "Log_LoanAmount",
    ]
    cat_features = [
        "Gender",
        "Married",
        "Education",
        "Self_Employed",
        "Credit_History",
        "Property_Area",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_features,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(n_estimators=10, random_state=42),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


pipeline = get_trained_pipeline()

# 3. Form UI
st.title("💳 Loan Underwriting Engine")
with st.form("loan_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", [0, 1, 2, "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    with col2:
        applicant_inc = st.number_input(
            "Applicant Income ($)", min_value=0, value=5000
        )
        coapplicant_inc = st.number_input(
            "Coapplicant Income ($)", min_value=0, value=0
        )
        loan_amt = st.number_input("Loan Amount (Thousands)", min_value=1, value=120)
        loan_term = st.number_input("Term (Months)", min_value=12, value=360)
        credit_history = st.selectbox("Credit History", [1.0, 0.0])

    property_area = st.radio("Property Area", ["Urban", "Semiurban", "Rural"])
    submit = st.form_submit_button("Run ML Prediction")

# 4. Run Inference
if submit:
    parsed_deps = 3.0 if dependents == "3+" else float(dependents)
    total_inc = float(applicant_inc + coapplicant_inc)

    input_df = pd.DataFrame(
        [
            {
                "Gender": gender,
                "Married": married,
                "Dependents": parsed_deps,
                "Education": education,
                "Self_Employed": self_employed,
                "Loan_Amount_Term": float(loan_term),
                "Credit_History": credit_history,
                "Property_Area": property_area,
                "Total_Income": total_inc,
                "Log_Total_Income": np.log1p(total_inc),
                "Log_LoanAmount": np.log1p(float(loan_amt)),
            }
        ]
    )

    pred = pipeline.predict(input_df)[0]
    status = "🟢 APPROVED" if pred == 1 else "🔴 REJECTED"
    st.metric(label="Model Decision", value=status)
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer  # Explicitly needed for MICE
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score

def build_and_train_pipeline(data_path="dataset.csv", model_path="loan_model.pkl"):
    # 1. Load Data
    df = pd.read_csv(data_path)
    
    # Drop structural identifier
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])
        
    # Split Features and Target
    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"].map({"Y": 1, "N": 0}) # Encode target strictly to binary integers

    # 2. Identify Column Types dynamically
    numeric_features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    categorical_features = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Credit_History", "Property_Area"]

    # 3. Create Advanced Transformers
    # MICE (IterativeImputer) handles numerical dependencies natively
    numeric_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(max_iter=10, random_state=42)),
        ('scaler', StandardScaler())
    ])

    # Frequent category mapping + OneHotEncoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine into a single preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 4. Define the Machine Learning Architecture
    # XGBoost handles structural tabular imbalances excellently
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=2.0,  # Balances precision/recall for minority class (No)
        random_state=42,
        eval_metric="logloss"
    )

    # Bundle preprocessing and model execution flawlessly into a single Pipeline
    clf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # 5. Train-Test Validation Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("⚡ Training model with optimized hyper-parameters...")
    clf_pipeline.fit(X_train, y_train)

    # 6. Evaluation metrics
    y_pred = clf_pipeline.predict(X_test)
    y_proba = clf_pipeline.predict_proba(X_test)[:, 1]
    
    print("\n📊 --- Evaluation Report ---")
    print(classification_report(y_test, y_pred, target_names=["Denied (N)", "Approved (Y)"]))
    print(f"ROC-AUC Performance Score: {roc_auc_score(y_test, y_proba):.4f}\n")

    # 7. Serialize and Save Pipeline 
    joblib.dump(clf_pipeline, model_path)
    print(f"📦 Model package successfully exported to: {model_path}")

if __name__ == "__main__":
    build_and_train_pipeline()
# D:\Automated Loan Approval System\utils.py
import pandas as pd
import numpy as np
import logging
import os
import joblib
from sklearn.metrics import accuracy_score, classification_report

def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger

def validate_data(df, expected_columns, numeric_cols=None, binary_cols=None, categorical_cols=None):
    logger = logging.getLogger(__name__)
    logger.info("Validating data...")
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
    extra_cols = [col for col in df.columns if col not in expected_columns]
    if extra_cols:
        logger.info(f"Extra columns found: {extra_cols}")
    missing_values = df.isnull().sum()
    if missing_values.any():
        logger.warning(f"Missing values found:\n{missing_values[missing_values > 0]}")
    
    numeric_cols = numeric_cols or []
    binary_cols = binary_cols or []
    categorical_cols = categorical_cols or []
    
    for col in numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in binary_cols:
        if col in df.columns and not df[col].isin([0, 1, np.nan]).all():
            logger.warning(f"Non-binary values in {col}: {df[col].unique()}")
    for col in categorical_cols:
        if col in df.columns and not pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype(str)
    logger.info("Data validation completed")
    return df

def impute_missing_values(df, numeric_cols=None, binary_cols=None, categorical_cols=None):
    logger = logging.getLogger(__name__)
    logger.info("Imputing missing values...")
    numeric_cols = numeric_cols or []
    binary_cols = binary_cols or []
    categorical_cols = categorical_cols or []
    
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"Imputed {col} with median: {median_val}")
    for col in binary_cols + categorical_cols:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            logger.info(f"Imputed {col} with mode: {mode_val}")
    return df

def handle_outliers(df, numeric_cols):
    logger = logging.getLogger(__name__)
    logger.info("Handling outliers...")
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            logger.info(f"Capped outliers in {col}")
    return df

def load_artifact(file_path):
    logger = logging.getLogger(__name__)
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} not found")
        raise FileNotFoundError(f"File {file_path} not found")
    logger.info(f"Loading artifact from {file_path}")
    return joblib.load(file_path)

def save_artifact(obj, file_path):
    logger = logging.getLogger(__name__)
    logger.info(f"Saving artifact to {file_path}")
    joblib.dump(obj, file_path)
    logger.info("Artifact saved successfully")

def evaluate_model(model, X_test, y_test, model_name):
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating model: {model_name}")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] * 100
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel: {model_name}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nSample Probabilities (first 5):")
    for i, prob in enumerate(y_prob[:5]):
        print(f"Sample {i+1}: {prob:.2f}%")
    return accuracy
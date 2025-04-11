# D:\Automated Loan Approval System\prediction_pipeline.py
from utils import setup_logging, validate_data, load_artifact
import pandas as pd

class PredictionPipeline:
    def __init__(self):
        self.model = None
        self.transformer = None
        self.expected_columns = None
        self.logger = setup_logging()

    def load_artifacts(self):
        self.model = load_artifact('artifacts/best_loan_model.pkl')
        self.transformer = load_artifact('artifacts/transformer.pkl')
        self.expected_columns = self.transformer.feature_names_in_

    def validate_input(self, features):
        features_df = pd.DataFrame([features])
        numeric_cols = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'revol.util', 'inq.last.6mths']
        binary_cols = ['not.fully.paid']
        categorical_cols = ['purpose'] if 'purpose' in self.expected_columns else []
        features_df = validate_data(features_df, self.expected_columns, numeric_cols, binary_cols, categorical_cols)
        return features_df[self.expected_columns]

    def predict(self, features):
        features_df = self.validate_input(features)
        features_transformed = self.transformer.transform(features_df)
        probability = self.model.predict_proba(features_transformed)[0]
        return probability[1] * 100

    def run_pipeline(self, features):
        self.load_artifacts()
        return self.predict(features)
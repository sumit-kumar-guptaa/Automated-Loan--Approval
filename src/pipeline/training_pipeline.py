# D:\Automated Loan Approval System\training_pipeline.py
from utils import (setup_logging, validate_data, impute_missing_values, handle_outliers, 
                  save_artifact, evaluate_model)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd

class TrainingPipeline:
    def __init__(self, input_path):
        self.input_path = input_path
        self.target_col = 'credit.policy'
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.transformer = None
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.logger = setup_logging()
        self.numeric_cols = ['int.rate', 'installment', 'log.annual.inc', 'dti', 
                           'fico', 'revol.util', 'inq.last.6mths']
        self.binary_cols = ['not.fully.paid']
        self.categorical_cols = ['purpose'] if 'purpose' in pd.read_csv(input_path).columns else []
        self.expected_columns = [self.target_col] + self.numeric_cols + self.binary_cols + self.categorical_cols

    def ingest_data(self):
        self.df = pd.read_csv(self.input_path)
        self.df = validate_data(self.df, self.expected_columns, self.numeric_cols, self.binary_cols, self.categorical_cols)
        self.df = impute_missing_values(self.df, self.numeric_cols, self.binary_cols + [self.target_col], self.categorical_cols)
        return self.df

    def transform_data(self):
        self.y = self.df[self.target_col]
        self.X = self.df.drop(columns=[self.target_col])
        self.X = handle_outliers(self.X, self.numeric_cols)
        
        transformers = [
            ('num', StandardScaler(), self.numeric_cols),
            ('bin', 'passthrough', self.binary_cols)
        ]
        if self.categorical_cols:
            transformers.append(('cat', LabelEncoder(), self.categorical_cols))
        
        self.transformer = ColumnTransformer(transformers=transformers, remainder='drop')
        self.X = self.transformer.fit_transform(self.X)
        self.X = pd.DataFrame(self.X, columns=self.numeric_cols + self.binary_cols + self.categorical_cols, index=self.df.index)
        return self.X, self.y

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_models(self):
        models = {
            'Logistic Regression': {
                'model': LogisticRegression(),
                'params': {'C': [0.01, 0.1, 1], 'max_iter': [1000]}
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=42),
                'params': {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
            }
        }
        
        for name, config in models.items():
            grid_search = GridSearchCV(config['model'], config['params'], cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            test_score = evaluate_model(grid_search.best_estimator_, self.X_test, self.y_test, name)
            if test_score > self.best_score:
                self.best_score = test_score
                self.best_model = grid_search.best_estimator_
                self.best_model_name = name
        return self.best_model

    def save_artifacts(self):
        save_artifact(self.best_model, 'artifacts/best_loan_model.pkl')
        save_artifact(self.transformer, 'artifacts/transformer.pkl')

    def run_pipeline(self):
        self.ingest_data()
        self.transform_data()
        self.split_data()
        self.train_models()
        self.save_artifacts()
        return self.best_model, self.best_model_name
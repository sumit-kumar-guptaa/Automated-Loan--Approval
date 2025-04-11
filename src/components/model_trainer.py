import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, input_path='transformed_data.csv', target_col='credit.policy'):
        """
        Initialize the model trainer.
        
        Args:
            input_path (str): Path to the transformed data file
            target_col (str): Name of the target column
        """
        self.input_path = input_path
        self.target_col = target_col
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0

    def load_data(self):
        """
        Load the transformed data and split into features and target.
        """
        try:
            if not os.path.exists(self.input_path):
                raise FileNotFoundError(f"File {self.input_path} not found")
            
            logger.info(f"Loading data from {self.input_path}")
            df = pd.read_csv(self.input_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            
            self.y = df[self.target_col]
            self.X = df.drop(columns=[self.target_col])
            return self.X, self.y
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        """
        try:
            logger.info("Splitting data into train and test sets...")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state
            )
            logger.info(f"Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
            return self.X_train, self.X_test, self.y_train, self.y_test
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise

    def train_models(self):
        """
        Train multiple models with hyperparameter tuning.
        """
        try:
            logger.info("Training models...")
            
            # Define models and parameter grids
            self.models = {
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

            for name, config in self.models.items():
                logger.info(f"Training {name}...")
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1
                )
                grid_search.fit(self.X_train, self.y_train)
                self.models[name]['grid_search'] = grid_search
                logger.info(f"{name} - Best params: {grid_search.best_params_}, "
                          f"Best CV score: {grid_search.best_score_:.4f}")
                
                # Evaluate on test set
                y_pred = grid_search.predict(self.X_test)
                test_score = accuracy_score(self.y_test, y_pred)
                logger.info(f"{name} - Test accuracy: {test_score:.4f}")
                
                # Update best model
                if test_score > self.best_score:
                    self.best_score = test_score
                    self.best_model = grid_search.best_estimator_
                    self.best_model_name = name
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise

    def evaluate_best_model(self):
        """
        Evaluate the best model with detailed metrics.
        """
        try:
            if self.best_model is None:
                raise ValueError("No model trained yet.")
            
            logger.info(f"Evaluating best model: {self.best_model_name}")
            y_pred = self.best_model.predict(self.X_test)
            y_prob = self.best_model.predict_proba(self.X_test)[:, 1] * 100  # Percentage probabilities
            
            # Print metrics
            print(f"\nBest Model: {self.best_model_name}")
            print(f"Test Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
            print("\nSample Probabilities (first 5):")
            for i, prob in enumerate(y_prob[:5]):
                print(f"Sample {i+1}: {prob:.2f}%")
            
            return self.best_model
        except Exception as e:
            logger.error(f"Error evaluating best model: {str(e)}")
            raise

    def save_model(self, output_path='best_loan_model.pkl'):
        """
        Save the best model.
        """
        try:
            if self.best_model is None:
                raise ValueError("No model trained yet.")
            
            logger.info(f"Saving best model to {output_path}")
            joblib.dump(self.best_model, output_path)
            logger.info("Best model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def run_pipeline(self, model_path='best_loan_model.pkl'):
        """
        Run the full model training pipeline.
        """
        try:
            self.load_data()
            self.split_data()
            self.train_models()
            self.evaluate_best_model()
            self.save_model(model_path)
            logger.info("Model training pipeline completed successfully")
            return self.best_model, self.best_model_name
        except Exception as e:
            logger.error(f"Model training pipeline failed: {str(e)}")
            raise

# Example Usage
if __name__ == "__main__":
    # Specify input file from transformation step
    input_path = 'transformed_data.csv'  # Replace with your actual file path
    
    # Initialize and run the pipeline
    trainer = ModelTrainer(input_path)
    best_model, best_model_name = trainer.run_pipeline()
    
    # Print final summary
    print(f"\nTraining completed. Best model: {best_model_name}")
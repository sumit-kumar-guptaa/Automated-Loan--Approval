import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
import os
try:
    import joblib
except ImportError:
    raise ImportError("joblib is required. Install it using 'pip install joblib'.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self, file_path, expected_columns=None):
        """
        Initialize the data ingestion pipeline.
        
        Args:
            file_path (str): Path to the input data file (e.g., CSV)
            expected_columns (list): List of expected column names
        """
        self.file_path = file_path
        self.expected_columns = expected_columns or [
            'credit.policy', 'int.rate', 'installment', 'log.annual.inc', 
            'dti', 'fico', 'revol.util', 'inq.last.6mths', 'not.fully.paid'
        ]
        self.df = None
        self.label_encoder = None

    def load_data(self):
        """
        Load the data from the specified file.
        """
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File {self.file_path} not found")
            
            logger.info(f"Loading data from {self.file_path}")
            self.df = pd.read_csv(self.file_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def validate_data(self):
        """
        Validate the data for expected columns, data types, and integrity.
        """
        try:
            logger.info("Validating data...")
            
            # Check for missing columns
            actual_cols = set(self.df.columns)
            expected_cols = set(self.expected_columns)
            missing_cols = expected_cols - actual_cols
            extra_cols = actual_cols - expected_cols
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
            if extra_cols:
                logger.info(f"Extra columns found: {extra_cols}")
                # Optionally drop extra columns
                self.df = self.df[list(expected_cols & actual_cols) + ['purpose'] if 'purpose' in self.df.columns else []]
            
            # Check data types
            numeric_cols = ['int.rate', 'installment', 'log.annual.inc', 'dti', 
                          'fico', 'revol.util', 'inq.last.6mths']
            binary_cols = ['credit.policy', 'not.fully.paid']
            categorical_cols = ['purpose'] if 'purpose' in self.df.columns else []

            for col in numeric_cols:
                if col in self.df.columns and not pd.api.types.is_numeric_dtype(self.df[col]):
                    logger.warning(f"Column {col} should be numeric but found {self.df[col].dtype}")
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

            for col in binary_cols:
                if col in self.df.columns and not self.df[col].isin([0, 1, np.nan]).all():
                    logger.warning(f"Column {col} has non-binary values: {self.df[col].unique()}")

            # Check missing values
            missing_values = self.df.isnull().sum()
            if missing_values.any():
                logger.warning(f"Missing values found:\n{missing_values[missing_values > 0]}")
            
            logger.info("Data validation completed")
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            raise

    def preprocess_data(self):
        """
        Preprocess the data: handle missing values and encode categorical variables.
        """
        try:
            logger.info("Preprocessing data...")
            
            # Handle missing values
            numeric_cols = ['int.rate', 'installment', 'log.annual.inc', 'dti', 
                          'fico', 'revol.util', 'inq.last.6mths']
            binary_cols = ['credit.policy', 'not.fully.paid']
            categorical_cols = ['purpose'] if 'purpose' in self.df.columns else []

            for col in numeric_cols:
                if col in self.df.columns and self.df[col].isnull().any():
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    logger.info(f"Imputed {col} with median: {median_val}")

            for col in binary_cols:
                if col in self.df.columns and self.df[col].isnull().any():
                    mode_val = self.df[col].mode()[0]
                    self.df[col].fillna(mode_val, inplace=True)
                    logger.info(f"Imputed {col} with mode: {mode_val}")

            # Encode categorical variables
            if categorical_cols:
                self.label_encoder = LabelEncoder()
                for col in categorical_cols:
                    self.df[col] = self.label_encoder.fit_transform(self.df[col].astype(str))
                    logger.info(f"Encoded categorical column: {col}")
                joblib.dump(self.label_encoder, 'label_encoder.pkl')
                logger.info("Label encoder saved as label_encoder.pkl")
            else:
                logger.info("No categorical columns to encode")

            logger.info("Preprocessing completed")
            return self.df
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise

    def save_cleaned_data(self, output_path='cleaned_data.csv'):
        """
        Save the cleaned and preprocessed data.
        """
        try:
            logger.info(f"Saving cleaned data to {output_path}")
            self.df.to_csv(output_path, index=False)
            logger.info("Cleaned data saved successfully")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

    def run_pipeline(self, output_path='cleaned_data.csv'):
        """
        Run the full data ingestion pipeline.
        """
        try:
            self.load_data()
            self.validate_data()
            self.preprocess_data()
            self.save_cleaned_data(output_path)
            logger.info("Data ingestion pipeline completed successfully")
            return self.df
        except Exception as e:
            logger.error(f"Data ingestion pipeline failed: {str(e)}")
            raise

# Example Usage
if __name__ == "__main__":
    file_path = 'your_data.csv'  # Replace with your actual file path
    ingestion = DataIngestion(file_path)
    cleaned_df = ingestion.run_pipeline()
    
    print("\nCleaned Data Summary:")
    print(cleaned_df.info())
    print("\nFirst few rows:")
    print(cleaned_df.head())
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataTransformation:
    def __init__(self, input_path='cleaned_data.csv', target_col='credit.policy'):
        """
        Initialize the data transformation pipeline.
        
        Args:
            input_path (str): Path to the input data file (e.g., cleaned_data.csv)
            target_col (str): Name of the target column
        """
        self.input_path = input_path
        self.target_col = target_col
        self.df = None
        self.X = None
        self.y = None
        self.transformer = None
        self.categorical_cols = ['purpose']  # Adjust based on your data
        self.numeric_cols = ['int.rate', 'installment', 'log.annual.inc', 'dti', 
                           'fico', 'revol.util', 'inq.last.6mths']
        self.binary_cols = ['not.fully.paid']  # Excluded from scaling, included in output

    def load_data(self):
        """
        Load the cleaned data from the ingestion step.
        """
        try:
            if not os.path.exists(self.input_path):
                raise FileNotFoundError(f"File {self.input_path} not found")
            
            logger.info(f"Loading data from {self.input_path}")
            self.df = pd.read_csv(self.input_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            
            # Separate features and target
            self.y = self.df[self.target_col]
            self.X = self.df.drop(columns=[self.target_col])
            return self.X, self.y
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def handle_outliers(self):
        """
        Handle outliers in numeric columns using IQR method.
        """
        try:
            logger.info("Handling outliers...")
            for col in self.numeric_cols:
                if col in self.X.columns:
                    Q1 = self.X[col].quantile(0.25)
                    Q3 = self.X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Cap outliers
                    self.X[col] = self.X[col].clip(lower=lower_bound, upper=upper_bound)
                    logger.info(f"Capped outliers in {col}")
            return self.X
        except Exception as e:
            logger.error(f"Error handling outliers: {str(e)}")
            raise

    def create_transformer(self):
        """
        Create a ColumnTransformer for categorical and numeric features.
        """
        try:
            logger.info("Creating transformer pipeline...")
            
            # Define transformers
            transformers = []
            
            # Categorical columns: LabelEncoder (already applied in ingestion, but included for new data)
            if self.categorical_cols and any(col in self.X.columns for col in self.categorical_cols):
                transformers.append(('cat', Pipeline([
                    ('label_encoder', LabelEncoder())
                ]), self.categorical_cols))
            
            # Numeric columns: StandardScaler
            if self.numeric_cols:
                transformers.append(('num', StandardScaler(), self.numeric_cols))
            
            # Binary columns: Pass through (no transformation needed)
            if self.binary_cols:
                transformers.append(('bin', 'passthrough', self.binary_cols))
            
            # Create ColumnTransformer
            self.transformer = ColumnTransformer(transformers=transformers, remainder='drop')
            logger.info("Transformer created successfully")
            return self.transformer
        except Exception as e:
            logger.error(f"Error creating transformer: {str(e)}")
            raise

    def fit_transform(self):
        """
        Fit the transformer on the data and transform it.
        """
        try:
            logger.info("Fitting and transforming data...")
            
            # Handle outliers
            self.handle_outliers()
            
            # Create and fit transformer
            self.create_transformer()
            X_transformed = self.transformer.fit_transform(self.X)
            
            # Convert back to DataFrame with appropriate column names
            transformed_cols = []
            if self.categorical_cols and any(col in self.X.columns for col in self.categorical_cols):
                transformed_cols.extend(self.categorical_cols)
            if self.numeric_cols:
                transformed_cols.extend(self.numeric_cols)
            if self.binary_cols:
                transformed_cols.extend(self.binary_cols)
            
            self.X = pd.DataFrame(X_transformed, columns=transformed_cols, index=self.X.index)
            logger.info(f"Data transformed successfully. Shape: {self.X.shape}")
            return self.X, self.y
        except Exception as e:
            logger.error(f"Error in fit_transform: {str(e)}")
            raise

    def save_transformer(self, output_path='transformer.pkl'):
        """
        Save the fitted transformer for prediction.
        """
        try:
            if self.transformer is None:
                raise ValueError("Transformer not fitted. Run fit_transform first.")
            
            logger.info(f"Saving transformer to {output_path}")
            joblib.dump(self.transformer, output_path)
            logger.info("Transformer saved successfully")
        except Exception as e:
            logger.error(f"Error saving transformer: {str(e)}")
            raise

    def save_transformed_data(self, output_path='transformed_data.csv'):
        """
        Save the transformed data along with the target.
        """
        try:
            logger.info(f"Saving transformed data to {output_path}")
            transformed_df = self.X.copy()
            transformed_df[self.target_col] = self.y
            transformed_df.to_csv(output_path, index=False)
            logger.info("Transformed data saved successfully")
        except Exception as e:
            logger.error(f"Error saving transformed data: {str(e)}")
            raise

    def run_pipeline(self, transformer_path='transformer.pkl', output_path='transformed_data.csv'):
        """
        Run the full data transformation pipeline.
        """
        try:
            self.load_data()
            self.fit_transform()
            self.save_transformer(transformer_path)
            self.save_transformed_data(output_path)
            logger.info("Data transformation pipeline completed successfully")
            return self.X, self.y
        except Exception as e:
            logger.error(f"Data transformation pipeline failed: {str(e)}")
            raise

# Example Usage
if __name__ == "__main__":
    # Specify input file from ingestion step
    input_path = 'cleaned_data.csv'  # Replace with your actual file path
    
    # Initialize and run the pipeline
    transformation = DataTransformation(input_path)
    X_transformed, y = transformation.run_pipeline()
    
    # Print summary
    print("\nTransformed Data Summary:")
    print(X_transformed.info())
    print("\nFirst few rows:")
    print(X_transformed.head())
    print("\nTarget sample:")
    print(y.head())
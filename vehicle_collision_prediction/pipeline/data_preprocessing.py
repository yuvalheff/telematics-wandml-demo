from typing import Optional
import pandas as pd
import numpy as np
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from vehicle_collision_prediction.config import DataConfig


class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: DataConfig):
        self.config: DataConfig = config
        self.label_encoder = None
        self.imputers = {}
        self.scaler = None
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataProcessor':
        """
        Fit the data processor to the data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        DataProcessor: The fitted processor.
        """
        X_processed = X.copy()
        
        # 1. Remove identifier columns
        for col in self.config.remove_columns:
            if col in X_processed.columns:
                X_processed = X_processed.drop(columns=[col])
        
        # 2. Fit categorical encoder
        for col in self.config.categorical_columns:
            if col in X_processed.columns:
                if self.label_encoder is None:
                    self.label_encoder = LabelEncoder()
                self.label_encoder.fit(X_processed[col].astype(str))
        
        # 3. Fit missing value imputers
        for col in self.config.imputation_columns:
            if col in X_processed.columns:
                imputer = SimpleImputer(strategy='median')
                imputer.fit(X_processed[[col]])
                self.imputers[col] = imputer
        
        # 4. Apply log transformation (no fitting needed)
        for col in self.config.log_transform_columns:
            if col in X_processed.columns:
                # Apply log1p transformation
                X_processed[col] = np.log1p(X_processed[col].clip(lower=0))
        
        # 5. Handle missing values after log transformation
        for col in self.config.imputation_columns:
            if col in X_processed.columns and col in self.imputers:
                X_processed[col] = self.imputers[col].transform(X_processed[[col]]).flatten()
        
        # 6. Apply categorical encoding
        for col in self.config.categorical_columns:
            if col in X_processed.columns:
                X_processed[col] = self.label_encoder.transform(X_processed[col].astype(str))
        
        # 7. Fit scaler on processed features
        scaling_cols = [col for col in self.config.scaling_columns if col in X_processed.columns]
        if scaling_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(X_processed[scaling_cols])
        
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data based on the fitted processors.

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features.
        """
        if not self.fitted:
            raise ValueError("DataProcessor must be fitted before transform")
        
        X_processed = X.copy()
        
        # 1. Remove identifier columns
        for col in self.config.remove_columns:
            if col in X_processed.columns:
                X_processed = X_processed.drop(columns=[col])
        
        # 2. Apply log transformation
        for col in self.config.log_transform_columns:
            if col in X_processed.columns:
                X_processed[col] = np.log1p(X_processed[col].clip(lower=0))
        
        # 3. Handle missing values
        for col in self.config.imputation_columns:
            if col in X_processed.columns and col in self.imputers:
                X_processed[col] = self.imputers[col].transform(X_processed[[col]]).flatten()
        
        # 4. Apply categorical encoding
        for col in self.config.categorical_columns:
            if col in X_processed.columns:
                X_processed[col] = self.label_encoder.transform(X_processed[col].astype(str))
        
        # 5. Apply scaling
        scaling_cols = [col for col in self.config.scaling_columns if col in X_processed.columns]
        if scaling_cols and self.scaler is not None:
            X_processed[scaling_cols] = self.scaler.transform(X_processed[scaling_cols])
        
        return X_processed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> pd.DataFrame:
        """
        Fit and transform the input data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        pd.DataFrame: The transformed features.
        """
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """
        Save the data processor as an artifact

        Parameters:
        path (str): The file path to save the processor.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> 'DataProcessor':
        """
        Load the data processor from a saved artifact.

        Parameters:
        path (str): The file path to load the processor from.

        Returns:
        DataProcessor: The loaded data processor.
        """
        with open(path, 'rb') as f:
            loaded_processor = pickle.load(f)
        return loaded_processor

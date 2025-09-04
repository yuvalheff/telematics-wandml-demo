from typing import Optional
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from vehicle_collision_prediction.config import DataConfig


class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: DataConfig):
        self.config: DataConfig = config

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataProcessor':
        """
        Fit the feature processor to the data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        DataProcessor: The fitted processor.
        """
        # Implement fitting logic if necessary
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data based on the configuration.

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features.
        """
        # Implement transformation logic based on the config
        # For example, you might want to select specific columns or apply transformations
        # Here we just return the input DataFrame as a placeholder
        return X

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
        Save the feature processor as an artifact

        Parameters:
        path (str): The file path to save the configuration.
        """

    def load(self, path: str) -> 'DataProcessor':
        """
        Load the feature processor from a saved artifact.

        Parameters:
        path (str): The file path to load the configuration from.

        Returns:
        FeatureProcessor: The loaded feature processor.
        """
        # Implement loading logic if necessary
        return self

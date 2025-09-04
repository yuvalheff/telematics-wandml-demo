from typing import Optional
import pandas as pd
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from vehicle_collision_prediction.config import FeaturesConfig


class FeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: FeaturesConfig):
        self.config: FeaturesConfig = config
        self.risk_scaler = None
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureProcessor':
        """
        Fit the feature processor to the data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        FeatureProcessor: The fitted processor.
        """
        X_processed = X.copy()
        
        # Create features first to fit scaler for composite risk score
        X_processed = self._create_features(X_processed)
        
        # Fit scaler for composite risk score if enabled
        if self.config.create_composite_risk_score:
            risk_features = ['time_speeding_hours', 'time_phoneuse_hours', 'night_drive_hrs']
            available_risk_features = [col for col in risk_features if col in X_processed.columns]
            
            if available_risk_features:
                self.risk_scaler = StandardScaler()
                self.risk_scaler.fit(X_processed[available_risk_features])
        
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input features based on the configuration.

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features.
        """
        if not self.fitted:
            raise ValueError("FeatureProcessor must be fitted before transform")
        
        X_processed = X.copy()
        
        # Create engineered features
        X_processed = self._create_features(X_processed)
        
        # Remove original features if configured to do so
        if not self.config.keep_original_features:
            # Define original feature columns to potentially remove
            original_features = ['count_trip', 'miles', 'drive_hours', 'count_brakes', 
                               'count_accelarations', 'time_speeding_hours', 'time_phoneuse_hours',
                               'highway_miles', 'night_drive_hrs', 'maximum_speed']
            
            # Only remove features that exist in the dataframe
            features_to_remove = [col for col in original_features if col in X_processed.columns]
            X_processed = X_processed.drop(columns=features_to_remove)
        
        return X_processed

    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features based on configuration.
        
        Parameters:
        X (pd.DataFrame): Input features
        
        Returns:
        pd.DataFrame: Features with engineered columns added
        """
        X_processed = X.copy()
        
        # 1. Create exposure-normalized features
        if self.config.create_exposure_normalized:
            # Handle division by zero by setting result to 0 when count_trip is 0
            if 'count_trip' in X_processed.columns:
                # Brakes per trip
                if 'count_brakes' in X_processed.columns:
                    X_processed['brakes_per_trip'] = np.where(
                        X_processed['count_trip'] == 0, 0,
                        X_processed['count_brakes'] / X_processed['count_trip']
                    )
                
                # Accelerations per trip
                if 'count_accelarations' in X_processed.columns:
                    X_processed['accel_per_trip'] = np.where(
                        X_processed['count_trip'] == 0, 0,
                        X_processed['count_accelarations'] / X_processed['count_trip']
                    )
                
                # Miles per trip
                if 'miles' in X_processed.columns:
                    X_processed['miles_per_trip'] = np.where(
                        X_processed['count_trip'] == 0, 0,
                        X_processed['miles'] / X_processed['count_trip']
                    )
                
                # Hours per trip
                if 'drive_hours' in X_processed.columns:
                    X_processed['hours_per_trip'] = np.where(
                        X_processed['count_trip'] == 0, 0,
                        X_processed['drive_hours'] / X_processed['count_trip']
                    )
        
        # 2. Create risk behavior ratios
        if self.config.create_risk_ratios:
            # Highway ratio
            if 'highway_miles' in X_processed.columns and 'miles' in X_processed.columns:
                X_processed['highway_ratio'] = np.where(
                    X_processed['miles'] == 0, 0,
                    X_processed['highway_miles'] / X_processed['miles']
                )
            
            # Speeding ratio
            if 'time_speeding_hours' in X_processed.columns and 'drive_hours' in X_processed.columns:
                X_processed['speeding_ratio'] = np.where(
                    X_processed['drive_hours'] == 0, 0,
                    X_processed['time_speeding_hours'] / X_processed['drive_hours']
                )
            
            # Phone use ratio
            if 'time_phoneuse_hours' in X_processed.columns and 'drive_hours' in X_processed.columns:
                X_processed['phone_ratio'] = np.where(
                    X_processed['drive_hours'] == 0, 0,
                    X_processed['time_phoneuse_hours'] / X_processed['drive_hours']
                )
        
        # 3. Create composite risk score
        if self.config.create_composite_risk_score:
            risk_features = ['time_speeding_hours', 'time_phoneuse_hours', 'night_drive_hrs']
            available_risk_features = [col for col in risk_features if col in X_processed.columns]
            
            if available_risk_features and self.risk_scaler is not None:
                # Standardize risk features and sum them
                standardized_risk = self.risk_scaler.transform(X_processed[available_risk_features])
                X_processed['risk_score'] = np.sum(standardized_risk, axis=1)
            elif available_risk_features:
                # Fallback: simple sum if scaler not available
                X_processed['risk_score'] = X_processed[available_risk_features].sum(axis=1)
        
        return X_processed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> pd.DataFrame:
        """
        Fit and transform the input features.

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
        path (str): The file path to save the processor.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> 'FeatureProcessor':
        """
        Load the feature processor from a saved artifact.

        Parameters:
        path (str): The file path to load the processor from.

        Returns:
        FeatureProcessor: The loaded feature processor.
        """
        with open(path, 'rb') as f:
            loaded_processor = pickle.load(f)
        return loaded_processor

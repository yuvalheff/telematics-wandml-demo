"""
ML Pipeline for Vehicle Collision Prediction

Complete pipeline that combines data preprocessing, feature engineering, and model prediction
for MLflow model registry deployment.
"""

import pandas as pd
import numpy as np
import pickle
from typing import Optional

from vehicle_collision_prediction.pipeline.data_preprocessing import DataProcessor
from vehicle_collision_prediction.pipeline.feature_preprocessing import FeatureProcessor
from vehicle_collision_prediction.pipeline.model import ModelWrapper


class ModelPipeline:
    """
    Complete ML pipeline for vehicle collision prediction.
    
    Combines data preprocessing, feature engineering, and model prediction
    into a single deployable unit.
    """
    
    def __init__(self, data_processor: Optional[DataProcessor] = None,
                 feature_processor: Optional[FeatureProcessor] = None,
                 model: Optional[ModelWrapper] = None):
        """
        Initialize the pipeline components.
        
        Parameters:
        data_processor: Fitted data preprocessing component
        feature_processor: Fitted feature engineering component
        model: Fitted model component
        """
        self.data_processor = data_processor
        self.feature_processor = feature_processor
        self.model = model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict collision class labels for input data.
        
        Parameters:
        X: Raw input features (before preprocessing)
        
        Returns:
        np.ndarray: Predicted class labels (0, 1, or 2 collisions)
        """
        # Validate pipeline components
        if self.data_processor is None:
            raise ValueError("Data processor not initialized")
        if self.feature_processor is None:
            raise ValueError("Feature processor not initialized")
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Apply full pipeline
        X_processed = self._preprocess_data(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict collision class probabilities for input data.
        
        Parameters:
        X: Raw input features (before preprocessing)
        
        Returns:
        np.ndarray: Predicted class probabilities
        """
        # Validate pipeline components
        if self.data_processor is None:
            raise ValueError("Data processor not initialized")
        if self.feature_processor is None:
            raise ValueError("Feature processor not initialized")
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Apply full pipeline
        X_processed = self._preprocess_data(X)
        return self.model.predict_proba(X_processed)
    
    def _preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply complete data preprocessing pipeline.
        
        Parameters:
        X: Raw input features
        
        Returns:
        pd.DataFrame: Fully processed features ready for model prediction
        """
        # Step 1: Apply data preprocessing (handle missing values, scaling, encoding)
        X_processed = self.data_processor.transform(X)
        
        # Step 2: Apply feature engineering (create new features)
        X_processed = self.feature_processor.transform(X_processed)
        
        return X_processed
    
    def get_feature_names(self) -> list:
        """
        Get names of features that the model expects.
        
        Returns:
        list: Feature names after preprocessing and engineering
        """
        if self.data_processor is None or self.feature_processor is None:
            return []
        
        # This would need to be implemented to track feature names through the pipeline
        # For now, return empty list as placeholder
        return []
    
    def save_pipeline(self, directory_path: str) -> None:
        """
        Save the complete pipeline to disk.
        
        Parameters:
        directory_path: Directory to save pipeline components
        """
        import os
        os.makedirs(directory_path, exist_ok=True)
        
        if self.data_processor is not None:
            self.data_processor.save(os.path.join(directory_path, "data_processor.pkl"))
        
        if self.feature_processor is not None:
            self.feature_processor.save(os.path.join(directory_path, "feature_processor.pkl"))
        
        if self.model is not None:
            self.model.save(os.path.join(directory_path, "model.pkl"))
    
    @classmethod
    def load_pipeline(cls, directory_path: str) -> 'ModelPipeline':
        """
        Load a complete pipeline from disk.
        
        Parameters:
        directory_path: Directory containing pipeline components
        
        Returns:
        ModelPipeline: Loaded pipeline ready for prediction
        """
        import os
        
        # Load data processor
        data_processor_path = os.path.join(directory_path, "data_processor.pkl")
        data_processor = None
        if os.path.exists(data_processor_path):
            with open(data_processor_path, 'rb') as f:
                data_processor = pickle.load(f)
        
        # Load feature processor
        feature_processor_path = os.path.join(directory_path, "feature_processor.pkl")
        feature_processor = None
        if os.path.exists(feature_processor_path):
            with open(feature_processor_path, 'rb') as f:
                feature_processor = pickle.load(f)
        
        # Load model
        model_path = os.path.join(directory_path, "model.pkl")
        model = None
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        return cls(data_processor=data_processor,
                   feature_processor=feature_processor,
                   model=model)

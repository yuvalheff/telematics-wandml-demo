import pandas as pd

from vehicle_collision_prediction.config import ModelConfig


class ModelWrapper:
    def __init__(self, config: ModelConfig):
        self.config: ModelConfig = config
        # self.model = ...

    def fit(self, X: pd.Series, y: pd.Series):
        """
        Fit the classifier to the training data.

        Parameters:
        X: Training features.
        y: Target labels.

        Returns:
        self: Fitted classifier.
        """
        # implement fitting logic
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict class labels for the input features.

        Parameters:
        X: Input features to predict.

        Returns:
        np.ndarray: Predicted class labels.
        """

        # implement prediction logic
        return

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict class probabilities for the input features.

        Parameters:
        X: Input features to predict probabilities.

        Returns:
        np.ndarray: Predicted class probabilities.
        """

        # implement probability prediction logic
        return
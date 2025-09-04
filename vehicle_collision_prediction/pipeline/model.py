import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import average_precision_score, make_scorer
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from vehicle_collision_prediction.config import ModelConfig


def pr_auc_macro_scorer(estimator, X, y_true):
    """Custom scorer for macro-averaged PR-AUC"""
    # Get probabilities from estimator
    y_proba = estimator.predict_proba(X)
    
    classes = np.unique(y_true)
    pr_aucs = []
    
    for i, cls in enumerate(classes):
        # Create binary targets for this class
        y_binary = (y_true == cls).astype(int)
        # Get probabilities for this class
        if y_proba.ndim == 1:
            y_prob_cls = y_proba if len(classes) == 2 else y_proba
        else:
            y_prob_cls = y_proba[:, i] if i < y_proba.shape[1] else np.zeros(len(y_true))
        
        # Calculate PR-AUC for this class
        pr_auc = average_precision_score(y_binary, y_prob_cls)
        pr_aucs.append(pr_auc)
    
    return np.mean(pr_aucs)


class ModelWrapper:
    def __init__(self, config: ModelConfig):
        self.config: ModelConfig = config
        self.model = None
        self.feature_selector = None
        self.fitted = False

    def _create_model(self, model_type: str, params: dict):
        """Create model instance based on type and parameters"""
        if model_type == "RandomForestClassifier":
            return RandomForestClassifier(**params)
        elif model_type == "XGBClassifier":
            if XGBClassifier is None:
                raise ImportError("XGBoost not available")
            # Convert class_weight to sample_weight for XGBoost
            xgb_params = params.copy()
            if 'class_weight' in xgb_params:
                xgb_params.pop('class_weight')
            return XGBClassifier(**xgb_params)
        elif model_type == "GradientBoostingClassifier":
            return GradientBoostingClassifier(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the classifier to the training data.

        Parameters:
        X: Training features.
        y: Target labels.

        Returns:
        self: Fitted classifier.
        """
        X_features = X.copy()
        
        # Create base model
        base_model = self._create_model(self.config.model_type, self.config.model_params)
        
        # Feature selection if enabled
        if self.config.feature_selection.get('enabled', False):
            method = self.config.feature_selection.get('method', 'RFECV')
            min_features = self.config.feature_selection.get('min_features', 5)
            
            if method == 'RFECV':
                # Use custom PR-AUC scorer  
                self.feature_selector = RFECV(
                    estimator=base_model,
                    min_features_to_select=min_features,
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                    scoring=pr_auc_macro_scorer
                )
                X_features = pd.DataFrame(
                    self.feature_selector.fit_transform(X_features, y),
                    columns=X_features.columns[self.feature_selector.support_],
                    index=X_features.index
                )
        
        # Hyperparameter tuning if enabled
        if self.config.hyperparameter_tuning.get('enabled', False):
            param_grid = self.config.hyperparameter_tuning.get('param_grid', {})
            
            if param_grid:
                # Use custom PR-AUC scorer
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                    scoring=pr_auc_macro_scorer,
                    n_jobs=-1
                )
                grid_search.fit(X_features, y)
                self.model = grid_search.best_estimator_
            else:
                self.model = base_model
                self.model.fit(X_features, y)
        else:
            self.model = base_model
            self.model.fit(X_features, y)
        
        self.fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for the input features.

        Parameters:
        X: Input features to predict.

        Returns:
        np.ndarray: Predicted class labels.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predict")
        
        X_features = X.copy()
        
        # Apply feature selection if it was used during training
        if self.feature_selector is not None:
            X_features = pd.DataFrame(
                self.feature_selector.transform(X_features),
                columns=X_features.columns[self.feature_selector.support_],
                index=X_features.index
            )
        
        return self.model.predict(X_features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for the input features.

        Parameters:
        X: Input features to predict probabilities.

        Returns:
        np.ndarray: Predicted class probabilities.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predict_proba")
        
        X_features = X.copy()
        
        # Apply feature selection if it was used during training
        if self.feature_selector is not None:
            X_features = pd.DataFrame(
                self.feature_selector.transform(X_features),
                columns=X_features.columns[self.feature_selector.support_],
                index=X_features.index
            )
        
        return self.model.predict_proba(X_features)

    def get_feature_importance(self) -> dict:
        """Get feature importance from the fitted model"""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            if self.feature_selector is not None:
                # Map back to original feature names
                feature_names = self.feature_selector.feature_names_in_[self.feature_selector.support_]
            else:
                feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
            
            return dict(zip(feature_names, self.model.feature_importances_))
        else:
            return {}

    def save(self, path: str) -> None:
        """Save the model wrapper as an artifact"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> 'ModelWrapper':
        """Load the model wrapper from a saved artifact"""
        with open(path, 'rb') as f:
            loaded_model = pickle.load(f)
        return loaded_model
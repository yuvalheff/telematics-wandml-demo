import os
import pandas as pd
import numpy as np
import mlflow
import sklearn
from pathlib import Path

from vehicle_collision_prediction.pipeline.feature_preprocessing import FeatureProcessor
from vehicle_collision_prediction.pipeline.data_preprocessing import DataProcessor
from vehicle_collision_prediction.pipeline.model import ModelWrapper
from vehicle_collision_prediction.config import Config
from vehicle_collision_prediction.model_pipeline import ModelPipeline
from experiment_scripts.evaluation import ModelEvaluator

DEFAULT_CONFIG = str(Path(__file__).parent / 'config.yaml')


class Experiment:
    def __init__(self):
        self._config = Config.from_yaml(DEFAULT_CONFIG)

    def run(self, train_dataset_path, test_dataset_path, output_dir, seed=42):
        """
        Execute the complete ML experiment pipeline.
        
        Parameters:
        train_dataset_path: Path to training data CSV
        test_dataset_path: Path to test data CSV  
        output_dir: Directory to save all outputs
        seed: Random seed for reproducibility
        
        Returns:
        dict: Experiment results with required format
        """
        # Set random seeds
        np.random.seed(seed)
        
        # Create output directories
        model_artifacts_dir = os.path.join(output_dir, "output", "model_artifacts") 
        general_artifacts_dir = os.path.join(output_dir, "output", "general_artifacts")
        os.makedirs(model_artifacts_dir, exist_ok=True)
        os.makedirs(general_artifacts_dir, exist_ok=True)
        
        try:
            # Load data
            print("Loading training and test datasets...")
            train_data = pd.read_csv(train_dataset_path)
            test_data = pd.read_csv(test_dataset_path)
            
            # Separate features and target
            target_col = 'collisions'
            X_train = train_data.drop(columns=[target_col])
            y_train = train_data[target_col]
            X_test = test_data.drop(columns=[target_col])
            y_test = test_data[target_col]
            
            print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
            print(f"Target distribution in training: {y_train.value_counts().sort_index()}")
            
            # Initialize pipeline components
            print("Initializing pipeline components...")
            data_processor = DataProcessor(self._config.data_prep)
            feature_processor = FeatureProcessor(self._config.feature_prep)
            model_wrapper = ModelWrapper(self._config.model)
            
            # Fit data preprocessing pipeline
            print("Fitting data preprocessing pipeline...")
            data_processor.fit(X_train)
            X_train_processed = data_processor.transform(X_train)
            X_test_processed = data_processor.transform(X_test)
            
            print(f"Data processed - Train shape: {X_train_processed.shape}, Test shape: {X_test_processed.shape}")
            
            # Fit feature engineering pipeline  
            print("Fitting feature engineering pipeline...")
            feature_processor.fit(X_train_processed)
            X_train_features = feature_processor.transform(X_train_processed)
            X_test_features = feature_processor.transform(X_test_processed)
            
            print(f"Features engineered - Train shape: {X_train_features.shape}, Test shape: {X_test_features.shape}")
            
            # Train model
            print("Training model...")
            model_wrapper.fit(X_train_features, y_train)
            
            print("Model training completed!")
            
            # Evaluate model
            print("Evaluating model on test set...")
            evaluator = ModelEvaluator(self._config.model_evaluation)
            evaluation_metrics = evaluator.evaluate_model(model_wrapper, X_test_features, y_test, os.path.join(output_dir, "output"))
            
            print(f"Primary metric (Macro PR-AUC): {evaluation_metrics['macro_pr_auc']:.4f}")
            
            # Save individual model artifacts
            print("Saving model artifacts...")
            data_processor.save(os.path.join(model_artifacts_dir, "data_processor.pkl"))
            feature_processor.save(os.path.join(model_artifacts_dir, "feature_processor.pkl"))
            model_wrapper.save(os.path.join(model_artifacts_dir, "trained_models.pkl"))
            
            # Create and save complete pipeline for MLflow
            print("Creating complete pipeline for MLflow...")
            pipeline = ModelPipeline(
                data_processor=data_processor,
                feature_processor=feature_processor, 
                model=model_wrapper
            )
            
            # Test pipeline end-to-end with sample data
            print("Testing pipeline end-to-end...")
            sample_input = X_test.head(5)
            sample_predictions = pipeline.predict(sample_input)
            sample_probabilities = pipeline.predict_proba(sample_input)
            print(f"Pipeline test successful - Sample predictions: {sample_predictions}")
            
            # Save and log MLflow model
            print("Saving MLflow model...")
            output_path = os.path.join(model_artifacts_dir, "mlflow_model")
            relative_path_for_return = "output/model_artifacts/mlflow_model/"
            
            # Clean up existing directory if it exists
            import shutil
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            
            # Always save the model to local path for harness validation
            print(f"💾 Saving model to local disk for harness: {output_path}")
            mlflow.sklearn.save_model(
                pipeline,
                path=output_path,
                signature=mlflow.models.infer_signature(sample_input, pipeline.predict(sample_input))
            )
            
            # If an MLflow run ID is provided, reconnect and log the model as an artifact
            active_run_id = "e90f0c32bd13465ba2dfcc840831071f"
            logged_model_uri = None
            if active_run_id and active_run_id != 'None' and active_run_id.strip():
                print(f"✅ Active MLflow run ID '{active_run_id}' detected. Reconnecting to log model as an artifact.")
                with mlflow.start_run(run_id=active_run_id):
                    logged_model_info = mlflow.sklearn.log_model(
                        pipeline,
                        artifact_path="model",
                        code_paths=["vehicle_collision_prediction"],
                        signature=mlflow.models.infer_signature(sample_input, pipeline.predict(sample_input))
                    )
                    logged_model_uri = logged_model_info.model_uri
            else:
                print("ℹ️ No active MLflow run ID provided. Skipping model logging.")
            
            # Prepare model artifacts list
            model_artifacts = [
                "data_processor.pkl",
                "feature_processor.pkl", 
                "trained_models.pkl",
                "mlflow_model/"
            ]
            
            # Prepare MLflow model info
            mlflow_model_info = {
                "model_path": relative_path_for_return,
                "logged_model_uri": logged_model_uri,
                "model_type": "sklearn",
                "task_type": "classification",
                "signature": {
                    "inputs": {col: str(dtype) for col, dtype in sample_input.dtypes.items()},
                    "outputs": {"predictions": "int64"}
                },
                "framework_version": sklearn.__version__,
                "input_example": sample_input.head(1).to_dict('records')[0]
            }
            
            print("Experiment completed successfully!")
            
            return {
                "metric_name": "macro_pr_auc",
                "metric_value": evaluation_metrics['macro_pr_auc'],
                "model_artifacts": model_artifacts,
                "mlflow_model_info": mlflow_model_info
            }
            
        except Exception as e:
            print(f"Error during experiment execution: {str(e)}")
            raise e